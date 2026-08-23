import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


def angles_lidar(angle_fovea_deg=45.0, rayons_fovea=96, n_rayons=128, demi_champ_deg=120.0):
    """
    Répartition angulaire des 128 rayons LiDAR, en radians, triée croissante.

    MOTIVATION (calcul fait, pas une intuition) : avec 128 rayons répartis
    uniformément sur 240°, l'espacement est de 1.89°. Un arbre (rayon 0.3m)
    vu à 20m ne fait que 1.72° de large -- soit MOINS qu'un intervalle entre
    deux rayons. Il peut donc passer entièrement entre deux rayons et rester
    totalement invisible. La limite est ~18m : au-delà, la détection d'un
    arbre devient une loterie qui dépend de son alignement exact avec un
    rayon. À ~21 m/s (vitesse observée), ça ne laisse que 0.87s de réaction
    dans le meilleur des cas, et parfois aucune détection du tout.

    Ici : on concentre `rayons_fovea` rayons sur le secteur avant
    (±`angle_fovea_deg`), et on répartit le reste sur la périphérie.
    Avec les défauts (96 rayons sur ±45°) : 0.947°/rayon devant, soit un
    arbre détectable jusqu'à 36m -- au-delà de la portée max (30m), donc
    fiable sur TOUTE la portée. La périphérie tombe à 4.7°/rayon, ce qui
    reste largement suffisant pour des murs (surfaces continues et immenses,
    aucun risque de passer entre deux rayons).

    Nombre total de rayons inchangé (128) -> observation toujours à 141
    dimensions, pas de changement d'architecture réseau.

    angle_fovea_deg=0 ou rayons_fovea=0 -> répartition uniforme (comportement
    historique, pour pouvoir comparer).
    """
    demi_champ = np.radians(demi_champ_deg)
    if angle_fovea_deg <= 0 or rayons_fovea <= 0 or rayons_fovea >= n_rayons:
        return np.linspace(-demi_champ, demi_champ, n_rayons)

    a = np.radians(angle_fovea_deg)
    n_cote = (n_rayons - rayons_fovea) // 2
    # Le fovéa inclut ses deux bornes (±a) ; les côtés les excluent pour
    # ne pas dupliquer un angle.
    fovea = np.linspace(-a, a, rayons_fovea)
    gauche = np.linspace(-demi_champ, -a, n_cote, endpoint=False)
    droite = np.linspace(a, demi_champ, n_cote + 1)[1:]
    return np.concatenate([gauche, fovea, droite])

import mujoco
import mujoco.viewer
import random
from stable_baselines3.common.env_checker import check_env
import time

class SpeedrunnerEnv(gym.Env):
    """
    v3 — randomisation de domaine (remplace le curriculum séquentiel figé) :
    1. Reward de proximité (dense) basée sur le LiDAR.
    2. Pénalité de mort rééquilibrée (2000 -> 100).
    3. Plafond de vitesse dans la reward (évite le reward hacking "fonce et meurs").
    4. NOUVEAU : densite_arbres et longueur_piste sont maintenant des PLAGES
       (min, max), tirées aléatoirement à CHAQUE reset() plutôt que fixées
       par étape. Le curriculum séquentiel entraînait un oubli catastrophique
       (validé empiriquement : succès ~0% en couloir vide après entraînement
       intensif sur une seule config avec obstacles) — cf papier Zurich 2026
       "Bridging Performance and Generalization in RL for Agile Flight".
       Chaque rollout contient maintenant un mélange de difficultés, donc le
       réseau ne peut plus se spécialiser sur une seule config et tout oublier
       du reste.
    """

    def __init__(self, longueur_piste_range=(30.0, 100.0), largeur_piste=10.0,
                 densite_arbres_range=(0.0, 0.8),
                 seuil_danger=0.3, coeff_danger=1.0, portee_lidar=20.0,
                 zone_securite=5.0, gear_roll_pitch=2.0, gear_yaw=1.0,
                 seuil_danger_sol=1.0, coeff_danger_sol=1.0, penalite_mort=100.0,
                 coeff_agressivite=0.02, coeff_saccade=0.01, coeff_orientation=0.0,
                 coeff_assiette=0.0, seuil_rotation_excessive=15.0,
                 coeff_penalite_distance=0.0,
                 angle_fovea=45.0, rayons_fovea=96,
                 vitesse_max=12.0, coeff_vitesse_max=0.0, n_stack=1):
        super(SpeedrunnerEnv, self).__init__()

        # NOUVEAU : gear configurable pour tester l'hypothèse gear/inertie.
        # Valeurs actuelles (2.0 / 1.0) donnent une accélération angulaire
        # théorique de ~1000 rad/s² à pleine commande (gear/diaginertia =
        # 2/0.002) -- beaucoup plus nerveux qu'un vrai quadricoptère, ce qui
        # peut expliquer la dérive de tangage/lacet observée sur les longs vols.
        self.gear_roll_pitch = gear_roll_pitch
        self.gear_yaw = gear_yaw

        self.longueur_piste_range = longueur_piste_range
        self.largeur_piste = largeur_piste
        self.densite_arbres_range = densite_arbres_range

        # Valeurs par défaut avant le premier reset() (seront re-tirées
        # aléatoirement dans reset(), cf plus bas)
        self.longueur_piste = longueur_piste_range[1]
        self.densite_arbres = densite_arbres_range[0]

        # NOUVEAU : rayon (en mètres) autour du spawn (0,0) où aucun arbre
        # n'est généré. Évite les épisodes injouables dès le départ à
        # forte densité (spawn entouré d'arbres avant même d'avoir bougé).
        self.zone_securite = zone_securite

        # Seuil de danger : en dessous de cette distance LiDAR normalisée (0-1),
        # on commence à pénaliser. Reste en proportion de portee_lidar, pas en mètres fixes.
        self.seuil_danger = seuil_danger
        self.coeff_danger = coeff_danger

        # NOUVEAU : pénalité de proximité au SOL, même principe que ci-dessus
        # mais sur l'altitude en mètres. Avant, voler au ras du sol ne coûtait
        # rien -> le drone a appris à piquer pour accélérer puis remonter au
        # dernier moment, une manœuvre gratuite qui finit par échouer
        # statistiquement sur les vols longs (observé : échec systématique
        # vers 300m en couloir vide, alors que rien ne devrait l'arrêter).
        self.seuil_danger_sol = seuil_danger_sol
        self.coeff_danger_sol = coeff_danger_sol

        # NOUVEAU : pénalité de mort configurable. Elle valait 100, réglée à
        # une époque où la reward de progrès était plafonnée (~1/step max).
        # Depuis qu'on a retiré ce plafond, mourir à 30m rapporte déjà 750
        # de progrès avant même la pénalité -- 100 ne pèse plus rien face à
        # ça, d'où le comportement "aucune peur de mourir" observé.
        self.penalite_mort = penalite_mort

        # NOUVEAU : coefficients de la pénalité de douceur de pilotage.
        # Valeurs de départ modestes (comme chez Swift, coefficients petits)
        # pour ne pas écraser la reward de progrès -- à ajuster empiriquement.
        self.coeff_agressivite = coeff_agressivite
        self.coeff_saccade = coeff_saccade

        # NOUVEAU : pénalité de CAP (pas de commande, l'angle réel du drone).
        # Différent de coeff_agressivite : celui-ci punit l'AMPLITUDE de la
        # commande de lacet à l'instant t (piloter à fond coûte cher, même
        # pour un virage ponctuel voulu). Celui-ci punit le fait d'ÊTRE
        # orienté loin de l'avant en continu, quelle que soit la commande
        # actuelle -- vise le cas où le drone s'installe dans un cap à 90°
        # et y reste (translation encore possible en combinant roll/pitch
        # dans ce référentiel tourné, donc rien dans l'ancienne reward ne
        # décourageait ça). Un virage bref pour esquiver coûte peu (peu de
        # steps à cet angle) ; un cap de travers soutenu coûte cher (accumulé
        # sur toute la durée où il y reste).
        self.coeff_orientation = coeff_orientation
        self.coeff_assiette = coeff_assiette

        # NOUVEAU : coefficient de pénalité de mort proportionnelle à la
        # distance. DÉSACTIVÉ par défaut (0.0) : testé à 30 sur sac11, a
        # provoqué un effondrement du comportement -- l'agent a appris que
        # ne PAS avancer était plus sûr qu'avancer (épisodes de 500-940
        # steps à faire du surplace, ep_rew_mean tombé autour de 0, et
        # surtout actor_loss passé POSITIF pour la première fois du projet,
        # signe que le critique voyait la quasi-totalité des états comme
        # de valeur négative). Confirme la méfiance d'Alex sur les grosses
        # pénalités, cohérente avec la littérature qu'il avait lue.
        self.coeff_penalite_distance = coeff_penalite_distance

        # NOUVEAU : répartition angulaire des rayons LiDAR (cf docstring de
        # angles_lidar plus haut pour le calcul qui motive ça). Calculée une
        # seule fois ici, pas à chaque step.
        self.angle_fovea = angle_fovea
        self.rayons_fovea = rayons_fovea
        self.angles_lidar = angles_lidar(angle_fovea_deg=angle_fovea,
                                          rayons_fovea=rayons_fovea)

        # NOUVEAU : vitesse cible et pénalité associée. DÉSACTIVÉ par défaut
        # (coeff_vitesse_max=0.0) -- 12 m/s est une estimation de départ tirée
        # de la littérature (état de l'art en évitement réactif : 7-10 m/s),
        # pas une valeur validée chez nous. À activer explicitement.
        self.vitesse_max = vitesse_max
        self.coeff_vitesse_max = coeff_vitesse_max

        # NOUVEAU : empilement de trames, implémenté DANS l'environnement
        # (pas via VecFrameStack de SB3). VecFrameStack ne recalcule pas
        # correctement la "terminal_observation" utilisée par le replay
        # buffer des algos off-policy (SAC) à la fin de chaque épisode --
        # bug connu de SB3, fait planter model.learn() dès qu'un épisode se
        # termine (ValueError: shape mismatch). En empilant nous-mêmes, la
        # trame terminale renvoyée par step() est déjà à la bonne taille,
        # aucune interaction possible avec ce mécanisme.
        self.n_stack = max(1, n_stack)
        self._historique = deque(maxlen=self.n_stack)

        # NOUVEAU : seuil de "rotation excessive" configurable -- avant figé
        # à 15.0 rad/s sans jamais avoir été vérifié empiriquement. Objectif :
        # tester si ce seuil coupe des esquives serrées mais réussies (mort
        # artificielle) ou détecte une vraie perte de contrôle (mort réelle,
        # juste un peu retardée). Défaut inchangé (15.0) pour ne rien casser
        # ailleurs -- ne se teste qu'explicitement, à l'évaluation.
        self.seuil_rotation_excessive = seuil_rotation_excessive

        # Portée LiDAR remise à 20m (valeur connue/testée) — le passage à 35m
        # reste un levier à part, testé séparément une fois la randomisation validée.
        self.portee_lidar = portee_lidar

        #[Gaz, Roulis, Tangage, Lacet] -- ordre = ordre des actuateurs dans le XML
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 1 (Altitude) + 3 (Vit. Linéaire) + 3 (Vit. Angulaire) + 4 (Quat) + 128 (LiDAR) = 139
        # NOUVEAU : +2 dimensions -- distance restante et position latérale
        # normalisées. Avant, le drone n'avait aucune idée d'où il en était
        # sur la piste, seulement ce que le LiDAR voit localement à l'instant
        # présent. Swift donne la position relative de la prochaine porte à
        # son agent -- on n'avait jamais donné l'équivalent au nôtre.
        # NOUVEAU : taille multipliée par n_stack (141 par trame). n_stack=1
        # -> 141, comportement strictement inchangé.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(141 * self.n_stack,), dtype=np.float32)

        self.model = None
        self.data = None
        self.viewer = None
        self._viewer_model = None
        self._dernier_lidar_origine = None
        self._dernier_lidar_directions = []
        self._dernier_lidar_distances = []

    def _generer_piste_xml(self):
        """Génère le code XML de MuJoCo avec une forêt aléatoire"""
        surface = self.longueur_piste * (self.largeur_piste * 2)
        nb_arbres = int((surface / 10) * self.densite_arbres)

        xml_debut = f"""
        <mujoco>
            <option timestep="0.002" gravity="0 0 -9.81"/>
            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="512" height="512"/>
                <material name="matplane" texture="texplane"/>
            </asset>
            <worldbody>
                <light pos="0 {self.longueur_piste/2} 10" dir="0 0 -1" diffuse="1 1 1"/>
                <geom name="sol" type="plane" pos="0 {self.longueur_piste/2} 0" size="{self.largeur_piste + 5} {self.longueur_piste/2 + 5} 0.1" material="matplane"/>

                <body name="drone_body" pos="0 0 3">
                    <joint type="free"/>
                    <inertial pos="0 0 0" mass="0.8" diaginertia="0.002 0.002 0.002"/>
                    <geom type="box" size="0.1 0.1 0.05" rgba="0.18 0.18 0.22 1" mass="0.8"/>
                    <geom type="box" pos="0 0.1 0" size="0.05 0.05 0.05" rgba="1 0.25 0.1 1"/>

                    <!-- HABILLAGE VISUEL UNIQUEMENT (bras + rotors).
                         Tous ces geoms ont contype="0" conaffinity="0" : ils ne
                         participent à AUCUNE collision, donc les causes de mort
                         (arbre/mur/sol) sont strictement inchangées.
                         La dynamique est inchangée aussi : le tag <inertial>
                         ci-dessus fixe explicitement masse et inertie, donc
                         MuJoCo ignore toute inertie qui serait dérivée des geoms.
                         Le LiDAR n'est pas affecté non plus : mj_ray exclut déjà
                         tout le corps du drone via bodyexclude=drone_id.
                         Conclusion : purement cosmétique, un modèle entraîné
                         avant cet ajout reste valable tel quel. -->

                    <!-- Bras en X, du centre vers chaque rotor -->
                    <geom type="capsule" fromto="0 0 0.01  0.17 0.17 0.03" size="0.012"
                          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="capsule" fromto="0 0 0.01 -0.17 0.17 0.03" size="0.012"
                          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="capsule" fromto="0 0 0.01  0.17 -0.17 0.03" size="0.012"
                          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="capsule" fromto="0 0 0.01 -0.17 -0.17 0.03" size="0.012"
                          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>

                    <!-- Moteurs. AVANT (+Y) en orange, ARRIÈRE en sombre :
                         permet de lire l'orientation du drone en un coup d'oeil
                         dans une vidéo, ce qui est impossible avec un cube gris. -->
                    <geom type="cylinder" pos="0.17 0.17 0.035" size="0.028 0.022"
                          rgba="1 0.35 0.12 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="-0.17 0.17 0.035" size="0.028 0.022"
                          rgba="1 0.35 0.12 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="0.17 -0.17 0.035" size="0.028 0.022"
                          rgba="0.2 0.2 0.24 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="-0.17 -0.17 0.035" size="0.028 0.022"
                          rgba="0.2 0.2 0.24 1" contype="0" conaffinity="0" mass="0"/>

                    <!-- Disques d'hélices, translucides : évoquent la rotation
                         sans avoir à animer quoi que ce soit. -->
                    <geom type="cylinder" pos="0.17 0.17 0.06" size="0.10 0.003"
                          rgba="0.6 0.75 0.95 0.28" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="-0.17 0.17 0.06" size="0.10 0.003"
                          rgba="0.6 0.75 0.95 0.28" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="0.17 -0.17 0.06" size="0.10 0.003"
                          rgba="0.6 0.75 0.95 0.28" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="-0.17 -0.17 0.06" size="0.10 0.003"
                          rgba="0.6 0.75 0.95 0.28" contype="0" conaffinity="0" mass="0"/>

                    <site name="center_of_mass" pos="0 0 0" size="0.01"/>
                </body>
        """

        xml_arbres = ""
        # NOUVEAU : transition progressive au lieu d'une coupure nette.
        # Un cutoff dur créait un effet "falaise" -- juste après la limite,
        # la densité repart à fond, donc des arbres pouvaient se masser
        # juste au bord, techniquement hors zone mais toujours injouable.
        # Ici, la probabilité de retirer un arbre décroît linéairement de
        # 100% (à zone_securite) à 0% (à zone_securite + marge_transition).
        marge_transition = 5.0
        arbres_places = 0
        tentatives = 0
        max_tentatives = nb_arbres * 20
        while arbres_places < nb_arbres and tentatives < max_tentatives:
            tentatives += 1
            x = random.uniform(-self.largeur_piste, self.largeur_piste)
            y = random.uniform(0.0, self.longueur_piste)
            distance_spawn = (x * x + y * y) ** 0.5

            if distance_spawn < self.zone_securite:
                continue  # toujours interdit, comme avant
            elif distance_spawn < self.zone_securite + marge_transition:
                proba_retrait = 1.0 - (distance_spawn - self.zone_securite) / marge_transition
                if random.random() < proba_retrait:
                    continue  # retiré selon la probabilité décroissante

            xml_arbres += f'                <geom type="cylinder" pos="{x:.2f} {y:.2f} 5.0" size="0.3 5.0" rgba="0.7 0.7 0.7 1"/>\n'
            arbres_places += 1

        xml_fin = f"""
                <!-- NOUVEAU : murs latéraux en vraie géométrie, collidables et
                     donc visibles par le LiDAR (mj_ray les détecte comme
                     n'importe quel arbre) -- au lieu d'un test de position
                     invisible au capteur. La mort au contact d'un mur passe
                     maintenant par la même détection de collision que les
                     arbres, pas par une règle codée en dur dans l'observation. -->
                <geom name="mur_gauche" type="box" pos="{-self.largeur_piste:.2f} {self.longueur_piste/2:.2f} 5.0"
                      size="0.1 {self.longueur_piste/2 + 1:.2f} 5.0" rgba="0.5 0.1 0.1 0.6"/>
                <geom name="mur_droit" type="box" pos="{self.largeur_piste:.2f} {self.longueur_piste/2:.2f} 5.0"
                      size="0.1 {self.longueur_piste/2 + 1:.2f} 5.0" rgba="0.5 0.1 0.1 0.6"/>
                <!-- Ligne d'arrivée retirée du monde physique. Plus de geom
                     à filtrer du LiDAR -- elle ne peut plus être vue puisqu'elle
                     n'existe plus. La victoire est détectée uniquement par
                     position (cf step() : xpos[1] >= longueur_piste), déjà
                     le mécanisme réel utilisé même quand le geom existait. -->
            </worldbody>
            <actuator>
            <motor name="thrust" site="center_of_mass" gear="0 0 20 0 0 0" ctrllimited="true" ctrlrange="-1 1"/>
            <motor name="roll" site="center_of_mass" gear="0 0 0 {self.gear_roll_pitch} 0 0" ctrllimited="true" ctrlrange="-1 1"/>
            <motor name="pitch" site="center_of_mass" gear="0 0 0 0 {self.gear_roll_pitch} 0" ctrllimited="true" ctrlrange="-1 1"/>
            <motor name="yaw" site="center_of_mass" gear="0 0 0 0 0 {self.gear_yaw}" ctrllimited="true" ctrlrange="-1 1"/>
        </actuator>
        </mujoco>
        """

        return xml_debut + xml_arbres + xml_fin

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        # NOUVEAU : randomisation de domaine — nouvelle config à chaque épisode
        self.densite_arbres = random.uniform(*self.densite_arbres_range)
        self.longueur_piste = random.uniform(*self.longueur_piste_range)

        xml_string = self._generer_piste_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        mujoco.mj_forward(self.model, self.data)

        # NOUVEAU : suivi du meilleur point atteint (pour la reward de progrès).
        # Remis à zéro à chaque épisode.
        self.meilleure_distance = 0.0

        # NOUVEAU : suivi de l'action précédente, pour la pénalité de douceur
        # de pilotage (coût sur les à-coups entre deux steps consécutifs).
        self.action_precedente = np.zeros(4, dtype=np.float32)

        # NOUVEAU : au reset, l'historique est rempli en répétant la toute
        # première trame n_stack fois (convention standard du frame stacking
        # -- pas d'historique réel disponible avant le premier step, donc on
        # ne peut pas faire mieux que dupliquer l'état initial).
        obs_brute = self._get_obs()
        self._historique = deque([obs_brute.copy() for _ in range(self.n_stack)], maxlen=self.n_stack)
        return self._empiler(), {}

    def step(self, action):
        action_reelle = np.array(action, dtype=np.float32)
        action_reelle[0] = (action[0] * 0.6) + 0.392

        self.data.ctrl[:] = action_reelle

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        altitude = obs[0]
        z_up = self.data.body("drone_body").xmat[8]
        vitesse_rotation = np.linalg.norm(self.data.qvel[3:6])

        en_collision = self.data.ncon > 0

        # NOUVEAU : identification de la cause précise de collision --
        # avant, "en_collision" mélangeait arbres/murs/sol dans un seul
        # booléen. On ne pouvait pas vérifier l'hypothèse "il meurt surtout
        # sur les murs" faute de données. Ici on regarde les contacts actifs
        # et on identifie ce qui est touché via les noms de géométries.
        cause_collision = None
        if en_collision:
            cause_collision = "arbre"  # par défaut : géométrie non nommée = arbre
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                nom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                nom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                noms = {nom1, nom2}
                if "mur_gauche" in noms or "mur_droit" in noms:
                    cause_collision = "mur"
                    break
                elif "sol" in noms:
                    cause_collision = "sol"
                    break

        # --- RÉCOMPENSES ---
        # SIMPLIFIÉ : coût de vie (-0.1) retiré -- seul terme jamais testé
        # seul, hérité du tout premier code sans avoir jamais été validé.
        reward = 0.0
        # NOUVEAU : reward de PROGRÈS, pas de vitesse. Récompense uniquement
        # le fait de dépasser le meilleur point jamais atteint dans l'épisode.
        # Reculer, s'arrêter, hésiter devient neutre (ni pénalisé ni récompensé)
        # au lieu d'être activement puni comme avec l'ancienne reward de vitesse
        # instantanée -> le drone peut apprendre à reculer/contourner un
        # cul-de-sac sans que ça lui coûte de la reward, contrairement aux
        # cerceaux (toujours franchissables) les arbres peuvent mener à des
        # impasses qu'il faut savoir abandonner.
        distance_actuelle = self.data.body("drone_body").xpos[1]
        progres = max(0.0, distance_actuelle - self.meilleure_distance)
        # PLUS DE PLAFOND DE VITESSE. Avec cette reward de progrès (seul le
        # dépassement du meilleur point compte), le total gagné pour
        # atteindre une distance X vaut 25*X, QUELLE QUE SOIT la vitesse --
        # contrairement à l'ancienne reward de vitesse instantanée, il n'y a
        # plus moyen de gonfler artificiellement la reward en accumulant des
        # lectures de vitesse élevée sans progrès net. Le plafond qu'on avait
        # mis ici recopiait un fix qui ne s'applique plus à cette formule --
        # il ne faisait que tuer toute incitation à aller vite au-delà de 2 m/s.
        reward += progres * 25.0
        if progres > 0:
            self.meilleure_distance = distance_actuelle

        # NOUVEAU : pénalité de proximité dense basée sur le LiDAR.
        # rayons = obs[13:141] (décalé de 2 à cause de distance_restante/position_laterale
        # insérées avant le LiDAR), déjà normalisés entre 0 (collé) et 1 (rien en vue).
        rayons = obs[13:141]
        min_dist = np.min(rayons)
        if min_dist < self.seuil_danger:
            # Pénalité qui augmente à mesure qu'on se rapproche du seuil de danger.
            reward -= self.coeff_danger * (self.seuil_danger - min_dist)

        # NOUVEAU : pénalité de proximité au sol -- même logique, sur l'altitude.
        # Empêche le "piqué gratuit" (accélérer en rasant le sol, remonter au
        # dernier moment) qui ne coûtait rien avant et finissait par échouer
        # statistiquement sur les vols longs.
        if altitude < self.seuil_danger_sol:
            reward -= self.coeff_danger_sol * (self.seuil_danger_sol - altitude)

        # NOUVEAU : pénalité de douceur de pilotage, inspirée du r_cmd de
        # Swift (Kaufmann et al. 2023) -- coûte à chaque step, indépendamment
        # de si ça tue ou pas, contrairement à toutes nos pénalités
        # précédentes qui ne s'activent qu'au moment de mourir. Deux
        # composantes : l'amplitude des commandes roll/pitch/yaw (piloter
        # à fond tout le temps coûte cher), et l'à-coup entre deux steps
        # (changer brutalement d'action coûte cher). Vise directement le
        # comportement "aucune retenue", pas juste sa conséquence.
        commandes_rotation = action_reelle[1:4]
        cout_agressivite = self.coeff_agressivite * np.sum(np.square(commandes_rotation))
        cout_saccade = self.coeff_saccade * np.sum(np.square(action_reelle - self.action_precedente))
        reward -= cout_agressivite + cout_saccade
        self.action_precedente = action_reelle.copy()

        # NOUVEAU : pénalité de cap -- calcul du yaw réel depuis le quaternion
        # (même formule que le diagnostic ajouté dans enjoy.py). Coûte à
        # chaque step proportionnellement au carré de l'écart au cap avant
        # (0 rad), donc un grand écart soutenu coûte disproportionnellement
        # plus qu'un petit écart bref -- cohérent avec "il peut tourner un
        # peu à gauche/droite pour esquiver, mais pas s'installer de travers."
        quat = self.data.body("drone_body").xquat
        w, x, y, z = quat
        yaw_rad = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cout_orientation = self.coeff_orientation * (yaw_rad ** 2)
        reward -= cout_orientation

        # NOUVEAU : pénalité d'assiette -- même principe que coeff_orientation
        # mais sur roll/pitch (angle réel du corps, pas la commande). Constat
        # sur sac9_agressivite : coeff_agressivite (qui punit la COMMANDE
        # active) n'a pas du tout réduit le vol "banqué en croisière" (roll
        # jusqu'à -73°, pitch jusqu'à +68°, soutenus en continu) -- un vol
        # banqué en régime établi ne demande quasi aucune commande active
        # pour se maintenir, donc ce levier ne pouvait structurellement pas
        # le toucher. Exactement la même erreur déjà faite et corrigée pour
        # le lacet (coeff_agressivite sur la commande de lacet n'avait rien
        # réglé ; coeff_orientation, sur l'angle réel, avait marché).
        roll_rad = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch_rad = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        cout_assiette = self.coeff_assiette * (roll_rad ** 2 + pitch_rad ** 2)
        reward -= cout_assiette

        # NOUVEAU : pénalité de survitesse, PAS un plafond dur. Diagnostic sac12 :
        # roll passe de -2° à -60/-78° dans les 20 premiers steps, AVANT tout
        # obstacle proche (dist_restante encore ~1.0) -- une manœuvre de lancement
        # pour prendre de la vitesse, systématique, jamais arbitrée par un vrai
        # danger. La deep search confirme : l'état de l'art en évitement réactif
        # (pas sur circuit connu) plafonne à 7-10 m/s, nous tournons à ~21 m/s.
        # Pénalité quadratique au-delà du seuil -- coûte cher si on dépasse
        # beaucoup, rien si on reste dessous, jamais un mur dur qui empêcherait
        # une pointe de vitesse ponctuelle pour une esquive.
        vitesse_actuelle = np.linalg.norm(self.data.qvel[0:3])
        exces_vitesse = max(0.0, vitesse_actuelle - self.vitesse_max)
        cout_vitesse = self.coeff_vitesse_max * (exces_vitesse ** 2)
        reward -= cout_vitesse

        # NOUVEAU : pénalité de mort qui domine TOUJOURS le progrès engrangé,
        # peu importe la distance déjà parcourue. Avant : pénalité fixe
        # (self.penalite_mort=400), dépassée dès que le progrès (25/mètre)
        # dépasse 400 -- soit 16m. Mourir à 30m, 50m, 100m restait donc
        # POSITIF en reward total, aucune vraie dissuasion passé ce seuil.
        # Ici : self.penalite_mort (fixe, coûte cher même à 0m) +
        # self.coeff_penalite_distance * distance_parcourue, avec
        # coeff_penalite_distance > 25 (le taux de la reward de progrès) --
        # garantit mathématiquement que mourir reste négatif quelle que soit
        # la distance atteinte, sans avoir à deviner une valeur fixe énorme.
        distance_parcourue = self.data.body("drone_body").xpos[1]
        penalite_mort_effective = self.penalite_mort + self.coeff_penalite_distance * distance_parcourue

        terminated = False
        is_success = False  # <-- On initialise à False par défaut
        cause_mort = None   # NOUVEAU : pour distinguer mur / sol / arbre / autre

        if en_collision:
            terminated = True
            reward -= penalite_mort_effective
            cause_mort = cause_collision  # "mur", "sol" ou "arbre"
        elif altitude > 8.0:
            terminated = True
            reward -= penalite_mort_effective
            cause_mort = "plafond"
        elif z_up < 0.0:
            terminated = True
            reward -= penalite_mort_effective
            cause_mort = "retournement"
        elif abs(self.data.body("drone_body").xpos[0]) > self.largeur_piste:
            terminated = True
            reward -= penalite_mort_effective
            cause_mort = "sortie_piste_laterale"  # garde-fou position, cf plus haut
        elif self.data.body("drone_body").xpos[1] >= self.longueur_piste:
            terminated = True
            reward += 50.0
            is_success = True  # <-- LE DRONE A GAGNÉ !
            print("🏁 Piste terminée proprement !")
        elif vitesse_rotation > self.seuil_rotation_excessive:
            terminated = True
            reward -= penalite_mort_effective
            cause_mort = "rotation_excessive"

        # On injecte l'information dans le dictionnaire
        info = {
            "distance_y": self.data.body("drone_body").xpos[1],
            "is_success": is_success,  # <-- Transmis à TensorBoard
            "z_up": float(z_up),  # NOUVEAU : purement pour diagnostic (enjoy.py),
                                   # ne touche ni à la reward ni à la physique.
            "cause_mort": cause_mort,  # NOUVEAU : "mur", "sol", "arbre", "plafond",
                                        # "retournement", "sortie_piste_laterale",
                                        # "rotation_excessive", ou None si succès/vivant.
        }

        # NOUVEAU : empilement -- `obs` (brute, une seule trame) a servi à
        # tout le calcul de reward/terminaison ci-dessus, inchangé. Seule
        # l'observation RENVOYÉE est empilée, y compris à la toute dernière
        # trame d'un épisode qui se termine -- donc jamais de décalage de
        # taille possible pour le replay buffer, contrairement à VecFrameStack.
        self._historique.append(obs.copy())
        return self._empiler(), float(reward), terminated, False, info

    def _empiler(self):
        """Concatène l'historique de trames dans l'ordre [plus ancienne ->
        plus récente]. Avec n_stack=1, équivalent strict à la trame brute
        seule (aucun changement de comportement)."""
        if self.n_stack == 1:
            return self._historique[-1]
        return np.concatenate(list(self._historique)).astype(np.float32)

    def _get_obs(self):
        drone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone_body")

        pos = self.data.body("drone_body").xpos
        vel_lin = self.data.qvel[0:3]
        vel_ang = self.data.qvel[3:6]
        quat = self.data.body("drone_body").xquat

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        mat = mat.reshape(3, 3)

        # NOUVEAU : LiDAR stabilisé -- ne suit QUE le cap (yaw), pas le
        # roll/pitch. Avant, `mat` (rotation complète) faisait basculer tout
        # l'éventail hors du plan horizontal dès que le drone s'inclinait
        # pour manoeuvrer -- les rayons pouvaient partir vers le sol/ciel
        # au lieu de viser les arbres à son altitude, précisément pendant
        # les virages où il a le plus besoin de bien voir. Comme un vrai
        # LiDAR gyro-stabilisé : le cône reste à l'horizontale quelle que
        # soit l'inclinaison, seul le cap fait tourner la direction visée.
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        mat_lidar = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ])

        rayons = []
        # NOUVEAU : capture des rayons pour l'affichage visuel dans render().
        # Stockés en mètres réels (pas normalisés), pour tracer les vraies
        # lignes dans le monde 3D plus tard.
        self._dernier_lidar_origine = pos.copy()
        self._dernier_lidar_directions = []
        self._dernier_lidar_distances = []
        geom_id = np.zeros(1, dtype=np.int32)

        angles = self.angles_lidar

        for angle in angles:
            vec_local = np.array([np.sin(angle), np.cos(angle), 0.0])
            vec_global = mat_lidar @ vec_local
            dist = mujoco.mj_ray(self.model, self.data, pos, vec_global, None, 1, drone_id, geom_id)
            if dist < 0:
                dist = self.portee_lidar
            else:
                # NOUVEAU : plafond à portee_lidar. Sans ça, un hit lointain
                # (mur, sol) au-delà de la portée nominale renvoyait sa vraie
                # distance non bornée -- valeur normalisée >1.0, jamais vue
                # pendant l'entraînement (repéré via un lidar_avant=3.289
                # observé en pleine bascule extrême sur sac8_cap_v2, hors
                # distribution d'entraînement).
                dist = min(dist, self.portee_lidar)
            rayons.append(dist / self.portee_lidar)
            self._dernier_lidar_directions.append(vec_global)
            self._dernier_lidar_distances.append(dist)

        # NOUVEAU : distance restante (normalisée par longueur_piste, donc
        # toujours ~1.0 au départ et ~0.0 à l'arrivée, peu importe la
        # longueur réelle tirée) et position latérale (normalisée par
        # largeur_piste, donc -1 = collé au mur gauche, +1 = collé au mur
        # droit, 0 = centré). Le drone sait maintenant OÙ il en est, pas
        # seulement ce qu'il voit localement à l'instant présent.
        distance_restante = (self.longueur_piste - pos[1]) / self.longueur_piste
        position_laterale = pos[0] / self.largeur_piste

        obs = np.concatenate([
            [pos[2]],
            vel_lin,
            vel_ang,
            quat,
            [distance_restante, position_laterale],
            rayons
        ]).astype(np.float32)

        return obs

    def render(self, afficher_lidar=True, tous_les_rayons=False):
        """
        afficher_lidar : dessine les rayons LiDAR par-dessus la scène 3D.
        Par défaut, seulement 3 rayons pour rester lisible : le plus proche
        (rouge -- danger) et les deux plus lointains (vert -- chemins les
        plus dégagés). tous_les_rayons=True dessine les 128 (plus dense,
        utile pour un vrai audit visuel complet mais plus chargé à l'œil).
        """
        if self.viewer is None or self._viewer_model is not self.model:
            # NOUVEAU : reset() recrée entièrement self.model à chaque épisode
            # (nouvelle forêt). Le viewer passif reste lié au modèle avec
            # lequel il a été lancé -- le garder après un reset() affiche une
            # scène obsolète (arbres de l'ancienne forêt) pendant que les
            # positions viennent déjà du nouveau modèle -> décalage visuel,
            # rendu incohérent ("téléportation"). On le referme et relance
            # avec le modèle courant dès qu'on détecte le changement.
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer_model = self.model
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = 1
            self.viewer.cam.distance = 5.0
            self.viewer.cam.elevation = -20

        if afficher_lidar and self._dernier_lidar_origine is not None and self._dernier_lidar_distances:
            self._dessiner_lidar(tous_les_rayons=tous_les_rayons)
        else:
            self.viewer.user_scn.ngeom = 0

        self.viewer.sync()

    def _dessiner_lidar(self, tous_les_rayons=False, portee_affichage=8.0):
        """Trace les rayons LiDAR comme des segments dans le viewer MuJoCo,
        via user_scn (couche de dessin superposée, ne touche pas au modèle
        physique). Rouge = obstacle le plus proche, VERT = les deux bords du
        champ de vision (-120° et +120°) -- jusqu'où le capteur voit sur les
        côtés, JAUNE = rayon "tout droit devant" (index 0°),
        toujours tracé peu importe sa distance -- pour comparer directement
        au petit cube rouge du XML (pos="0 0.1 0", le repère visuel du nez)
        et trancher si un désaccord gauche/droite est un vrai bug de
        convention de signe ou juste un effet de caméra (vue de face au
        lieu de derrière, qui inverse gauche/droite à l'écran sans que rien
        ne soit faux dans les données).

        portee_affichage : plafond de longueur AFFICHÉE en mètres, purement
        visuel -- ne touche jamais aux vraies distances utilisées par la
        politique. Les deux rayons "les plus loin" sont presque toujours
        plafonnés à portee_lidar (30m, rien détecté dans cette direction) ;
        la caméra suit le drone à seulement 5m (cam.distance), donc un trait
        de 30m part très largement hors champ -- invisible en pratique même
        si correctement tracé. 8m reste largement dans le cadre."""
        scn = self.viewer.user_scn
        scn.ngeom = 0

        origine = self._dernier_lidar_origine
        distances = np.array(self._dernier_lidar_distances)
        directions = self._dernier_lidar_directions
        distances_affichees = np.minimum(distances, portee_affichage)

        if tous_les_rayons:
            # Tous les rayons, colorés du rouge (proche) au vert (loin) selon
            # leur distance relative -- vue d'ensemble complète du LiDAR.
            d_min, d_max = distances.min(), distances.max()
            etendue = max(d_max - d_min, 1e-6)
            for direction, dist, dist_aff in zip(directions, distances, distances_affichees):
                t = (dist - d_min) / etendue  # 0=proche/rouge, 1=loin/vert
                couleur = np.array([1.0 - t, t, 0.1, 0.5])
                self._ajouter_segment(scn, origine, origine + direction * dist_aff,
                                      largeur=0.015, rgba=couleur)
        else:
            idx_proche = int(np.argmin(distances))
            # CORRIGÉ : vert = les deux bords du champ de vision (-120° et
            # +120°, premier et dernier index de angles_lidar, qui est trié
            # croissant), pas les deux plus longues distances. Objectif :
            # montrer jusqu'où le capteur voit sur les côtés, pas juste où
            # c'est dégagé -- deux questions différentes.
            idx_bords = (0, len(distances) - 1)

            self._ajouter_segment(
                scn, origine, origine + directions[idx_proche] * distances_affichees[idx_proche],
                largeur=0.04, rgba=np.array([1.0, 0.15, 0.1, 0.9]))
            for i in idx_bords:
                self._ajouter_segment(
                    scn, origine, origine + directions[i] * distances_affichees[i],
                    largeur=0.025, rgba=np.array([0.15, 1.0, 0.2, 0.7]))

        # NOUVEAU : rayon "avant" (0°) toujours tracé en jaune, indépendamment
        # de sa distance -- c'est LUI qu'il faut comparer au cube rouge du nez.
        angles_deg = np.degrees(self.angles_lidar)
        idx_avant = int(np.argmin(np.abs(angles_deg)))
        self._ajouter_segment(
            scn, origine, origine + directions[idx_avant] * distances_affichees[idx_avant],
            largeur=0.03, rgba=np.array([1.0, 0.9, 0.0, 1.0]))

    @staticmethod
    def _ajouter_segment(scn, depart, arrivee, largeur, rgba):
        """Ajoute un segment (type LINE) à la scène de rendu, si la capacité
        du buffer le permet -- au-delà de maxgeom, on ignore silencieusement
        plutôt que de planter (le rendu n'est qu'un outil de debug visuel)."""
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3), pos=np.zeros(3), mat=np.eye(3).flatten(),
            rgba=rgba.astype(np.float32),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, largeur, depart, arrivee)
        scn.ngeom += 1


if __name__ == "__main__":

    # --- Test rapide : randomisation de domaine active ---
    # Chaque reset() tire une config différente dans les plages par défaut.
    env = SpeedrunnerEnv(longueur_piste_range=(30.0, 100.0), largeur_piste=10.0,
                         densite_arbres_range=(0.0, 0.8))
    check_env(env)
    print("Environnement Gym valide (randomisation de domaine) !")

    obs, _ = env.reset()
    print(f"Config tirée : densite={env.densite_arbres:.2f}, longueur={env.longueur_piste:.1f}m")
    print(f"Masse totale vue par le moteur : {env.model.body_mass[1]} kg")
    for _ in range(500):
        env.render()
        env.step(np.array([0, 0, 0, 0]))
        time.sleep(0.02)
"""
Entraînement SAC -- même environnement, même reward, même physique que
train.py (PPO). Seul l'algorithme change.

Pourquoi : après 11 runs PPO, le plafond observé (20-40% de succès avec
obstacles, alors que le vol pur sans obstacle atteint 97-100%) suggère que
le goulot d'étranglement est la prise de décision face aux obstacles, pas
le vol de base. Un papier trouvé plus tôt dans le projet (Kalidas et al.,
comparant DQN/PPO/SAC sur de l'évitement d'obstacles en drone) trouve SAC
nettement meilleur que PPO sur cette famille de tâche précise.
"""

import argparse
import os
import re

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np

from environment import SpeedrunnerEnv


# --- LE RADAR (identique à train.py) ---
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.distances_finales = []
        self.successes = []

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                dist = info.get("distance_y", 0)
                succ = info.get("is_success", False)
                self.distances_finales.append(dist)
                self.successes.append(1.0 if succ else 0.0)
        return True

    def _on_rollout_end(self) -> None:
        if len(self.distances_finales) > 0:
            self.logger.record("rollout/ep_distance_mean", np.mean(self.distances_finales))
            self.logger.record("rollout/success_rate_percent", np.mean(self.successes) * 100)
            self.distances_finales = []
            self.successes = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-density", type=float, default=0.0)
    parser.add_argument("--max-density", type=float, default=0.5)
    parser.add_argument("--min-length", type=float, default=30.0)
    parser.add_argument("--max-length", type=float, default=100.0)
    parser.add_argument("--continue-from", type=str, default=None,
                         help="Chemin vers un .zip SAC, pour reprendre l'entraînement")
    parser.add_argument("--timesteps", type=int, default=200_000,
                         help="SAC est off-policy, généralement plus efficace en échantillons "
                              "que PPO -- 200k pour commencer, pas forcément besoin de 500k.")
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--run-id", type=str, default="sac1")
    parser.add_argument("--gear-roll-pitch", type=float, default=0.5)
    parser.add_argument("--gear-yaw", type=float, default=0.25)
    parser.add_argument("--portee-lidar", type=float, default=30.0)
    parser.add_argument("--seuil-danger", type=float, default=0.3,
                         help="Fraction de portee_lidar en dessous de laquelle la pénalité de "
                              "proximité s'active (0.3 = 9m avec portee_lidar=30m). Repéré comme "
                              "trop haut : le couloir fait 20m de large (10m de chaque côté du "
                              "centre), donc à 0.3 la pénalité est quasi CONSTAMMENT active à "
                              "cause des murs seuls, même sans arbre proche -- noie le signal "
                              "'obstacle proche' dans un bruit de fond permanent. Valeur cible "
                              "discutée : ~0.067 (2m).")
    parser.add_argument("--penalite-mort", type=float, default=400.0)
    parser.add_argument("--seuil-rotation", type=float, default=15.0,
                         help="Seuil de vitesse de rotation (rad/s) qui déclenche la mort "
                              "'rotation_excessive', indépendamment de z_up (le vrai signal de "
                              "perte de contrôle). Testé à 999 en évaluation sur sac15 : résultat "
                              "mitigé (0.2/150m légèrement mieux, 0.3/150m pareil voire moins bien "
                              "-- les rotation_excessive se reconvertissent surtout en arbre/"
                              "retournement à haute densité, pas en succès). Pas de gain net "
                              "confirmé pour l'instant, à re-tester avec une valeur intermédiaire "
                              "si on veut creuser davantage.")
    parser.add_argument("--coeff-agressivite", type=float, default=0.0,
                         help="Coût sur l'amplitude des commandes roll/pitch/yaw, à chaque "
                              "step. Jamais testé sur SAC jusqu'ici (codé en dur à 0.0) -- "
                              "hypothèse : contrairement à PPO (objectif clippé, changement "
                              "de politique déjà borné d'une update à l'autre), SAC n'a rien "
                              "qui empêche l'acteur de sortir une politique moyenne très "
                              "à-coups, ce qui pourrait expliquer la dérive de lacet observée "
                              "en vol pur (sac5 : yaw +138° et crash à 94m sans aucun arbre). "
                              "Le coefficient qui a fait régresser PPO (run9, 0.02) était "
                              "probablement mal calibré pour ce contexte précis -- commencer "
                              "nettement plus bas (ex: 0.002) et vérifier isolément.")
    parser.add_argument("--coeff-saccade", type=float, default=0.0,
                         help="Coût sur le changement brutal d'action entre deux steps. "
                              "Même raison que ci-dessus, même prudence sur le coefficient "
                              "de départ (ex: 0.001, à comparer à 0.01 qui avait fait "
                              "régresser PPO).")
    parser.add_argument("--coeff-orientation", type=float, default=0.0,
                         help="Coût sur l'écart de CAP soutenu (angle réel, pas la commande) "
                              "par rapport à l'avant. Observé sur sac7_controle : le drone "
                              "s'installe parfois à ~90° de travers en continu -- rien "
                              "n'empêchait ça avant puisque la reward de progrès ne se soucie "
                              "que de la position Y, jamais de l'orientation. Coûte cher si "
                              "soutenu (accumulé sur la durée), peu si bref (un virage "
                              "d'esquive normal). Commencer bas (ex: 0.05) et vérifier sur le "
                              "diagnostic yaw plutôt que de deviner.")
    parser.add_argument("--coeff-assiette", type=float, default=0.0,
                         help="Coût sur l'écart d'ASSIETTE soutenu (roll²+pitch² réels, pas la "
                              "commande) -- même principe que --coeff-orientation mais pour le "
                              "roulis/tangage. Observé sur sac9_agressivite : coeff_agressivite "
                              "(qui punit la commande active) n'a pas réduit le vol banqué en "
                              "croisière (roll jusqu'à -73°, pitch jusqu'à +68°, soutenus) -- un "
                              "maintien en régime établi demande peu de commande active, donc ce "
                              "levier ne pouvait pas le toucher. Commencer bas (ex: 0.05, comme "
                              "coeff_orientation) et vérifier sur le diagnostic roll/pitch.")
    parser.add_argument("--coeff-penalite-distance", type=float, default=0.0,
                         help="Pénalité de mort proportionnelle à la distance parcourue, en plus "
                              "de --penalite-mort (fixe). DÉSACTIVÉ par défaut (0.0) : testé à 30 "
                              "sur sac11, a fait s'effondrer le comportement -- l'agent a appris "
                              "que ne PAS avancer était plus sûr qu'avancer (épisodes de 500-940 "
                              "steps à faire du surplace, actor_loss passé positif pour la "
                              "première fois du projet = le critique voyait presque tous les états "
                              "comme négatifs). Le problème de fond reste réel (mourir après 16m "
                              "reste positif en reward total), mais une pénalité qui grandit sans "
                              "borne n'est pas la bonne forme. À revisiter éventuellement avec une "
                              "valeur petite et bornée, jamais 30.")
    parser.add_argument("--angle-fovea", type=float, default=45.0,
                         help="Demi-angle (degrés) du secteur avant où les rayons LiDAR sont "
                              "concentrés. Le nombre total de rayons reste 128 (observation "
                              "inchangée à 141 dims). Motivation chiffrée : à 128 rayons uniformes "
                              "sur 240°, l'espacement est de 1.89°, alors qu'un arbre (rayon 0.3m) "
                              "vu à 20m ne fait que 1.72° -- il peut passer ENTIÈREMENT entre deux "
                              "rayons et rester invisible au-delà de ~18m. Avec 96 rayons sur "
                              "±45° : 0.947°/rayon, arbre détectable jusqu'à 36m, donc fiable sur "
                              "toute la portée (30m). 0 = répartition uniforme historique.")
    parser.add_argument("--rayons-fovea", type=int, default=96,
                         help="Nombre de rayons alloués au secteur avant (le reste va à la "
                              "périphérie, moitié/moitié). 96 sur 128 laisse 16 par côté sur 75°, "
                              "soit 4.7°/rayon -- largement suffisant pour des murs (surfaces "
                              "continues, aucun risque de passer entre deux rayons).")
    parser.add_argument("--vitesse-max", type=float, default=12.0,
                         help="Vitesse cible (m/s) au-delà de laquelle --coeff-vitesse-max "
                              "pénalise. 12 m/s = estimation de départ tirée de la littérature "
                              "(deep search) : l'état de l'art en évitement réactif documenté "
                              "plafonne à 7-10 m/s, contre ~21 m/s mesurés chez nous. Pas une "
                              "valeur validée chez nous, à ajuster selon les résultats.")
    parser.add_argument("--coeff-vitesse-max", type=float, default=0.0,
                         help="Pénalité quadratique au-delà de --vitesse-max. DÉSACTIVÉ par "
                              "défaut (0.0) : rien ne change tant qu'on ne l'active pas "
                              "explicitement. Diagnostic sac12 : le drone banque à 60-78° dès "
                              "les 20 premiers steps, systématiquement, AVANT tout obstacle "
                              "proche (dist_restante encore ~1.0) -- une manœuvre de lancement "
                              "pour prendre de la vitesse, jamais arbitrée par un vrai danger. "
                              "Commencer bas (ex: 0.05) : pénalité quadratique, coûte cher si on "
                              "dépasse beaucoup, rien si on reste dessous, jamais un mur dur.")
    parser.add_argument("--frame-stack", type=int, default=1,
                         help="Nombre de trames d'observation empilées. 1 (défaut) = inchangé. "
                              "Une observation LiDAR instantanée donne la distance des obstacles "
                              "mais pas leur vitesse relative -- impossible de distinguer un "
                              "arbre qu'on approche de face d'un arbre frôlé à la même distance. "
                              "2-4 trames (recommandation de la littérature) permettent au réseau "
                              "d'inférer cette dynamique par différence entre trames. Change la "
                              "dimension d'observation -> réseau neuf obligatoire, jamais de "
                              "reprise avec une valeur différente de celle du checkpoint d'origine.")
    args = parser.parse_args()

    print(f"--- SAC -- densité=[{args.min_density}, {args.max_density}], "
          f"piste=[{args.min_length}, {args.max_length}]m ---")
    print(f"🆔 Run ID : {args.run_id}")
    print(f"⚙️ Agressivité={args.coeff_agressivite}, saccade={args.coeff_saccade}")

    def make_env():
        return SpeedrunnerEnv(
            longueur_piste_range=(args.min_length, args.max_length),
            largeur_piste=10.0,
            densite_arbres_range=(args.min_density, args.max_density),
            gear_roll_pitch=args.gear_roll_pitch,
            gear_yaw=args.gear_yaw,
            portee_lidar=args.portee_lidar,
            seuil_danger=args.seuil_danger,
            penalite_mort=args.penalite_mort,
            seuil_rotation_excessive=args.seuil_rotation,
            coeff_agressivite=args.coeff_agressivite,
            coeff_saccade=args.coeff_saccade,
            coeff_orientation=args.coeff_orientation,
            coeff_assiette=args.coeff_assiette,
            coeff_penalite_distance=args.coeff_penalite_distance,
            angle_fovea=args.angle_fovea,
            rayons_fovea=args.rayons_fovea,
            vitesse_max=args.vitesse_max,
            coeff_vitesse_max=args.coeff_vitesse_max,
            n_stack=args.frame_stack,
        )

    if args.continue_from is not None and args.frame_stack > 1:
        print(f"🚨 ATTENTION : --continue-from ET --frame-stack={args.frame_stack} combinés. "
              f"Le SAC.load ci-dessous va probablement échouer si le checkpoint d'origine "
              f"n'utilisait pas le même --frame-stack (dimension d'observation différente). "
              f"Le frame stacking exige un réseau neuf, jamais une reprise avec une valeur "
              f"différente de celle du checkpoint.")

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    os.makedirs("./modeles/", exist_ok=True)
    chemin_tensorboard = f"./logs_drone/run_{args.run_id}/"

    if args.continue_from is not None:
        candidat_final = args.continue_from.replace(".zip", "_vecnorm.pkl")
        candidat_checkpoint = re.sub(r"_(\d+)_steps\.zip$", r"_vecnormalize_\1_steps.pkl", args.continue_from)
        prev_vecnorm = candidat_final if os.path.exists(candidat_final) else (
            candidat_checkpoint if os.path.exists(candidat_checkpoint) else None
        )
        if prev_vecnorm is not None:
            vec_env = VecNormalize.load(prev_vecnorm, vec_env)
            print(f"Stats de normalisation reprises depuis {prev_vecnorm}")
        else:
            print("🚨 Pas de vecnorm trouvé pour ce checkpoint.")
            reponse = input("   Continuer avec des stats neuves ? (taper 'oui') : ")
            if reponse.strip().lower() != "oui":
                raise SystemExit("Arrêt.")
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    if args.frame_stack > 1:
        print(f"🧩 Frame stacking : {args.frame_stack} trames, empilées DANS "
              f"SpeedrunnerEnv (pas via VecFrameStack -- bug connu avec le "
              f"replay buffer de SAC en fin d'épisode). VecNormalize voit déjà des observations à "
              f"{141 * args.frame_stack} dims.")

    if args.continue_from is not None:
        print(f"🧠 Reprise SAC depuis {args.continue_from}")
        model = SAC.load(args.continue_from, env=vec_env, tensorboard_log=chemin_tensorboard)

        # NOUVEAU : recharge le replay buffer s'il existe. Sans ça, SAC
        # repart avec une mémoire d'expérience vide à chaque reprise -- les
        # poids sont bons mais les mises à jour redeviennent efficaces
        # seulement une fois le buffer reconstitué, d'où le temps plus long
        # pour retrouver le même niveau de performance qu'avant la reprise.
        chemin_buffer = args.continue_from.replace(".zip", "_buffer.pkl")
        if os.path.exists(chemin_buffer):
            model.load_replay_buffer(chemin_buffer)
            print(f"Replay buffer rechargé depuis {chemin_buffer}")
        else:
            print(f"⚠️ Pas de replay buffer trouvé ({chemin_buffer}) -- repart avec une mémoire vide.")
    else:
        print("🧠 Création d'un nouveau cerveau SAC")
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=300_000,   # taille du replay buffer -- SAC est off-policy
            batch_size=256,        # plus grand que PPO, classique pour SAC
            tau=0.005,             # vitesse de mise à jour des réseaux cibles
            gamma=0.99,
            train_freq=1,          # entraîne à chaque step
            gradient_steps=1,
            ent_coef="auto",       # SAC ajuste automatiquement l'exploration -- pas de std à régler à la main
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], qf=[128, 128])),
            tensorboard_log=chemin_tensorboard,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path="./modeles/",
        name_prefix=f"drone_{args.run_id}",
        save_vecnormalize=True,
    )
    callback_list = CallbackList([checkpoint_callback, MetricsCallback()])

    print(f"🚀 Lancement SAC sur {args.n_envs} cœurs, {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps, callback=callback_list, log_interval=4)

    model_path = f"./modeles/drone_{args.run_id}_final"
    model.save(model_path)
    vec_env.save(f"{model_path}_vecnorm.pkl")
    model.save_replay_buffer(f"{model_path}_buffer")  # NOUVEAU : pour permettre une vraie reprise
    print(f"✅ Modèle sauvegardé : {model_path}.zip (+ buffer, + vecnorm)")


if __name__ == "__main__":
    main()
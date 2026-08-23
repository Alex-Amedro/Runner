import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

# Parameters: configure environment ranges, rewards, LiDAR, and frame stacking in SpeedrunnerEnv.


def angles_lidar(angle_fovea_deg=45.0, rayons_fovea=96, n_rayons=128, demi_champ_deg=120.0):
    demi_champ = np.radians(demi_champ_deg)
    if angle_fovea_deg <= 0 or rayons_fovea <= 0 or rayons_fovea >= n_rayons:
        return np.linspace(-demi_champ, demi_champ, n_rayons)

    a = np.radians(angle_fovea_deg)
    n_cote = (n_rayons - rayons_fovea) // 2
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

        self.gear_roll_pitch = gear_roll_pitch
        self.gear_yaw = gear_yaw

        self.longueur_piste_range = longueur_piste_range
        self.largeur_piste = largeur_piste
        self.densite_arbres_range = densite_arbres_range

        self.longueur_piste = longueur_piste_range[1]
        self.densite_arbres = densite_arbres_range[0]

        self.zone_securite = zone_securite

        self.seuil_danger = seuil_danger
        self.coeff_danger = coeff_danger

        self.seuil_danger_sol = seuil_danger_sol
        self.coeff_danger_sol = coeff_danger_sol

        self.penalite_mort = penalite_mort

        self.coeff_agressivite = coeff_agressivite
        self.coeff_saccade = coeff_saccade

        self.coeff_orientation = coeff_orientation
        self.coeff_assiette = coeff_assiette

        self.coeff_penalite_distance = coeff_penalite_distance

        self.angle_fovea = angle_fovea
        self.rayons_fovea = rayons_fovea
        self.angles_lidar = angles_lidar(angle_fovea_deg=angle_fovea,
                                          rayons_fovea=rayons_fovea)

        self.vitesse_max = vitesse_max
        self.coeff_vitesse_max = coeff_vitesse_max

        self.n_stack = max(1, n_stack)
        self._historique = deque(maxlen=self.n_stack)

        self.seuil_rotation_excessive = seuil_rotation_excessive

        self.portee_lidar = portee_lidar

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

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

                    <geom type="capsule" fromto="0 0 0.01  0.17 0.17 0.03" size="0.012"
                          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="capsule" fromto="0 0 0.01 -0.17 0.17 0.03" size="0.012"
                          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="capsule" fromto="0 0 0.01  0.17 -0.17 0.03" size="0.012"
                          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="capsule" fromto="0 0 0.01 -0.17 -0.17 0.03" size="0.012"
                          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>

                    <geom type="cylinder" pos="0.17 0.17 0.035" size="0.028 0.022"
                          rgba="1 0.35 0.12 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="-0.17 0.17 0.035" size="0.028 0.022"
                          rgba="1 0.35 0.12 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="0.17 -0.17 0.035" size="0.028 0.022"
                          rgba="0.2 0.2 0.24 1" contype="0" conaffinity="0" mass="0"/>
                    <geom type="cylinder" pos="-0.17 -0.17 0.035" size="0.028 0.022"
                          rgba="0.2 0.2 0.24 1" contype="0" conaffinity="0" mass="0"/>

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
                continue
            elif distance_spawn < self.zone_securite + marge_transition:
                proba_retrait = 1.0 - (distance_spawn - self.zone_securite) / marge_transition
                if random.random() < proba_retrait:
                    continue

            xml_arbres += f'                <geom type="cylinder" pos="{x:.2f} {y:.2f} 5.0" size="0.3 5.0" rgba="0.7 0.7 0.7 1"/>\n'
            arbres_places += 1

        xml_fin = f"""
                <geom name="mur_gauche" type="box" pos="{-self.largeur_piste:.2f} {self.longueur_piste/2:.2f} 5.0"
                      size="0.1 {self.longueur_piste/2 + 1:.2f} 5.0" rgba="0.5 0.1 0.1 0.6"/>
                <geom name="mur_droit" type="box" pos="{self.largeur_piste:.2f} {self.longueur_piste/2:.2f} 5.0"
                      size="0.1 {self.longueur_piste/2 + 1:.2f} 5.0" rgba="0.5 0.1 0.1 0.6"/>
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
        self.densite_arbres = random.uniform(*self.densite_arbres_range)
        self.longueur_piste = random.uniform(*self.longueur_piste_range)

        xml_string = self._generer_piste_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        mujoco.mj_forward(self.model, self.data)

        self.meilleure_distance = 0.0

        self.action_precedente = np.zeros(4, dtype=np.float32)

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

        cause_collision = None
        if en_collision:
            cause_collision = "arbre"
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

        reward = 0.0
        distance_actuelle = self.data.body("drone_body").xpos[1]
        progres = max(0.0, distance_actuelle - self.meilleure_distance)
        reward += progres * 25.0
        if progres > 0:
            self.meilleure_distance = distance_actuelle

        rayons = obs[13:141]
        min_dist = np.min(rayons)
        if min_dist < self.seuil_danger:
            reward -= self.coeff_danger * (self.seuil_danger - min_dist)

        if altitude < self.seuil_danger_sol:
            reward -= self.coeff_danger_sol * (self.seuil_danger_sol - altitude)

        commandes_rotation = action_reelle[1:4]
        cout_agressivite = self.coeff_agressivite * np.sum(np.square(commandes_rotation))
        cout_saccade = self.coeff_saccade * np.sum(np.square(action_reelle - self.action_precedente))
        reward -= cout_agressivite + cout_saccade
        self.action_precedente = action_reelle.copy()

        quat = self.data.body("drone_body").xquat
        w, x, y, z = quat
        yaw_rad = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cout_orientation = self.coeff_orientation * (yaw_rad ** 2)
        reward -= cout_orientation

        roll_rad = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch_rad = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        cout_assiette = self.coeff_assiette * (roll_rad ** 2 + pitch_rad ** 2)
        reward -= cout_assiette

        vitesse_actuelle = np.linalg.norm(self.data.qvel[0:3])
        exces_vitesse = max(0.0, vitesse_actuelle - self.vitesse_max)
        cout_vitesse = self.coeff_vitesse_max * (exces_vitesse ** 2)
        reward -= cout_vitesse

        distance_parcourue = self.data.body("drone_body").xpos[1]
        penalite_mort_effective = self.penalite_mort + self.coeff_penalite_distance * distance_parcourue

        terminated = False
        is_success = False
        cause_mort = None

        if en_collision:
            terminated = True
            reward -= penalite_mort_effective
            cause_mort = cause_collision
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
            cause_mort = "sortie_piste_laterale"
        elif self.data.body("drone_body").xpos[1] >= self.longueur_piste:
            terminated = True
            reward += 50.0
            is_success = True
            print("🏁 Piste terminée proprement !")
        elif vitesse_rotation > self.seuil_rotation_excessive:
            terminated = True
            reward -= penalite_mort_effective
            cause_mort = "rotation_excessive"

        info = {
            "distance_y": self.data.body("drone_body").xpos[1],
            "is_success": is_success,
            "z_up": float(z_up),
            "cause_mort": cause_mort,
        }

        self._historique.append(obs.copy())
        return self._empiler(), float(reward), terminated, False, info

    def _empiler(self):
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

        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        mat_lidar = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ])

        rayons = []
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
                dist = min(dist, self.portee_lidar)
            rayons.append(dist / self.portee_lidar)
            self._dernier_lidar_directions.append(vec_global)
            self._dernier_lidar_distances.append(dist)

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
        if self.viewer is None or self._viewer_model is not self.model:
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
        scn = self.viewer.user_scn
        scn.ngeom = 0

        origine = self._dernier_lidar_origine
        distances = np.array(self._dernier_lidar_distances)
        directions = self._dernier_lidar_directions
        distances_affichees = np.minimum(distances, portee_affichage)

        if tous_les_rayons:
            d_min, d_max = distances.min(), distances.max()
            etendue = max(d_max - d_min, 1e-6)
            for direction, dist, dist_aff in zip(directions, distances, distances_affichees):
                t = (dist - d_min) / etendue
                couleur = np.array([1.0 - t, t, 0.1, 0.5])
                self._ajouter_segment(scn, origine, origine + direction * dist_aff,
                                      largeur=0.015, rgba=couleur)
        else:
            idx_proche = int(np.argmin(distances))
            idx_bords = (0, len(distances) - 1)

            self._ajouter_segment(
                scn, origine, origine + directions[idx_proche] * distances_affichees[idx_proche],
                largeur=0.04, rgba=np.array([1.0, 0.15, 0.1, 0.9]))
            for i in idx_bords:
                self._ajouter_segment(
                    scn, origine, origine + directions[i] * distances_affichees[i],
                    largeur=0.025, rgba=np.array([0.15, 1.0, 0.2, 0.7]))

        angles_deg = np.degrees(self.angles_lidar)
        idx_avant = int(np.argmin(np.abs(angles_deg)))
        self._ajouter_segment(
            scn, origine, origine + directions[idx_avant] * distances_affichees[idx_avant],
            largeur=0.03, rgba=np.array([1.0, 0.9, 0.0, 1.0]))

    @staticmethod
    def _ajouter_segment(scn, depart, arrivee, largeur, rgba):
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
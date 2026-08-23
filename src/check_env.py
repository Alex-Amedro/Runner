"""
Script de vérification de santé de l'environnement -- fait tourner N steps
(avec un modèle entraîné si fourni, sinon des actions aléatoires) et vérifie
que toutes les valeurs restent dans des plages attendues. Objectif : attraper
des bugs comme le LiDAR non plafonné (repéré manuellement sur sac8_cap_v2)
systématiquement, au lieu de tomber dessus par hasard en lisant un log.

Usage :
    python check_env.py --episodes 5 --steps-max 500
    python check_env.py --modele ./modeles/drone_sac8_cap_v2_final.zip --vecnorm ./modeles/drone_sac8_cap_v2_final_vecnorm.pkl --algo sac --densite 0.0 --longueur 300 --episodes 3
"""

import argparse
import numpy as np

from environment import SpeedrunnerEnv


def verifier_observation(obs, nb_steps, episode_idx, anomalies):
    """Vérifie qu'une observation brute (non normalisée par VecNormalize) est saine."""
    altitude = obs[0]
    vel_lin = obs[1:4]
    vel_ang = obs[4:7]
    quat = obs[7:11]
    distance_restante = obs[11]
    position_laterale = obs[12]
    rayons = obs[13:141]

    def signaler(message):
        anomalies.append(f"[ép.{episode_idx} step {nb_steps}] {message}")

    # NaN / Inf n'importe où -- toujours une vraie anomalie, jamais normal
    if not np.all(np.isfinite(obs)):
        signaler("NaN ou Inf détecté dans l'observation")

    # LiDAR : doit TOUJOURS être dans [0, 1] par construction (0=collé, 1=rien vu)
    if np.any(rayons < -1e-6) or np.any(rayons > 1.0 + 1e-6):
        pires = rayons[(rayons < -1e-6) | (rayons > 1.0 + 1e-6)]
        signaler(f"LiDAR hors de [0,1] : {pires[:5]} (c'est le bug qu'on vient de corriger -- "
                 f"si ça réapparaît, le fix n'a pas pris)")

    # Quaternion : doit être unitaire (norme ~1) pour représenter une vraie rotation
    norme_quat = np.linalg.norm(quat)
    if abs(norme_quat - 1.0) > 0.01:
        signaler(f"Quaternion non-unitaire : norme={norme_quat:.4f} (devrait être 1.0)")

    # Altitude : négative = sous le sol, physiquement impossible sauf bug
    if altitude < -0.5:
        signaler(f"Altitude négative suspecte : {altitude:.3f}")
    if altitude > 8.5:
        signaler(f"Altitude au-delà du plafond de mort (8.0) sans avoir terminé : {altitude:.3f}")

    # Distance restante : ~1.0 au départ, ~0.0 à l'arrivée, un peu de marge tolérée
    if distance_restante < -0.2 or distance_restante > 1.2:
        signaler(f"distance_restante hors plage raisonnable : {distance_restante:.3f}")

    # Position latérale : au-delà de ±1.5 c'est déjà largement hors piste
    if abs(position_laterale) > 1.5:
        signaler(f"position_laterale extrême : {position_laterale:.3f}")

    # Vitesses : pas de valeur absurde (explosion numérique)
    if np.any(np.abs(vel_lin) > 200) or np.any(np.abs(vel_ang) > 200):
        signaler(f"Vitesse extrême : vel_lin={vel_lin}, vel_ang={vel_ang}")


def verifier_reward(reward, nb_steps, episode_idx, anomalies):
    if not np.isfinite(reward):
        anomalies.append(f"[ép.{episode_idx} step {nb_steps}] reward non-finie : {reward}")
    if abs(reward) > 1000:
        anomalies.append(f"[ép.{episode_idx} step {nb_steps}] reward suspecte (>1000 en valeur "
                          f"absolue) : {reward:.1f}")


def _obs_et_reward_brutes(vec_env):
    """Retrouve get_original_obs()/get_original_reward() même si vec_env est
    un VecFrameStack qui enveloppe un VecNormalize (VecFrameStack n'a pas ces
    méthodes lui-même). Le VecNormalize sous-jacent travaille sur l'obs à UNE
    trame (141 dims) même quand un VecFrameStack empile plusieurs trames
    par-dessus -- donc verifier_observation() n'a rien à changer."""
    e = vec_env
    while not hasattr(e, "get_original_obs"):
        if not hasattr(e, "venv"):
            return None, None
        e = e.venv
    return e.get_original_obs()[0], e.get_original_reward()[0]


def jouer_et_verifier(env, model, vec_env, n_episodes, steps_max):
    anomalies = []
    total_steps = 0

    for episode_idx in range(1, n_episodes + 1):
        if model is not None:
            obs_norm = vec_env.reset()
        else:
            obs, _ = env.reset()

        nb_steps = 0
        while nb_steps < steps_max:
            nb_steps += 1
            total_steps += 1

            if model is not None:
                action, _ = model.predict(obs_norm, deterministic=True)
                obs_norm, reward, done, info = vec_env.step(action)
                obs_brut, reward_brut = _obs_et_reward_brutes(vec_env)
                if obs_brut is None:
                    anomalies.append(f"[ép.{episode_idx} step {nb_steps}] "
                                      f"VecNormalize introuvable sous les wrappers -- "
                                      f"vérification impossible pour ce step")
                    termine = bool(done[0])
                    if termine:
                        break
                    continue
                termine = bool(done[0])
            else:
                action = env.action_space.sample()
                obs_brut, reward_brut, termine, _, info = env.step(action)

            verifier_observation(obs_brut, nb_steps, episode_idx, anomalies)
            verifier_reward(reward_brut, nb_steps, episode_idx, anomalies)

            if termine:
                break

    return anomalies, total_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modele", type=str, default=None,
                         help="Chemin vers un modèle entraîné (.zip). Si omis, actions aléatoires.")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"])
    parser.add_argument("--densite", type=float, default=0.0)
    parser.add_argument("--longueur", type=float, default=100.0)
    parser.add_argument("--largeur", type=float, default=10.0)
    parser.add_argument("--gear-roll-pitch", type=float, default=0.5)
    parser.add_argument("--gear-yaw", type=float, default=0.25)
    parser.add_argument("--portee-lidar", type=float, default=30.0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps-max", type=int, default=1000,
                         help="Limite de sécurité par épisode, pour ne pas boucler indéfiniment")
    parser.add_argument("--frame-stack", type=int, default=1,
                         help="DOIT être exactement la même valeur que --frame-stack utilisée "
                              "pour entraîner ce checkpoint (train_sac.py). 1 = pas d'empilement.")
    parser.add_argument("--seuil-rotation", type=float, default=15.0,
                         help="DOIT être exactement la même valeur que --seuil-rotation utilisée "
                              "pour entraîner ce checkpoint. Sinon le modèle est évalué contre une "
                              "limite qu'il n'a jamais appris à respecter (ou l'inverse) -- mort "
                              "quasi instantanée et chiffres sans rapport avec le vrai niveau du "
                              "modèle (vu sur sac16 : 3-4 steps, rotation_excessive en masse, "
                              "check_env.py ne le signale pas comme anomalie -- il vérifie les "
                              "bornes des valeurs, pas si l'épisode s'arrête anormalement tôt).")
    args = parser.parse_args()

    env = SpeedrunnerEnv(
        longueur_piste_range=(args.longueur, args.longueur),
        largeur_piste=args.largeur,
        densite_arbres_range=(args.densite, args.densite),
        gear_roll_pitch=args.gear_roll_pitch,
        gear_yaw=args.gear_yaw,
        portee_lidar=args.portee_lidar,
        n_stack=args.frame_stack,
        seuil_rotation_excessive=args.seuil_rotation,
    )

    model = None
    vec_env = None
    if args.modele is not None:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        def make_env():
            return SpeedrunnerEnv(
                longueur_piste_range=(args.longueur, args.longueur),
                largeur_piste=args.largeur,
                densite_arbres_range=(args.densite, args.densite),
                gear_roll_pitch=args.gear_roll_pitch,
                gear_yaw=args.gear_yaw,
                portee_lidar=args.portee_lidar,
                n_stack=args.frame_stack,
                seuil_rotation_excessive=args.seuil_rotation,
            )

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        classe_algo = SAC if args.algo == "sac" else PPO
        model = classe_algo.load(args.modele, env=vec_env)
        print(f"🧠 Modèle chargé : {args.modele}")
    else:
        print("🎲 Aucun modèle fourni -- actions aléatoires")

    print(f"🔍 Vérification sur {args.episodes} épisodes "
          f"(densité={args.densite}, longueur={args.longueur}m)...\n")

    anomalies, total_steps = jouer_et_verifier(env, model, vec_env, args.episodes, args.steps_max)

    print(f"--- Bilan : {total_steps} steps vérifiés sur {args.episodes} épisodes ---\n")

    if anomalies:
        print(f"⚠️  {len(anomalies)} anomalie(s) détectée(s) :\n")
        # N'affiche pas les 500 lignes si ça spam -- un échantillon suffit à diagnostiquer
        for a in anomalies[:30]:
            print(f"  {a}")
        if len(anomalies) > 30:
            print(f"  ... et {len(anomalies) - 30} de plus (tronqué)")
    else:
        print("✅ Aucune anomalie détectée -- toutes les valeurs sont restées dans des plages saines.")
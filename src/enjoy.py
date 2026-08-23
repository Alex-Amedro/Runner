import argparse
import time
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import SpeedrunnerEnv, angles_lidar


def charger_modele_et_env(chemin_modele, chemin_vecnorm, densite_arbres, longueur_piste,
                           largeur_piste=10.0, gear_roll_pitch=2.0, gear_yaw=1.0,
                           portee_lidar=20.0, algo="ppo", seuil_rotation_excessive=15.0,
                           frame_stack=1):
    def make_env():
        return SpeedrunnerEnv(
            longueur_piste_range=(longueur_piste, longueur_piste),
            largeur_piste=largeur_piste,
            densite_arbres_range=(densite_arbres, densite_arbres),
            gear_roll_pitch=gear_roll_pitch,
            gear_yaw=gear_yaw,
            portee_lidar=portee_lidar,
            seuil_rotation_excessive=seuil_rotation_excessive,
            n_stack=frame_stack,
        )

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(chemin_vecnorm, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    classe_algo = SAC if algo.lower() == "sac" else PPO
    model = classe_algo.load(chemin_modele, env=vec_env)
    return vec_env, model


def _env_de_rendu(vec_env):
    e = vec_env
    while not hasattr(e, "envs"):
        if not hasattr(e, "venv"):
            return None
        e = e.venv
    return e.envs[0]


def _obs_brute(vec_env):
    e = vec_env
    while not hasattr(e, "get_original_obs"):
        if not hasattr(e, "venv"):
            return None
        e = e.venv
    return e.get_original_obs()


def jouer_episode(vec_env, model, render=True, delai=0.02, deterministic=True, diagnostic=False, frame_stack=1):
    obs = vec_env.reset()
    nb_steps = 0

    _angles_deg = np.degrees(angles_lidar())
    idx_gauche_90 = int(np.argmin(np.abs(_angles_deg - (-90.0))))
    idx_droite_90 = int(np.argmin(np.abs(_angles_deg - 90.0)))
    idx_avant = int(np.argmin(np.abs(_angles_deg - 0.0)))

    while True:
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = vec_env.step(action)
        nb_steps += 1

        decalage = (frame_stack - 1) * 141

        distance_restante_brute = None
        if diagnostic:
            obs_brut_check = _obs_brute(vec_env)
            if obs_brut_check is not None:
                distance_restante_brute = obs_brut_check[0][decalage + 11]

        proche_arrivee = diagnostic and distance_restante_brute is not None and distance_restante_brute < 0.1
        if diagnostic and (nb_steps % 25 == 0 or proche_arrivee or nb_steps <= 30):
            obs_brut = _obs_brute(vec_env)
            if obs_brut is None:
                print("  (diagnostic indisponible : VecNormalize introuvable sous les wrappers)")
            else:
                w, x, y, z = obs_brut[0][decalage + 7:decalage + 11]
                yaw_deg = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

                roll_deg = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
                pitch_deg = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))

                lidar_brut = obs_brut[0][decalage + 13:decalage + 141]
                marqueur = " [APPROCHE ARRIVÉE]" if proche_arrivee else ""
                print(f"  step {nb_steps} — yaw={yaw_deg:+.1f}° roll={roll_deg:+.1f}° pitch={pitch_deg:+.1f}° "
                      f"— z_up={info[0].get('z_up', float('nan')):.3f} "
                      f"— dist_restante={distance_restante_brute:.3f} "
                      f"— lidar_avant≈0°={lidar_brut[idx_avant]:.3f} "
                      f"— lidar_gauche≈90°={lidar_brut[idx_gauche_90]:.3f} "
                      f"— lidar_droite≈90°={lidar_brut[idx_droite_90]:.3f}{marqueur}")

        if render:
            env_rendu = _env_de_rendu(vec_env)
            if env_rendu is not None:
                env_rendu.render()
            time.sleep(delai)

        if done[0]:
            return {
                "succes": bool(info[0].get("is_success", False)),
                "distance": float(info[0].get("distance_y", 0.0)),
                "duree_steps": nb_steps,
                "cause_mort": info[0].get("cause_mort", None),
            }


def jouer_plusieurs_episodes(vec_env, model, n_episodes=10, render=True, delai=0.02, diagnostic=False, frame_stack=1):
    resultats = []
    for i in range(n_episodes):
        resultat = jouer_episode(vec_env, model, render=render, delai=delai, diagnostic=diagnostic, frame_stack=frame_stack)
        resultats.append(resultat)
        statut = "🏁 réussi" if resultat["succes"] else f"💥 échoué ({resultat['cause_mort']})"
        print(f"Épisode {i + 1}/{n_episodes} — {statut} — "
              f"distance={resultat['distance']:.1f}m — {resultat['duree_steps']} steps")

    taux_succes = sum(r["succes"] for r in resultats) / len(resultats)
    distance_moyenne = np.mean([r["distance"] for r in resultats])

    print("\n--- Bilan ---")
    print(f"Taux de succès  : {taux_succes * 100:.1f}% ({sum(r['succes'] for r in resultats)}/{n_episodes})")
    print(f"Distance moyenne : {distance_moyenne:.1f}m")

    echecs = [r for r in resultats if not r["succes"]]
    if echecs:
        print("\n--- Causes de mort ---")
        causes = {}
        for r in echecs:
            c = r["cause_mort"] or "inconnue"
            causes[c] = causes.get(c, 0) + 1
        for cause, nb in sorted(causes.items(), key=lambda x: -x[1]):
            print(f"  {cause} : {nb}/{len(echecs)} échecs ({nb / len(echecs) * 100:.0f}%)")

    return resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modele", type=str, default="./modeles/drone_run4_final.zip")
    parser.add_argument("--vecnorm", type=str, default="./modeles/drone_run4_final_vecnorm.pkl")
    parser.add_argument("--densite", type=float, default=0.0,
                         help="Densité d'arbres fixe pour ce test (0.0 = couloir vide)")
    parser.add_argument("--longueur", type=float, default=30.0,
                         help="Longueur de piste fixe pour ce test")
    parser.add_argument("--largeur", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=10,
                         help="Nombre d'épisodes à jouer (pas de boucle infinie)")
    parser.add_argument("--no-render", action="store_true",
                         help="Désactive le rendu visuel (pour aller plus vite)")
    parser.add_argument("--diagnostic", action="store_true",
                         help="Affiche l'inclinaison (z_up) toutes les 25 steps")
    parser.add_argument("--gear-roll-pitch", type=float, default=2.0,
                         help="Doit matcher le gear utilisé à l'entraînement : "
                              "2.0 pour run4 (original), 0.5 pour gear_test")
    parser.add_argument("--gear-yaw", type=float, default=1.0,
                         help="Doit matcher le gear utilisé à l'entraînement : "
                              "1.0 pour run4 (original), 0.25 pour gear_test")
    parser.add_argument("--portee-lidar", type=float, default=20.0,
                         help="Doit matcher la portée utilisée à l'entraînement : "
                              "20 (défaut, pour run4/run5) ou 30 pour run6 et suivants")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"],
                         help="Doit matcher le script d'entraînement utilisé : "
                              "'ppo' pour train.py, 'sac' pour train_sac.py")
    parser.add_argument("--seuil-rotation", type=float, default=15.0,
                         help="Seuil de vitesse de rotation (rad/s) qui déclenche la mort "
                              "'rotation_excessive'. Défaut 15.0 = comportement d'entraînement "
                              "inchangé. Augmente-le (ex: 50 ou 999) pour voir si le modèle "
                              "termine sa manoeuvre proprement au-delà de ce seuil (esquive "
                              "punie à tort) ou continue à perdre le contrôle (vraie instabilité).")
    parser.add_argument("--frame-stack", type=int, default=1,
                         help="DOIT être exactement la même valeur que --frame-stack utilisée "
                              "pour entraîner ce checkpoint (train_sac.py). 1 = pas d'empilement.")
    args = parser.parse_args()

    print(f"🎬 densité={args.densite}, longueur={args.longueur}m, {args.episodes} épisodes "
          f"(algo={args.algo}, gear roll/pitch={args.gear_roll_pitch}, yaw={args.gear_yaw}, "
          f"portée LiDAR={args.portee_lidar}m)")

    vec_env, model = charger_modele_et_env(
        args.modele, args.vecnorm, args.densite, args.longueur, args.largeur,
        gear_roll_pitch=args.gear_roll_pitch, gear_yaw=args.gear_yaw,
        portee_lidar=args.portee_lidar, algo=args.algo,
        seuil_rotation_excessive=args.seuil_rotation,
        frame_stack=args.frame_stack,
    )

    try:
        jouer_plusieurs_episodes(vec_env, model, n_episodes=args.episodes,
                                  render=not args.no_render, diagnostic=args.diagnostic,
                                  frame_stack=args.frame_stack)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt manuel demandé.")
    finally:
        vec_env.close()
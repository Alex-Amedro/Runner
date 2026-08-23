"""
mesure_distance.py — Distance moyenne parcourue avant échec, sur plusieurs
densités, en parallèle sur N environnements (comme train_sac.py le fait pour
l'entraînement, mais ici pour l'évaluation).

Pourquoi ce script plutôt que enjoy.py : enjoy.py tourne sur un seul
environnement séquentiellement. Sur des pistes de 10 km avec 100 épisodes par
densité, ça peut prendre des heures. Ici, --n-envs environnements tournent en
parallèle (SubprocVecEnv), donc le temps est divisé par ~n_envs.

Usage :
    python mesure_distance.py \
        --modele ./modeles/drone_sac16_seuil_rotation_final.zip \
        --vecnorm ./modeles/drone_sac16_seuil_rotation_final_vecnorm.pkl \
        --algo sac --frame-stack 3 --seuil-rotation 999 \
        --densites 0.0,0.2,0.3,0.5 --longueur 10000 --episodes 100 --n-envs 10
"""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from test2 import SpeedrunnerEnv


def construire_env(densite, longueur, largeur, gear_roll_pitch, gear_yaw,
                    portee_lidar, seuil_rotation, n_stack):
    """Retourne une fonction usine (nécessaire pour SubprocVecEnv, qui doit
    pouvoir recréer l'environnement dans chaque sous-processus)."""
    def _init():
        return SpeedrunnerEnv(
            longueur_piste_range=(longueur, longueur),
            largeur_piste=largeur,
            densite_arbres_range=(densite, densite),
            gear_roll_pitch=gear_roll_pitch,
            gear_yaw=gear_yaw,
            portee_lidar=portee_lidar,
            seuil_rotation_excessive=seuil_rotation,
            n_stack=n_stack,
        )
    return _init


def mesurer_densite(modele, vecnorm, algo, densite, longueur, largeur,
                     gear_roll_pitch, gear_yaw, portee_lidar, seuil_rotation,
                     n_stack, n_episodes, n_envs, verbeux=True):
    """Lance n_envs environnements en parallèle sur cette densité, collecte
    la distance à la fin de chaque épisode (info["distance_y"], déjà calculée
    par SpeedrunnerEnv.step() -- succès ou échec, peu importe, on veut juste
    savoir jusqu'où il est allé), jusqu'à en avoir au moins n_episodes."""
    n_envs = min(n_envs, n_episodes)  # inutile de lancer plus d'envs que d'épisodes voulus
    fns = [construire_env(densite, longueur, largeur, gear_roll_pitch, gear_yaw,
                           portee_lidar, seuil_rotation, n_stack)
           for _ in range(n_envs)]
    vec = SubprocVecEnv(fns)
    vec = VecNormalize.load(vecnorm, vec)
    vec.training = False
    vec.norm_reward = False

    classe = SAC if algo.lower() == "sac" else PPO
    model = classe.load(modele, env=vec)

    obs = vec.reset()
    distances = []
    succes = []
    t0 = time.time()

    while len(distances) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec.step(action)
        for i, done in enumerate(dones):
            if done and "distance_y" in infos[i]:
                distances.append(infos[i]["distance_y"])
                succes.append(bool(infos[i].get("is_success", False)))
                if verbeux and len(distances) % max(1, n_episodes // 10) == 0:
                    print(f"    ... {len(distances)}/{n_episodes} épisodes "
                          f"({time.time() - t0:.0f}s écoulées)")
                if len(distances) >= n_episodes:
                    break

    vec.close()
    distances = np.array(distances[:n_episodes])
    succes = np.array(succes[:n_episodes])
    dt = time.time() - t0
    return distances, succes, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modele", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    p.add_argument("--densites", default="0.0,0.2,0.3,0.5",
                    help="Liste séparée par des virgules, ex: 0.0,0.2,0.3,0.5")
    p.add_argument("--longueur", type=float, default=10000.0)
    p.add_argument("--largeur", type=float, default=10.0)
    p.add_argument("--gear-roll-pitch", type=float, default=0.5)
    p.add_argument("--gear-yaw", type=float, default=0.25)
    p.add_argument("--portee-lidar", type=float, default=30.0)
    p.add_argument("--seuil-rotation", type=float, default=15.0)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--episodes", type=int, default=100,
                    help="Par densité non nulle. Densité 0.0 force 1 seul "
                         "épisode (déterministe, inutile de répéter).")
    p.add_argument("--n-envs", type=int, default=10)
    args = p.parse_args()

    densites = [float(d) for d in args.densites.split(",")]
    resultats = []

    for d in densites:
        n_ep = 1 if d == 0.0 else args.episodes
        print(f"\n=== densité {d} -- {n_ep} épisode(s), longueur max {args.longueur:.0f}m ===")
        distances, succes, dt = mesurer_densite(
            args.modele, args.vecnorm, args.algo, d, args.longueur, args.largeur,
            args.gear_roll_pitch, args.gear_yaw, args.portee_lidar,
            args.seuil_rotation, args.frame_stack, n_ep, args.n_envs,
        )
        moy, med = float(np.mean(distances)), float(np.median(distances))
        std, mn, mx = float(np.std(distances)), float(np.min(distances)), float(np.max(distances))
        taux_succes = 100.0 * float(np.mean(succes))
        print(f"  distance moyenne : {moy:.1f}m  (médiane {med:.1f}m, "
              f"écart-type {std:.1f}m, min {mn:.1f}m, max {mx:.1f}m)")
        print(f"  taux de succès (atteint {args.longueur:.0f}m sans échec) : {taux_succes:.1f}%")
        print(f"  temps écoulé : {dt:.0f}s")
        resultats.append({
            "densite": d, "moyenne": moy, "mediane": med, "ecart_type": std,
            "min": mn, "max": mx, "taux_succes": taux_succes, "n": n_ep,
        })

    print("\n\n### Tableau récapitulatif (prêt à coller dans le README)\n")
    print("| Densité d'obstacles | Distance moyenne avant échec |")
    print("|---|---|")
    for r in resultats:
        if r["taux_succes"] >= 99.9:
            valeur = f"≥{args.longueur/1000:.0f} km, aucun échec observé"
        else:
            valeur = f"{r['moyenne']:.0f} m"
        print(f"| {r['densite']} | {valeur} |")

    print("\n### Détail complet (pour toi, pas pour le README)\n")
    print("| Densité | Moyenne | Médiane | Écart-type | Min | Max | Succès | n |")
    print("|---|---|---|---|---|---|---|---|")
    for r in resultats:
        print(f"| {r['densite']} | {r['moyenne']:.0f}m | {r['mediane']:.0f}m | "
              f"{r['ecart_type']:.0f}m | {r['min']:.0f}m | {r['max']:.0f}m | "
              f"{r['taux_succes']:.0f}% | {r['n']} |")


if __name__ == "__main__":
    main()
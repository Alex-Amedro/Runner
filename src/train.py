import argparse
import os
import re
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# Le bon environnement physique V2
from environment import SpeedrunnerEnv


def linear_schedule(valeur_initiale, valeur_finale):
    """
    Schedule linéaire pour le learning_rate : décroît de valeur_initiale
    à valeur_finale au fil de l'entraînement. SB3 appelle cette fonction
    avec progress_remaining qui va de 1.0 (début) à 0.0 (fin).
    Remplace target_kl : au lieu de couper les mises à jour trop grandes
    après coup, on réduit la taille des pas progressivement, ce qui
    correspond à la vraie cause du problème (approx_kl qui grimpe quand
    la politique se resserre, à learning_rate fixe).
    """
    def schedule(progress_remaining):
        return valeur_finale + progress_remaining * (valeur_initiale - valeur_finale)
    return schedule


# --- LE RADAR (Distance + Taux de Victoire) ---
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
                if "terminal_info" in info:
                    dist = info["terminal_info"].get("distance_y", 0)
                    succ = info["terminal_info"].get("is_success", False)
                else:
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

# --- RANDOMISATION DE DOMAINE (remplace le CURRICULUM séquentiel figé) ---
# Cf découverte : le curriculum étape par étape entraînait un oubli
# catastrophique (0% de succès en couloir vide après entraînement intensif
# avec obstacles). Chaque episode tire maintenant sa propre config dans ces
# plages, donc l'agent voit un mélange de difficultés en permanence.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-density", type=float, default=0.0,
                         help="Densité d'arbres minimale tirée aléatoirement par épisode")
    parser.add_argument("--max-density", type=float, default=0.8,
                         help="Densité d'arbres maximale tirée aléatoirement par épisode")
    parser.add_argument("--min-length", type=float, default=30.0,
                         help="Longueur de piste minimale tirée aléatoirement par épisode")
    parser.add_argument("--max-length", type=float, default=100.0,
                         help="Longueur de piste maximale tirée aléatoirement par épisode")
    parser.add_argument("--continue-from", type=str, default=None,
                         help="Chemin vers un .zip, pour reprendre l'entraînement")
    parser.add_argument("--timesteps", type=int, default=100000,
                         help="Nombre de steps À AJOUTER à la session (ex: 30000)")
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--run-id", type=str, default="1",
                         help="Numéro ou nom du run (ex: 1, 2, ou 'test_rapide'). Isole les logs et sauvegardes.")
    parser.add_argument("--gear-roll-pitch", type=float, default=0.5,
                         help="Validé par test_gear.py : 0.5 au lieu de l'original 2.0 "
                              "(0.97-1.0 de succès contre 0.20-0.33 avant)")
    parser.add_argument("--gear-yaw", type=float, default=0.25,
                         help="Validé par test_gear.py : 0.25 au lieu de l'original 1.0")
    parser.add_argument("--portee-lidar", type=float, default=30.0,
                         help="Portée du LiDAR en mètres. Augmentée de 20 à 30 par défaut : "
                              "avec le plafond de vitesse retiré, le drone va plus vite, donc "
                              "20m ne laissait plus assez de temps de réaction pour esquiver "
                              "(0% de succès observé à densité 0.5 avec 20m)")
    parser.add_argument("--penalite-mort", type=float, default=400.0)
    parser.add_argument("--coeff-agressivite", type=float, default=0.0,
                         help="Coût sur l'amplitude des commandes roll/pitch/yaw, à chaque "
                              "step (inspiré du r_cmd de Swift). DÉSACTIVÉ par défaut (0.0) : "
                              "testé à 0.02 sur run9, a causé une régression nette vs run8 "
                              "(40%->20% succès, 300-472m->111m en couloir vide). Pas réactivé "
                              "tant qu'on n'a pas isolé pourquoi ça a empiré les choses.")
    parser.add_argument("--coeff-saccade", type=float, default=0.0,
                         help="Coût sur le changement brutal d'action entre deux steps. "
                              "DÉSACTIVÉ par défaut (0.0), même raison que ci-dessus.")
    args = parser.parse_args()

    print(f"--- Randomisation de domaine : densité=[{args.min_density}, {args.max_density}], "
          f"piste=[{args.min_length}, {args.max_length}]m ---")
    print(f"🆔 Run ID : {args.run_id}")
    print(f"⏱️ Entraînement commandé pour {args.timesteps} steps supplémentaires.")
    print(f"⚙️ Gear roll/pitch={args.gear_roll_pitch}, yaw={args.gear_yaw}, portée LiDAR={args.portee_lidar}m, "
          f"pénalité mort={args.penalite_mort}, agressivité={args.coeff_agressivite}, saccade={args.coeff_saccade}")

    def make_env():
        return SpeedrunnerEnv(
            longueur_piste_range=(args.min_length, args.max_length),
            largeur_piste=10.0,
            densite_arbres_range=(args.min_density, args.max_density),
            gear_roll_pitch=args.gear_roll_pitch,
            gear_yaw=args.gear_yaw,
            portee_lidar=args.portee_lidar,
            penalite_mort=args.penalite_mort,
            coeff_agressivite=args.coeff_agressivite,
            coeff_saccade=args.coeff_saccade,
        )

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    os.makedirs("./modeles/", exist_ok=True)

    # 🆕 LE DOSSIER TENSORBOARD EST MAINTENANT UNIQUE POUR CHAQUE RUN-ID
    chemin_tensorboard = f"./logs_drone/run_{args.run_id}/"

    # Gestion de la normalisation
    if args.continue_from is not None:
        # Deux conventions de nommage possibles selon d'où vient le checkpoint :
        # 1) Sauvegarde finale manuelle : "..._final.zip" -> "..._final_vecnorm.pkl"
        # 2) CheckpointCallback(save_vecnormalize=True) : "..._701760_steps.zip"
        #    -> "..._vecnormalize_701760_steps.pkl" (insertion, pas suffixe)
        candidat_final = args.continue_from.replace(".zip", "_vecnorm.pkl")
        candidat_checkpoint = re.sub(r"_(\d+)_steps\.zip$", r"_vecnormalize_\1_steps.pkl", args.continue_from)

        if os.path.exists(candidat_final):
            prev_vecnorm = candidat_final
        elif os.path.exists(candidat_checkpoint):
            prev_vecnorm = candidat_checkpoint
        else:
            prev_vecnorm = None

        if prev_vecnorm is not None:
            vec_env = VecNormalize.load(prev_vecnorm, vec_env)
            print(f"Stats de normalisation reprises depuis {prev_vecnorm}")
        else:
            # AVANT : ce cas passait presque inaperçu et repartait sur des stats
            # neuves, ce qui a corrompu un run entier (KL qui explose direct).
            # Maintenant ça s'arrête, sauf si tu confirmes explicitement.
            print("🚨 ATTENTION : aucun fichier vecnormalize trouvé pour ce checkpoint.")
            print(f"   Cherché : {candidat_final}")
            print(f"   Cherché : {candidat_checkpoint}")
            print("   Reprendre l'entraînement SANS les bonnes stats de normalisation")
            print("   va corrompre le modèle (comme le run précédent).")
            reponse = input("   Continuer quand même avec des stats neuves ? (taper 'oui' pour confirmer) : ")
            if reponse.strip().lower() != "oui":
                raise SystemExit("Arrêt. Vérifie le chemin du checkpoint ou celui du fichier vecnormalize.")
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Chargement ou création du Cerveau
    if args.continue_from is not None:
        print(f"🧠 Reprise de l'entraînement depuis {args.continue_from}")
        # learning_rate forcé via custom_objects : sinon la reprise garde
        # l'ancien schedule sauvegardé dans le modèle.
        model = PPO.load(
            args.continue_from,
            env=vec_env,
            tensorboard_log=chemin_tensorboard,
            custom_objects={"learning_rate": linear_schedule(3e-4, 1e-5)},
        )
    else:
        print("🧠 Création d'un nouveau cerveau PPO (Table Rase)")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            # NOUVEAU : schedule décroissant (3e-4 -> 1e-5) à la place de
            # target_kl. Remplace le pansement (couper les mises à jour
            # trop grandes après coup) par le vrai traitement (des pas
            # plus petits à mesure que la politique se resserre).
            learning_rate=linear_schedule(3e-4, 1e-5),
            n_steps=512,
            batch_size=64,
            # NOUVEAU : réseau 128x128 (policy ET value), comme Swift.
            # On tournait sur le défaut SB3 (plus petit), jamais comparé.
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            tensorboard_log=chemin_tensorboard, # Isolation ici
        )

    frequence_sauvegarde = max(10000, args.timesteps // 5)
    checkpoint_callback = CheckpointCallback(
        save_freq=frequence_sauvegarde // args.n_envs,
        save_path="./modeles/",
        name_prefix=f"drone_run{args.run_id}",
        save_vecnormalize=True,  # NOUVEAU : sauvegarde les stats VecNormalize
                                  # avec CHAQUE checkpoint, pas juste à la fin.
                                  # Sans ça, tes checkpoints intermédiaires sont
                                  # inutilisables (mauvaises stats de normalisation).
    )

    callback_list = CallbackList([checkpoint_callback, MetricsCallback()])

    print(f"🚀 Lancement sur {args.n_envs} cœurs...")

    # NOUVEAU : reset_num_timesteps=True TOUJOURS, même en reprise.
    # Avant : en reprise, SB3 calcule le budget du schedule de LR comme
    # (steps déjà accumulés + args.timesteps), donc le schedule ne repart
    # jamais de 3e-4 -- il continue de décroître sur TOUTE l'historique
    # cumulée, et arrive déjà proche du plancher (1e-5) après plusieurs
    # reprises, même quand on introduit un truc tout neuf (les murs).
    # Ici, chaque lancement de train.py obtient son propre cycle complet
    # 3e-4 -> 1e-5 sur le --timesteps demandé, peu importe l'historique.
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback_list,
        reset_num_timesteps=True,
        tb_log_name="domain_rand"
    )

    # 🆕 SAUVEGARDE FINALE AVEC RUN-ID
    model_path = f"./modeles/drone_run{args.run_id}_final"
    model.save(model_path)
    vec_env.save(f"{model_path}_vecnorm.pkl")

    print(f"✅ Modèle sauvegardé : {model_path}.zip")

if __name__ == "__main__":
    main()
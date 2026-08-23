import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv 

from test1 import SpeedrunnerEnv 

# --- CRÉATION DE LA COURBE TENSORBOARD ---
class DistanceCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.distances_finales = []

    def _on_step(self) -> bool:
        # À chaque step, on regarde si un des 10 drones s'est crashé
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        
        for i, done in enumerate(dones):
            if done:
                # Récupère la distance finale (compatible avec toutes les versions de Gym/SB3)
                info = infos[i]
                if "terminal_info" in info:
                    dist = info["terminal_info"].get("distance_y", 0)
                else:
                    dist = info.get("distance_y", 0)
                self.distances_finales.append(dist)
        return True

    def _on_rollout_end(self) -> None:
        # Quand TensorBoard met à jour l'écran, on calcule la moyenne
        if len(self.distances_finales) > 0:
            self.logger.record("rollout/ep_distance_mean", np.mean(self.distances_finales))
            self.distances_finales = []
# -----------------------------------------

if __name__ == "__main__":
    def make_env():
        return SpeedrunnerEnv(longueur_piste=150.0, largeur_piste=10.0, densite_arbres=0.4)
    
    vec_env = make_vec_env(make_env, n_envs=10, vec_env_cls=SubprocVecEnv)
    
    # --- HYPERPARAMÈTRES (On le calme) ---
    # Paramètres d'entraînement "Fine-Tuning de Précision"
    custom_objects = {
        "learning_rate": 5e-5,  # Plus bas, pour de l'ajustement millimétrique (avant: 1e-4)
        "ent_coef": 0.0,        # Plus d'exploration forcée, il faut de la performance
    }
    

    # CHARGE TON DERNIER "BON" CHECKPOINT ICI
    chemin_sauvegarde = "./modeles_finetuning/drone_expert_4300000_steps.zip" 
    print(f"🧠 Chargement du cerveau : {chemin_sauvegarde}")
    
    model = PPO.load(
        chemin_sauvegarde, 
        env=vec_env, 
        custom_objects=custom_objects,
        tensorboard_log="./logs_drone/"
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./modeles_finetuning/', name_prefix='drone_expert')
    
    # On ajoute notre Callback de distance dans une liste
    callback_list = [checkpoint_callback, DistanceCallback()]

    print("🚀 Début du Fine-Tuning avec radar de distance...")
    model.learn(total_timesteps=5000000, callback=callback_list, reset_num_timesteps=False)
    model.save("modele_drone_v2_expert")
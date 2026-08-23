import os
from tensorboard import program

# On cible le dossier où PPO écrit ses données
logdir = "./logs_drone"

print("\n--- 🕵️ DIAGNOSTIC TENSORBOARD ---")

# 1. Vérification du chemin
chemin_absolu = os.path.abspath(logdir)
if not os.path.exists(chemin_absolu):
    print(f"❌ ERREUR : Le dossier est introuvable ici :\n{chemin_absolu}")
    print("👉 Solution : Tu n'es pas dans le bon dossier dans ton terminal.")
    exit()

print(f"✅ 1. Dossier de logs trouvé : {chemin_absolu}")

# 2. Lancement forcé
try:
    print("⏳ 2. Démarrage du serveur...")
    tb = program.TensorBoard()
    
    # On force '127.0.0.1' pour éviter le conflit classique Windows/localhost
    # On laisse TensorBoard choisir le port 6006, ou un autre s'il est bloqué
    tb.configure(argv=[None, '--logdir', chemin_absolu, '--host', '127.0.0.1'])
    url = tb.launch()
    
    print("\n🎉 SUCCÈS ! Clique directement sur ce lien (CTRL + Clic) :")
    print(f"➡️  {url}")
    print("\n(Laisse ce terminal ouvert pour garder la page active)")
    
    # Garde le programme en vie
    input()
    
except Exception as e:
    print(f"\n❌ ERREUR RÉSEAU : {e}")
    print("👉 Solution : Un autre programme bloque probablement le port.")
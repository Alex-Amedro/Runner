"""
enjoy_sac.py — Boîte à outils de test pour les checkpoints SAC du drone speedrunner.

Aucun modèle n'est codé en dur : les chemins sont résolus dynamiquement.
On peut désigner un checkpoint par son chemin complet, par un fragment de nom
("sac12"), ou ne rien préciser du tout (le plus avancé est pris).

    python enjoy_sac.py lister
    python enjoy_sac.py balayage --run sac12 --episodes 30
    python enjoy_sac.py regarder --run sac12 --densite 0.35
    python enjoy_sac.py comparer --run sac12 --contre sac9 --episodes 30

Importable aussi :
    from enjoy_sac import charger, jouer_episode, evaluer, balayage_densites, comparer_modeles
"""

import argparse
import inspect
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import SpeedrunnerEnv

DENSITES_STANDARD = (0.0, 0.2, 0.35, 0.5)
# Dossiers fouillés par défaut, en plus du dossier courant. La recherche est
# récursive, donc un checkpoint rangé dans un sous-dossier est trouvé aussi.
RACINES = (".", "modeles", "models", "checkpoints", "logs", "runs")


# --------------------------------------------------------------------------
# 0. Résolution des chemins de checkpoints
# --------------------------------------------------------------------------

def _racines(racine=None):
    if racine:
        return [Path(racine)]
    return [Path(r) for r in RACINES if Path(r).exists()]


def lister_checkpoints(racine=None):
    """Tous les .zip trouvés récursivement, du plus récent au plus ancien, sans doublon."""
    vus, trouves = set(), []
    for base in _racines(racine):
        for chemin in base.rglob("*.zip"):
            reel = chemin.resolve()
            if reel not in vus:
                vus.add(reel)
                trouves.append(chemin)
    return sorted(trouves, key=lambda p: p.stat().st_mtime, reverse=True)


def _steps_du_nom(nom):
    """Extrait le nombre de steps d'un nom de checkpoint, pour trier proprement."""
    m = re.search(r"(\d+)_steps", nom)
    return int(m.group(1)) if m else -1


def resoudre_modele(motif=None, racine=None):
    """Trouve un checkpoint à partir d'un chemin, d'un fragment de nom, ou de rien.

    Le tri se fait par nombre de steps, pas alphabétiquement : sinon
    "..._500000_steps" passerait avant "..._1000000_steps".
    """
    if motif and Path(motif).is_file():
        return Path(motif)

    candidats = lister_checkpoints(racine)
    if not candidats:
        raise FileNotFoundError(
            f"Aucun .zip trouvé sous {[str(r) for r in _racines(racine)]}. "
            "Précise --racine si tes checkpoints sont ailleurs."
        )

    if motif:
        # Correspondance souple mais pas laxiste : "sac1" ne doit PAS matcher "sac12",
        # d'où le \D (ou fin de chaîne) exigé juste après le motif.
        motif_strict = re.compile(rf"{re.escape(motif)}(\D|$)", re.IGNORECASE)
        filtres = [c for c in candidats if motif_strict.search(c.name)]
        if not filtres:
            filtres = [c for c in candidats if motif.lower() in c.name.lower()]
        if not filtres:
            dispo = "\n  ".join(c.name for c in candidats[:15])
            raise FileNotFoundError(f"Rien ne correspond à '{motif}'. Trouvés :\n  {dispo}")
        candidats = filtres

    # Le plus avancé en steps ; à égalité (ou sans steps dans le nom), le plus récent.
    return max(candidats, key=lambda p: (_steps_du_nom(p.name), p.stat().st_mtime))


def resoudre_vecnorm(chemin_modele, racine=None):
    """Trouve le .pkl de normalisation qui va avec un checkpoint donné.

    Les deux fichiers ne suivent pas la même convention de nommage
    ("..._500000_steps.zip" contre "..._vecnormalize_500000_steps.pkl"), donc
    on apparie sur le préfixe de run ET le nombre de steps, pas sur le nom brut.
    Charger le vecnorm d'un autre run fausserait toutes les observations en
    silence, sans lever d'erreur — d'où la vérification stricte.
    """
    chemin_modele = Path(chemin_modele)
    tige = chemin_modele.stem
    steps = _steps_du_nom(tige)
    prefixe = re.sub(r"_?\d+_steps$", "", tige)
    prefixe = re.sub(r"_final$", "", prefixe)

    pkls, vus = [], set()
    for base in [chemin_modele.parent] + _racines(racine):
        if not base.exists():
            continue
        for p in base.rglob("*.pkl"):
            reel = p.resolve()
            if reel not in vus:
                vus.add(reel)
                pkls.append(p)

    def score(p):
        nom = p.stem
        s = 0
        if prefixe.lower() in nom.lower():
            s += 10
        if steps >= 0 and _steps_du_nom(nom) == steps:
            s += 5
        if "vecnorm" in nom.lower():
            s += 1
        return s

    if not pkls:
        return None
    meilleur = max(pkls, key=score)
    if score(meilleur) < 10:
        print(f"⚠️  Aucun vecnormalize ne correspond clairement à {chemin_modele.name} "
              f"— chargement SANS normalisation (les chiffres seront faux si le modèle "
              f"a été entraîné avec VecNormalize).")
        return None
    if steps >= 0 and _steps_du_nom(meilleur.stem) != steps:
        print(f"⚠️  {meilleur.name} n'a pas le même nombre de steps que "
              f"{chemin_modele.name} — vérifie que c'est voulu.")
    return meilleur


# --------------------------------------------------------------------------
# 1. Construction de l'environnement
# --------------------------------------------------------------------------

def _kwargs_compatibles(cls, **souhaits):
    """Ne garde que les arguments réellement acceptés par le constructeur.
    Évite de casser si la signature de SpeedrunnerEnv a bougé entre deux runs."""
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in souhaits.items() if k in params}


def construire_env(densite=0.2, longueur=300.0, largeur=10.0, render=False):
    """Environnement figé sur UNE config précise (densité et longueur non aléatoires),
    pour que le chiffre mesuré veuille dire quelque chose."""
    souhaits = dict(
        densite_arbres_range=(densite, densite),
        longueur_piste_range=(longueur, longueur),
        densite_arbres=densite,
        longueur_piste=longueur,
        largeur_piste=largeur,
        render_mode="human" if render else None,
    )
    return SpeedrunnerEnv(**_kwargs_compatibles(SpeedrunnerEnv, **souhaits))


def charger(modele=None, vecnorm=None, densite=0.2, longueur=300.0,
            largeur=10.0, render=False, racine=None, verbeux=True):
    """Charge modèle + normalisation figée. `modele` et `vecnorm` acceptent un chemin,
    un fragment de nom, ou None (auto). Retourne (model, vec_env)."""
    chemin_modele = resoudre_modele(modele, racine)
    chemin_vecnorm = Path(vecnorm) if vecnorm and Path(vecnorm).is_file() \
        else resoudre_vecnorm(chemin_modele, racine)

    if verbeux:
        print(f"📦 modèle  : {chemin_modele}")
        print(f"📊 vecnorm : {chemin_vecnorm if chemin_vecnorm else '(aucun)'}")

    vec = DummyVecEnv([lambda: construire_env(densite, longueur, largeur, render)])
    if chemin_vecnorm:
        vec = VecNormalize.load(str(chemin_vecnorm), vec)
        vec.training = False      # les stats ne bougent plus
        vec.norm_reward = False   # reward brute, lisible
    model = SAC.load(str(chemin_modele), env=vec, device="auto")
    return model, vec


def _env_brut(vec):
    """Accès à l'env MuJoCo sous les couches VecNormalize / DummyVecEnv."""
    base = vec.venv.envs[0] if isinstance(vec, VecNormalize) else vec.envs[0]
    return getattr(base, "unwrapped", base)


# --------------------------------------------------------------------------
# 2. Lecture d'état
# --------------------------------------------------------------------------

def posture_deg(env):
    """(roll, pitch, yaw) en degrés depuis le quaternion.
    C'est ce qui révèle le banking soutenu (roll -40°/-70°, pitch +30°/+58° sur sac9)."""
    try:
        w, x, y, z = np.asarray(env.data.qpos[3:7], dtype=float)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.degrees([roll, pitch, yaw])
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def position(env):
    try:
        return np.asarray(env.data.qpos[:3], dtype=float)
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def _extraire(info, *cles, defaut=None):
    for c in cles:
        if isinstance(info, dict) and c in info:
            return info[c]
    return defaut


# --------------------------------------------------------------------------
# 3. Un épisode
# --------------------------------------------------------------------------

def jouer_episode(model, vec, render=False, delai=0.02, deterministic=True, seed=None):
    """Joue UN épisode complet. Retourne un dict de métriques."""
    if seed is not None:
        vec.seed(seed)
    obs = vec.reset()
    env = _env_brut(vec)

    rolls, pitches, xs = [], [], []
    reward_total, steps = 0.0, 0
    info = {}

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = vec.step(action)
        reward_total += float(reward[0])
        steps += 1

        r, p, _ = posture_deg(env)
        rolls.append(r)
        pitches.append(p)
        xs.append(position(env)[0])

        if render:
            try:
                env.render()
            except Exception:
                pass
            time.sleep(delai)

        if dones[0]:
            info = infos[0] if infos else {}
            break

    rolls, pitches, xs = np.array(rolls), np.array(pitches), np.array(xs)
    cause = _extraire(info, "cause_mort", "cause", defaut="inconnue")
    succes = _extraire(info, "succes", "success", "is_success")
    if succes is None:
        succes = cause in ("arrivee", "arrivée", "succes", "success", None)

    return {
        "succes": bool(succes),
        "cause_mort": cause,
        "distance_x": float(np.nanmax(xs)) if len(xs) else float("nan"),
        "steps": steps,
        "reward": reward_total,
        "roll_moy": float(np.nanmean(rolls)),
        "roll_abs_max": float(np.nanmax(np.abs(rolls))),
        "pitch_moy": float(np.nanmean(pitches)),
        "pitch_abs_max": float(np.nanmax(np.abs(pitches))),
        # part du temps en posture engagée : marge de manœuvre déjà consommée
        "temps_incline_30": float(np.mean((np.abs(rolls) > 30) | (np.abs(pitches) > 30))),
    }


# --------------------------------------------------------------------------
# 4. N épisodes → chiffres agrégés
# --------------------------------------------------------------------------

def evaluer(model=None, vec=None, episodes=30, densite=0.2, longueur=300.0,
            modele=None, vecnorm=None, racine=None,
            seed_depart=0, verbeux=True, deterministic=True):
    """Évalue une densité sur N épisodes. Seeds fixes (seed_depart + i) pour que
    deux modèles soient comparés sur exactement les mêmes forêts."""
    ferme = False
    if model is None or vec is None:
        model, vec = charger(modele, vecnorm, densite=densite, longueur=longueur,
                             racine=racine, verbeux=verbeux)
        ferme = True

    resultats = [
        jouer_episode(model, vec, render=False, deterministic=deterministic, seed=seed_depart + i)
        for i in range(episodes)
    ]
    if ferme:
        vec.close()

    def moy(cle):
        return float(np.nanmean([r[cle] for r in resultats]))

    stats = {
        "densite": densite,
        "episodes": episodes,
        "succes_pct": 100.0 * np.mean([r["succes"] for r in resultats]),
        "distance_moy": moy("distance_x"),
        "distance_med": float(np.nanmedian([r["distance_x"] for r in resultats])),
        "reward_moy": moy("reward"),
        "roll_moy": moy("roll_moy"),
        "roll_abs_max": float(np.nanmax([r["roll_abs_max"] for r in resultats])),
        "pitch_moy": moy("pitch_moy"),
        "pitch_abs_max": float(np.nanmax([r["pitch_abs_max"] for r in resultats])),
        "temps_incline_30": moy("temps_incline_30"),
        "causes": Counter(r["cause_mort"] for r in resultats if not r["succes"]),
    }

    if verbeux:
        print(f"\n— densité {densite} sur {episodes} épisodes —")
        print(f"  succès           : {stats['succes_pct']:.0f}%")
        print(f"  distance moy/méd : {stats['distance_moy']:.0f}m / {stats['distance_med']:.0f}m")
        print(f"  posture moy      : roll {stats['roll_moy']:+.0f}°  pitch {stats['pitch_moy']:+.0f}°")
        print(f"  posture pic      : |roll| {stats['roll_abs_max']:.0f}°  |pitch| {stats['pitch_abs_max']:.0f}°")
        print(f"  temps >30° incl. : {100 * stats['temps_incline_30']:.0f}%")
        if stats["causes"]:
            print("  causes de mort   : " +
                  ", ".join(f"{k}×{v}" for k, v in stats["causes"].most_common()))
    return stats


def balayage_densites(modele=None, vecnorm=None, densites=DENSITES_STANDARD,
                      episodes=30, longueur=300.0, seed_depart=0,
                      racine=None, nom=None):
    """Test principal : une ligne par densité, format prêt à coller dans EXPERIMENTS.md."""
    chemin = resoudre_modele(modele, racine)
    nom = nom or chemin.stem

    lignes = []
    for i, d in enumerate(densites):
        model, vec = charger(chemin, vecnorm, densite=d, longueur=longueur,
                             racine=racine, verbeux=(i == 0))
        lignes.append(evaluer(model, vec, episodes=episodes, densite=d,
                              seed_depart=seed_depart))
        vec.close()

    print(f"\n### {nom} — {episodes} ép./densité, seeds {seed_depart}..{seed_depart + episodes - 1}\n")
    print("| densité | succès | dist. moy | roll moy | pitch moy | temps >30° | cause principale |")
    print("|---|---|---|---|---|---|---|")
    for s in lignes:
        principale = s["causes"].most_common(1)[0][0] if s["causes"] else "—"
        print(f"| {s['densite']} | {s['succes_pct']:.0f}% | {s['distance_moy']:.0f}m | "
              f"{s['roll_moy']:+.0f}° | {s['pitch_moy']:+.0f}° | "
              f"{100 * s['temps_incline_30']:.0f}% | {principale} |")
    return lignes


def comparer_modeles(runs, densites=DENSITES_STANDARD, episodes=30,
                     seed_depart=0, racine=None):
    """runs : liste de motifs/chemins, ou dict {nom: motif}. Mêmes seeds pour tous."""
    if not isinstance(runs, dict):
        runs = {str(r): r for r in runs}

    tout = {nom: balayage_densites(motif, None, densites, episodes,
                                   seed_depart=seed_depart, racine=racine, nom=nom)
            for nom, motif in runs.items()}

    print("\n### Comparatif succès (%)\n")
    print("| densité | " + " | ".join(tout) + " |")
    print("|---" * (len(tout) + 1) + "|")
    for i, d in enumerate(densites):
        print(f"| {d} | " + " | ".join(f"{tout[n][i]['succes_pct']:.0f}%" for n in tout) + " |")
    return tout


def visualiser(modele=None, vecnorm=None, densite=0.35, episodes=3,
               longueur=300.0, delai=0.02, deterministic=True, racine=None):
    """Regarder voler, sans boucle infinie : N épisodes puis stop."""
    model, vec = charger(modele, vecnorm, densite=densite, longueur=longueur,
                         render=True, racine=racine)
    try:
        for i in range(episodes):
            r = jouer_episode(model, vec, render=True, delai=delai,
                              deterministic=deterministic, seed=i)
            etat = "ARRIVÉ" if r["succes"] else f"mort ({r['cause_mort']})"
            print(f"ép.{i}: {etat} — {r['distance_x']:.0f}m — "
                  f"roll moy {r['roll_moy']:+.0f}° / pitch moy {r['pitch_moy']:+.0f}°")
    except KeyboardInterrupt:
        print("\nArrêt manuel.")
    finally:
        vec.close()


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Test des checkpoints SAC du drone speedrunner")
    p.add_argument("commande", choices=["lister", "regarder", "eval", "balayage", "comparer"])
    p.add_argument("--run", default=None,
                   help="Chemin complet OU fragment de nom (ex: sac12). Défaut : le plus avancé.")
    p.add_argument("--vecnorm", default=None, help="Défaut : apparié automatiquement au run.")
    p.add_argument("--contre", default=None, help="Second run pour 'comparer'.")
    p.add_argument("--racine", default=None, help="Dossier de recherche des checkpoints.")
    p.add_argument("--densite", type=float, default=0.35)
    p.add_argument("--longueur", type=float, default=300.0)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed-depart", type=int, default=0)
    p.add_argument("--stochastique", action="store_true", help="predict(deterministic=False)")
    a = p.parse_args()
    det = not a.stochastique

    if a.commande == "lister":
        for c in lister_checkpoints(a.racine):
            v = resoudre_vecnorm(c, a.racine)
            print(f"{c}\n    ↳ {v.name if v else '(aucun vecnorm trouvé)'}")
    elif a.commande == "regarder":
        visualiser(a.run, a.vecnorm, a.densite, min(a.episodes, 5), a.longueur,
                   deterministic=det, racine=a.racine)
    elif a.commande == "eval":
        evaluer(episodes=a.episodes, densite=a.densite, longueur=a.longueur,
                modele=a.run, vecnorm=a.vecnorm, racine=a.racine,
                seed_depart=a.seed_depart, deterministic=det)
    elif a.commande == "balayage":
        balayage_densites(a.run, a.vecnorm, DENSITES_STANDARD, a.episodes,
                          a.longueur, a.seed_depart, racine=a.racine)
    elif a.commande == "comparer":
        if not a.contre:
            p.error("comparer exige --contre (ex: --run sac12 --contre sac9)")
        comparer_modeles({a.run or "recent": a.run, a.contre: a.contre},
                         episodes=a.episodes, seed_depart=a.seed_depart, racine=a.racine)


if __name__ == "__main__":
    main()
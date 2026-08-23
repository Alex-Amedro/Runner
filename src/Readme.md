# Drone Speedrunner

Autonomous obstacle avoidance in cluttered environments, trained with reinforcement learning.

<!-- HERO GIF: forest flythrough, dense obstacles, ~5-6s loop -->
<!-- ![hero](assets/hero.gif) -->

## Results

Average distance flown before failure, 100 episodes per configuration
(1 episode at zero density, which is deterministic):

| Obstacle density | Average distance before failure |
|---|---|
| none | TBD |
| low | TBD |
| medium | TBD |
| high | TBD |

<!-- Optional: LiDAR visualization GIF near this section -->
<!-- ![lidar](assets/lidar.gif) -->

## Environment

- **Perception**: 128-ray 2D LiDAR, non-uniform layout concentrated toward the front, gyroscopically stabilized so roll and pitch do not tilt the scan out of the horizontal plane.
- **Observation**: LiDAR rays, altitude, linear and angular velocity, orientation quaternion, remaining distance, lateral position. Last 3 frames stacked to expose obstacle relative velocity.
- **Action**: continuous thrust, roll, pitch, yaw torques. Throttle is offset so hover thrust is the default output rather than something the policy has to discover from zero.
- **Reward**: forward progress, proximity penalties (obstacles, ground), penalties on sustained heading and attitude deviation, a soft speed cap, a fixed collision penalty, a completion bonus.
- **Termination**: collision (tree, wall, ground), altitude ceiling, flip, lateral exit, excessive rotation rate.
- **Curriculum**: obstacle density and track length are both randomized over their full range every episode, not increased progressively. Narrowing the range to only hard cases caused catastrophic forgetting in an earlier version of this project.

## Training

- Algorithm: Soft Actor-Critic (Stable-Baselines3).
- Multiple environments run in parallel across CPU cores to collect experience faster.
- All reward and curriculum coefficients are exposed as CLI arguments.
- An earlier PPO baseline plateaued around 20 to 40 percent success once obstacles were present.

Full run-by-run history, with hypotheses written before each test, is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Repository structure

| File | Purpose |
|---|---|
| `environment.py` | environment definition |
| `train_sac.py` | trains SAC |
| `train.py` | trains PPO (earlier baseline) |
| `enjoy.py` | evaluates and renders a trained model |
| `check_env.py` | environment sanity checks |
| `mesure_distance.py` | parallel evaluation across many episodes |
| `EXPERIMENTS.md` | full experiment log |

## Usage

```bash
python train_sac.py --run-id my_run --timesteps 1500000

python enjoy.py --modele ./models/my_run_final.zip \
                 --vecnorm ./models/my_run_final_vecnorm.pkl \
                 --algo sac --densite 0.3 --longueur 150 --episodes 30

python check_env.py --modele ./models/my_run_final.zip \
                     --vecnorm ./models/my_run_final_vecnorm.pkl \
                     --algo sac --densite 0.0 --longueur 300 --episodes 3
```

Run each script with `--help` for the full set of options.

## What's next

Multi-drone extension in progress: multiple drones coordinating to intercept a moving target, sharing detections with each other. Not yet implemented.

## Stack

MuJoCo, Gymnasium, Stable-Baselines3, NumPy.

---

Shared for demonstration purposes. All rights reserved.
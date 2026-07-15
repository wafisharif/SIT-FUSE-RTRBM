"""
Data generation script for myoArmReachFixed-v0.
Generates trajectory data using a random policy.
Each episode saved as a separate .npy file (one per trial),
matching SFTemporalDataset's multi-trial loading convention.

Per Data Note:
- Environment: myoArmReachFixed-v0 (simplest reach task, lowest dimensional)
- Policy: random (diverse, plausible movement)
- Features recorded: full observation vector (qpos, qvel, act, end-effector)
- n_visible: 78 (confirmed from environment)
- Start small per Nick's guidance: 20 episodes
"""
import os
import numpy as np
import myosuite
import gymnasium as gym


def generate_episodes(
    env_name="myoArmReachFixed-v0",
    n_episodes=20,
    out_dir="data/reach_fixed",
    seed=42,
):
    """
    Runs n_episodes of random-policy rollouts and saves each as a
    separate .npy file of shape (n_timesteps, n_visible).

    :param env_name: MyoSuite environment name.
    :param n_episodes: Number of episodes to generate.
    :param out_dir: Directory to save .npy files.
    :param seed: Random seed for reproducibility.
    """
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(env_name)
    env.action_space.seed(seed)
    np.random.seed(seed)

    print(f"Environment: {env_name}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Generating {n_episodes} episodes -> {out_dir}/")
    print()

    episode_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        trajectory = [obs]
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(obs)
            done = terminated or truncated

        trajectory = np.array(trajectory, dtype=np.float32)
        episode_lengths.append(len(trajectory))

        fname = os.path.join(out_dir, f"episode_{ep:03d}.npy")
        np.save(fname, trajectory)

        print(f"  Episode {ep:03d}: {len(trajectory)} timesteps -> {fname}")

    env.close()

    print()
    print(f"Done. {n_episodes} episodes saved to {out_dir}/")
    print(f"Episode lengths: min={min(episode_lengths)}, "
          f"max={max(episode_lengths)}, "
          f"mean={np.mean(episode_lengths):.1f}")
    print(f"n_visible confirmed: {trajectory.shape[1]}")


if __name__ == "__main__":
    generate_episodes()

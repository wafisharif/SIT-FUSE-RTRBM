"""
Quick visualization of raw myoArmReachFixed-v0 trajectory data.
Per Nick's guidance: step through visualizations as you go, start small,
make sure everything looks sane before training.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "data/reach_fixed"
N_EPISODES_TO_PLOT = 5
N_FEATURES_TO_PLOT = 6  # plot first 6 features (first few qpos values)

# Load a few episodes
episodes = []
for i in range(N_EPISODES_TO_PLOT):
    fpath = os.path.join(DATA_DIR, f"episode_{i:03d}.npy")
    ep = np.load(fpath)
    episodes.append(ep)

print(f"Loaded {N_EPISODES_TO_PLOT} episodes")
print(f"Shape of episode 0: {episodes[0].shape}")
print(f"Value range across all episodes:")
all_data = np.concatenate(episodes, axis=0)
print(f"  min: {all_data.min():.4f}")
print(f"  max: {all_data.max():.4f}")
print(f"  mean: {all_data.mean():.4f}")
print(f"  std: {all_data.std():.4f}")

# Plot 1: first 6 features across time for each episode
fig, axes = plt.subplots(N_FEATURES_TO_PLOT, 1, figsize=(12, 10))
fig.suptitle(
    "Raw trajectories — first 6 features, 5 episodes\n(myoArmReachFixed-v0, random policy)", fontsize=12)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for feat_idx, ax in enumerate(axes):
    for ep_idx, ep in enumerate(episodes):
        ax.plot(ep[:, feat_idx], color=colors[ep_idx], alpha=0.7,
                label=f"ep {ep_idx}" if feat_idx == 0 else "")
    ax.set_ylabel(f"feat {feat_idx}", fontsize=8)
    ax.tick_params(labelsize=7)
    if feat_idx == 0:
        ax.legend(loc="upper right", fontsize=7)
    if feat_idx < N_FEATURES_TO_PLOT - 1:
        ax.set_xticks([])

axes[-1].set_xlabel("Timestep")
plt.tight_layout()
plt.savefig("experiments/reach_trajectories.png", dpi=120)
plt.close()
print("\nSaved: experiments/reach_trajectories.png")

# Plot 2: feature variance across episodes
# If all episodes look identical, random policy might not be diverse enough
fig2, ax2 = plt.subplots(figsize=(12, 4))
fig2.suptitle(
    "Per-feature std across 5 episodes\n(higher = more variation between episodes)", fontsize=11)
stds = np.array([ep.std(axis=0) for ep in episodes]).mean(axis=0)
ax2.bar(range(len(stds)), stds, color='steelblue', alpha=0.8)
ax2.set_xlabel("Feature index")
ax2.set_ylabel("Mean std across episodes")
ax2.axhline(stds.mean(), color='red', linestyle='--',
            label=f"mean={stds.mean():.3f}")
ax2.legend()
plt.tight_layout()
plt.savefig("experiments/reach_feature_variance.png", dpi=120)
plt.close()
print("Saved: experiments/reach_feature_variance.png")

print("\nDone. Check experiments/ folder for both plots.")

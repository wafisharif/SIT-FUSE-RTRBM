"""
RTVarianceGaussianRBM training experiment on real MyoSuite biomechanical data.

Key difference from RTGaussianRBM: learns per-feature variance (sigma),
so no per-batch normalization needed. Each feature's scale is learned
directly from data, fixing the flat-line issue on low-variance features.

Training strategy per Cho et al. (2011) Fig 1b:
  Phase 1 (warmup): freeze sigma, train W/a/b/W_prime/h0 only
  Phase 2 (full):   unfreeze sigma, continue training all parameters

This prevents early divergence from sigma learning before W has stabilized.
"""
from sit_fuse_rtrbm.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
from sit_fuse_rtrbm.datasets.sf_temporal_dataset import SFTemporalDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import glob
import os
import logging
logging.disable(logging.CRITICAL)

matplotlib.use("Agg")


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/reach_fixed"
OUT_DIR = "experiments/reach_training_variance"
SEQ_LEN = 20
N_VISIBLE = 78
N_HIDDEN = 64
BATCH_SIZE = 32
WARMUP_EPOCHS = 15   # freeze sigma (per Cho et al. 2011)
FULL_EPOCHS = 15   # unfreeze sigma
LR = 0.001  # small LR per Cho et al. 2011 -- GBRBMs sensitive
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load and split ────────────────────────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.npy")))
print(f"Found {len(all_files)} episodes")

np.random.seed(42)
indices = np.random.permutation(len(all_files))
train_files = [all_files[i] for i in indices[:14]]
val_files = [all_files[i] for i in indices[14:17]]
test_files = [all_files[i] for i in indices[17:]]
print(
    f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

# ── Scale (MinMaxScaler on train only) ───────────────────────────────────────
# NOTE: RTVarianceGaussianRBM doesn't require normalization -- sigma learns
# the per-feature scale. But MinMaxScaler still helps keep values in a
# reasonable range for initialization (sigma starts at 1.0, so values
# roughly in [0,1] are a good starting point before sigma adapts).
scaler = MinMaxScaler()
for fpath in train_files:
    scaler.partial_fit(np.load(fpath).astype(np.float32))
print("Scaler fitted.")


def load_and_scale(files, scaler, seq_len, do_shuffle):
    all_data, all_targets = [], []
    for i, fpath in enumerate(files):
        ep = scaler.transform(np.load(fpath).astype(np.float32))
        targets = np.array([(i, t) for t in range(len(ep))])
        all_data.append(ep)
        all_targets.append(targets)
    ds = SFTemporalDataset()
    ds.init_from_array(
        np.concatenate(all_data), np.concatenate(all_targets),
        seq_len=seq_len, do_shuffle=do_shuffle
    )
    return ds


train_ds = load_and_scale(train_files, scaler, SEQ_LEN, do_shuffle=True)
val_ds = load_and_scale(val_files,   scaler, SEQ_LEN, do_shuffle=False)
print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

sample_seq, _ = train_ds[0]
print(f"Sequence shape: {sample_seq.shape}")
print(f"Value range: [{sample_seq.min():.3f}, {sample_seq.max():.3f}]")

# ── Build model ───────────────────────────────────────────────────────────────
model = RTVarianceGaussianRBM(
    n_visible=N_VISIBLE,
    n_hidden=N_HIDDEN,
    steps=1,
    learning_rate=LR,
    momentum=0.0,
    decay=0.0,
    temperature=1.0,
    use_gpu=False,
)
print(
    f"\nModel: RTVarianceGaussianRBM(n_visible={N_VISIBLE}, n_hidden={N_HIDDEN})")
print(
    f"Training: {WARMUP_EPOCHS} warmup epochs (sigma frozen) + {FULL_EPOCHS} full epochs")

# ── Phase 1: warmup with sigma frozen ────────────────────────────────────────
print(f"\nPhase 1: warmup ({WARMUP_EPOCHS} epochs, sigma frozen)...")
model.sigma.requires_grad_(False)
model.fit(train_ds, batch_size=BATCH_SIZE, epochs=WARMUP_EPOCHS)
warmup_history = model.history["mse"].copy()
print(f"Warmup final MSE: {warmup_history[-1]:.4f}")
print(
    f"Sigma values after warmup (should still be ~1.0): mean={model.sigma.mean().item():.4f}")

# ── Phase 2: full training with sigma learning ────────────────────────────────
print(f"\nPhase 2: full training ({FULL_EPOCHS} epochs, sigma learning)...")
model.sigma.requires_grad_(True)
model.fit(train_ds, batch_size=BATCH_SIZE, epochs=FULL_EPOCHS)
full_history = model.history["mse"][WARMUP_EPOCHS:].copy()
all_history = model.history["mse"].copy()

print(f"Full training final MSE: {all_history[-1]:.4f}")
print(
    f"Sigma after full training: min={model.sigma.min().item():.4f}, max={model.sigma.max().item():.4f}, mean={model.sigma.mean().item():.4f}")

# ── Reconstruct on val ────────────────────────────────────────────────────────
print("\nReconstructing on val split...")
val_mse, val_probs = model.reconstruct(val_ds)
print(f"Val MSE: {val_mse:.4f}")

# ── Plot 1: Training MSE ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(all_history)+1), all_history, color="green", linewidth=2)
ax.axvline(x=WARMUP_EPOCHS, color="gray", linestyle="--", alpha=0.7,
           label=f"sigma unfrozen (epoch {WARMUP_EPOCHS})")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title(f"RTVarianceGaussianRBM Training MSE — myoArmReachFixed-v0\n"
             f"(n_visible={N_VISIBLE}, n_hidden={N_HIDDEN}, seq_len={SEQ_LEN}, lr={LR})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_mse_variance.png"), dpi=120)
plt.close()
print("Saved: training_mse_variance.png")

# ── Plot 2: Learned sigma values ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
sigma_vals = model.sigma.detach().numpy()
ax.bar(range(N_VISIBLE), sigma_vals, color="green", alpha=0.7)
ax.axhline(1.0, color="gray", linestyle="--",
           alpha=0.5, label="initial sigma=1.0")
ax.set_xlabel("Feature index")
ax.set_ylabel("Learned sigma")
ax.set_title("Learned per-feature sigma — RTVarianceGaussianRBM\n"
             "Lower sigma = model more confident about this feature's value")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "learned_sigma.png"), dpi=120)
plt.close()
print("Saved: learned_sigma.png")

# ── Plot 3: Real vs reconstructed ────────────────────────────────────────────
N_TO_PLOT = 3
N_FEAT_SHOW = 4

fig, axes = plt.subplots(N_TO_PLOT, N_FEAT_SHOW, figsize=(14, 8))
fig.suptitle("Real vs Reconstructed — RTVarianceGaussianRBM (val split)\n"
             "Blue = real, Orange = reconstructed (no normalization needed)", fontsize=11)

for seq_idx in range(N_TO_PLOT):
    real_seq, _ = val_ds[seq_idx]
    recon = val_probs[seq_idx].detach().numpy()
    for feat_idx in range(N_FEAT_SHOW):
        ax = axes[seq_idx, feat_idx]
        ax.plot(real_seq[:, feat_idx], color="steelblue",
                label="real" if feat_idx == 0 else "")
        ax.plot(recon[:, feat_idx], color="darkorange",
                linestyle="--", label="recon" if feat_idx == 0 else "")
        ax.set_title(f"Seq {seq_idx}, feat {feat_idx}", fontsize=8)
        ax.tick_params(labelsize=7)
        if feat_idx == 0 and seq_idx == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(
    OUT_DIR, "real_vs_reconstructed_variance.png"), dpi=120)
plt.close()
print("Saved: real_vs_reconstructed_variance.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"RTVarianceGaussianRBM training complete.")
print(f"  Warmup MSE (epoch 1):      {warmup_history[0]:.4f}")
print(f"  Warmup MSE (final):        {warmup_history[-1]:.4f}")
print(f"  Full training MSE (final): {all_history[-1]:.4f}")
print(f"  Val reconstruction MSE:    {val_mse:.4f}")
print(
    f"  Learned sigma range:       [{sigma_vals.min():.4f}, {sigma_vals.max():.4f}]")
print(f"  Plots saved to:            {OUT_DIR}/")
print(f"{'='*50}")

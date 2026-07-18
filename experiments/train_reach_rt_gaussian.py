"""
Gaussian RTRBM training experiment on real MyoSuite biomechanical data.
Companion to train_reach_rtrbm.py (Bernoulli) -- same data, same split,
same hyperparameters where possible, so results are directly comparable.

Key difference: RTGaussianRBM uses continuous-valued visible units
instead of binary Bernoulli units, giving smoother reconstructions
for continuous biomechanical data like joint angles and velocities.

Environment: myoArmReachFixed-v0
Episodes: 20 (each 151 timesteps, n_visible=78)
Split: by episode -- 14 train / 3 val / 3 test (same seed as Bernoulli run)
"""
from sit_fuse_rtrbm.temporal.rt_gaussian_rbm import RTGaussianRBM
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
OUT_DIR = "experiments/reach_training_gaussian"
BERN_DIR = "experiments/reach_training"
SEQ_LEN = 20
N_VISIBLE = 78
N_HIDDEN = 64
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.01
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all episode file paths ───────────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.npy")))
print(f"Found {len(all_files)} episodes")

# ── Same split as Bernoulli run (same seed=42) for fair comparison ────────────
np.random.seed(42)
indices = np.random.permutation(len(all_files))
train_files = [all_files[i] for i in indices[:14]]
val_files = [all_files[i] for i in indices[14:17]]
test_files = [all_files[i] for i in indices[17:]]
print(
    f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

# ── Fit scaler on training data only ─────────────────────────────────────────
scaler = MinMaxScaler()
for fpath in train_files:
    ep = np.load(fpath).astype(np.float32)
    scaler.partial_fit(ep)
print("Scaler fitted on training data.")

# ── Build datasets ────────────────────────────────────────────────────────────


def load_and_scale(files, scaler, seq_len, do_shuffle):
    all_data, all_targets = [], []
    for trial_idx, fpath in enumerate(files):
        ep = np.load(fpath).astype(np.float32)
        ep_scaled = scaler.transform(ep)
        targets = np.array([(trial_idx, t) for t in range(len(ep))])
        all_data.append(ep_scaled)
        all_targets.append(targets)
    combined_data = np.concatenate(all_data, axis=0)
    combined_targets = np.concatenate(all_targets, axis=0)
    ds = SFTemporalDataset()
    ds.init_from_array(combined_data, combined_targets,
                       seq_len=seq_len, do_shuffle=do_shuffle)
    return ds


train_ds = load_and_scale(train_files, scaler, SEQ_LEN, do_shuffle=True)
val_ds = load_and_scale(val_files,   scaler, SEQ_LEN, do_shuffle=False)

print(f"Train sequences: {len(train_ds)}")
print(f"Val sequences:   {len(val_ds)}")

sample_seq, _ = train_ds[0]
print(f"Sample sequence shape: {sample_seq.shape}")
print(f"Scaled value range: [{sample_seq.min():.3f}, {sample_seq.max():.3f}]")

# ── Build Gaussian model ──────────────────────────────────────────────────────
# normalize=False: data already scaled externally with MinMaxScaler,
# so we don't want additional per-batch normalization on top of that.
model = RTGaussianRBM(
    n_visible=N_VISIBLE,
    n_hidden=N_HIDDEN,
    steps=1,
    learning_rate=LR,
    momentum=0.0,
    decay=0.0,
    temperature=1.0,
    use_gpu=False,
    normalize=True,
    input_normalize=False,
)
print(f"\nModel: RTGaussianRBM(n_visible={N_VISIBLE}, n_hidden={N_HIDDEN})")

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
model.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS)

train_mse_history = model.history["mse"]
print(f"\nFinal train MSE: {train_mse_history[-1]:.4f}")

# Save MSE history for comparison
np.save(os.path.join(OUT_DIR, "mse_history.npy"),
        np.array(train_mse_history))

# ── Reconstruct on val split ──────────────────────────────────────────────────
print("\nReconstructing on validation split...")
val_mse, val_probs = model.reconstruct(val_ds)
print(f"Val reconstruction MSE: {val_mse:.4f}")

# Save val probs for comparison plot
np.save(os.path.join(OUT_DIR, "val_probs.npy"),
        val_probs.detach().numpy())

# ── Plot 1: Training MSE ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, EPOCHS + 1), train_mse_history,
        color="darkorange", linewidth=2, label="Gaussian RTRBM")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title(f"RTGaussianRBM Training MSE — myoArmReachFixed-v0\n"
             f"(n_visible={N_VISIBLE}, n_hidden={N_HIDDEN}, seq_len={SEQ_LEN})")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_mse_gaussian.png"), dpi=120)
plt.close()
print("Saved: training_mse_gaussian.png")

# ── Plot 2: Gaussian real vs reconstructed ───────────────────────────────────
N_TO_PLOT = 3
N_FEAT_SHOW = 4

fig, axes = plt.subplots(N_TO_PLOT, N_FEAT_SHOW, figsize=(14, 8))
fig.suptitle("Real vs Reconstructed — RTGaussianRBM (val split)\n"
             "Blue = real, Orange = reconstructed (continuous)", fontsize=11)

for seq_idx in range(N_TO_PLOT):
    real_seq, _ = val_ds[seq_idx]
    # Normalize real_seq to match what model sees during reconstruction
    real_flat = real_seq.reshape(-1, N_VISIBLE)
    real_norm = (real_flat - real_flat.mean(axis=0, keepdims=True)
                 ) / (real_flat.std(axis=0, keepdims=True) + 1e-6)
    real_seq_plot = real_norm.reshape(SEQ_LEN, N_VISIBLE)
    recon = val_probs[seq_idx].detach().numpy()
    for feat_idx in range(N_FEAT_SHOW):
        ax = axes[seq_idx, feat_idx]
        ax.plot(real_seq_plot[:, feat_idx], color="steelblue",
                label="real" if feat_idx == 0 else "")
        ax.plot(recon[:, feat_idx], color="darkorange",
                linestyle="--", label="recon" if feat_idx == 0 else "")
        ax.set_title(f"Seq {seq_idx}, feat {feat_idx}", fontsize=8)
        ax.tick_params(labelsize=7)
        if feat_idx == 0 and seq_idx == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(
    OUT_DIR, "real_vs_reconstructed_gaussian.png"), dpi=120)
plt.close()
print("Saved: real_vs_reconstructed_gaussian.png")

# ── Plot 3: Side-by-side Bernoulli vs Gaussian (the key comparison) ──────────
bern_probs_path = os.path.join(BERN_DIR, "val_probs.npy")
if os.path.exists(bern_probs_path):
    bern_probs = np.load(bern_probs_path)
    fig, axes = plt.subplots(2, N_FEAT_SHOW, figsize=(14, 6))
    fig.suptitle("Bernoulli vs Gaussian RTRBM — same val sequence\n"
                 "Blue = real, Orange = reconstructed", fontsize=11)

    real_seq, _ = val_ds[0]
    rows = [("Bernoulli RTRBM (binary outputs)", bern_probs[0]),
            ("Gaussian RTRBM (continuous outputs)", val_probs[0].detach().numpy())]

    for row_idx, (title, recon) in enumerate(rows):
        for feat_idx in range(N_FEAT_SHOW):
            ax = axes[row_idx, feat_idx]
            ax.plot(real_seq_plot[:, feat_idx], color="steelblue",
                    label="real" if feat_idx == 0 else "")
            ax.plot(recon[:, feat_idx], color="darkorange",
                    linestyle="--", label="recon" if feat_idx == 0 else "")
            if feat_idx == 0:
                ax.set_ylabel(title, fontsize=7)
            ax.set_title(f"feat {feat_idx}", fontsize=8)
            ax.tick_params(labelsize=7)
            if feat_idx == 0 and row_idx == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bernoulli_vs_gaussian.png"), dpi=120)
    plt.close()
    print("Saved: bernoulli_vs_gaussian.png")
else:
    print("Note: run train_reach_rtrbm.py with val_probs saving first for comparison plot")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Gaussian RTRBM training complete.")
print(f"  Train MSE (first epoch):  {train_mse_history[0]:.4f}")
print(f"  Train MSE (final epoch):  {train_mse_history[-1]:.4f}")
print(
    f"  MSE improvement:          {train_mse_history[0] - train_mse_history[-1]:.4f}")
print(f"  Val reconstruction MSE:   {val_mse:.4f}")
print(f"  Plots saved to:           {OUT_DIR}/")
print(f"{'='*50}")

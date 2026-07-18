"""
Initial RTRBM training experiment on real MyoSuite biomechanical data.
Environment: myoArmReachFixed-v0
Episodes: 20 (each 151 timesteps, n_visible=78)
Split: by episode (not by window) -- 14 train / 3 val / 3 test
Per Nick's guidance: start small, step through visualizations as you go.
"""
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM
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
OUT_DIR = "experiments/reach_training"
SEQ_LEN = 20      # confirmed with Nick
N_VISIBLE = 78      # confirmed from environment
N_HIDDEN = 64      # starting point -- reasonable for 78 visible units
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.01
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all episode file paths ───────────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.npy")))
print(f"Found {len(all_files)} episodes")

# ── Split by episode (not by window) -- per Data Note ────────────────────────
# 14 train / 3 val / 3 test
np.random.seed(42)
indices = np.random.permutation(len(all_files))
train_files = [all_files[i] for i in indices[:14]]
val_files = [all_files[i] for i in indices[14:17]]
test_files = [all_files[i] for i in indices[17:]]

print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test episodes")

# ── Fit scaler on training data only ─────────────────────────────────────────
# Load raw training data to fit scaler -- stats must come from train only
scaler = MinMaxScaler()
for fpath in train_files:
    ep = np.load(fpath).astype(np.float32)
    scaler.partial_fit(ep)

print(f"Scaler fitted on training data.")

# ── Build datasets ────────────────────────────────────────────────────────────
# Apply scaler externally, then load via init_from_array per episode


def load_and_scale(files, scaler, seq_len, do_shuffle):
    all_data = []
    all_targets = []
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

# Quick sanity check on scaled data range
sample_seq, _ = train_ds[0]
print(f"Sample sequence shape: {sample_seq.shape}")
print(f"Scaled value range: [{sample_seq.min():.3f}, {sample_seq.max():.3f}]")

# ── Build model ───────────────────────────────────────────────────────────────
model = RTRBM(
    n_visible=N_VISIBLE,
    n_hidden=N_HIDDEN,
    steps=1,
    learning_rate=LR,
    momentum=0.0,
    decay=0.0,
    temperature=1.0,
    use_gpu=False,
)
print(f"\nModel: RTRBM(n_visible={N_VISIBLE}, n_hidden={N_HIDDEN})")

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
model.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS)

train_mse_history = model.history["mse"]
print(f"\nFinal train MSE: {train_mse_history[-1]:.4f}")

# ── Reconstruct on val split ──────────────────────────────────────────────────
print("\nReconstructing on validation split...")
val_mse, val_probs = model.reconstruct(val_ds)
# Save val probs and MSE history for Gaussian comparison plot
np.save(os.path.join(OUT_DIR, "val_probs.npy"), val_probs.detach().numpy())
np.save(os.path.join(OUT_DIR, "mse_history.npy"), np.array(train_mse_history))
print(f"Val reconstruction MSE: {val_mse:.4f}")

# ── Plot 1: Training MSE over epochs ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, EPOCHS + 1), train_mse_history,
        color="steelblue", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title(f"RTRBM Training MSE — myoArmReachFixed-v0\n"
             f"(n_visible={N_VISIBLE}, n_hidden={N_HIDDEN}, seq_len={SEQ_LEN})")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_mse.png"), dpi=120)
plt.close()
print("Saved: experiments/reach_training/training_mse.png")

# ── Plot 2: Real vs reconstructed sequences (first 3 val sequences) ───────────
# Show 4 features side by side for visual inspection
N_TO_PLOT = 3
N_FEAT_TO_SHOW = 4

fig, axes = plt.subplots(N_TO_PLOT, N_FEAT_TO_SHOW, figsize=(14, 8))
fig.suptitle("Real vs Reconstructed sequences (val split)\n"
             "Blue = real, Orange = reconstructed", fontsize=11)

for seq_idx in range(N_TO_PLOT):
    real_seq, _ = val_ds[seq_idx]  # (seq_len, n_visible)
    real_seq_t = torch.from_numpy(real_seq).unsqueeze(
        0)  # (1, seq_len, n_visible)

    # Reconstruct this single sequence
    recon_probs = val_probs[seq_idx].detach().numpy()  # (seq_len, n_visible)

    for feat_idx in range(N_FEAT_TO_SHOW):
        ax = axes[seq_idx, feat_idx]
        ax.plot(real_seq[:, feat_idx], color="steelblue",
                label="real" if feat_idx == 0 else "")
        ax.plot(recon_probs[:, feat_idx], color="orange",
                linestyle="--", label="recon" if feat_idx == 0 else "")
        ax.set_title(f"Seq {seq_idx}, feat {feat_idx}", fontsize=8)
        ax.tick_params(labelsize=7)
        if feat_idx == 0 and seq_idx == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "real_vs_reconstructed.png"), dpi=120)
plt.close()
print("Saved: experiments/reach_training/real_vs_reconstructed.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Training complete.")
print(f"  Train MSE (final epoch): {train_mse_history[-1]:.4f}")
print(f"  Train MSE (first epoch): {train_mse_history[0]:.4f}")
print(
    f"  MSE improvement:         {train_mse_history[0] - train_mse_history[-1]:.4f}")
print(f"  Val reconstruction MSE:  {val_mse:.4f}")
print(f"  Plots saved to:          {OUT_DIR}/")
print(f"{'='*50}")

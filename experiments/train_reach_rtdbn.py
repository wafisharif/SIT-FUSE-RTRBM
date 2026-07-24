"""
End-to-end RTDBN experiment on real MyoSuite biomechanical data.

Pipeline:
  1. Load + scale MyoSuite reach episodes
  2. Train RTDBN encoder (RTVarianceGaussianRBM, warmup then full)
  3. Cluster temporal embeddings with k-means
  4. Visualize: cluster distribution, PCA of embeddings, example sequences

NOTE ON CLUSTERING APPROACH:
IIC requires training the encoder and clustering head jointly end-to-end
to learn discriminative features -- training a clustering head on frozen
RTRBM embeddings leads to collapse when embeddings lack discriminability.
Per SIT-FUSE's own history (BIRCH/k-means before transitioning to IIC),
k-means on frozen encoder embeddings is the correct starting point for
initial qualitative analysis. IIC joint training is a natural next step
once the pipeline is proven end-to-end.

Per Nick: start small, get to end-to-end output, sanity check what comes out.
"""
from sit_fuse_rtrbm.temporal.rtdbn import RTDBN
from sit_fuse_rtrbm.datasets.sf_temporal_dataset import SFTemporalDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
OUT_DIR = "experiments/reach_rtdbn"
SEQ_LEN = 20
N_VISIBLE = 78
N_HIDDEN = 64
N_CLUSTERS = 10
BATCH_SIZE = 32
WARMUP_EPOCHS = 15
FULL_EPOCHS = 15
ENCODER_LR = 0.001
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

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
for fpath in train_files:
    scaler.partial_fit(np.load(fpath).astype(np.float32))
print("Scaler fitted on training data only.")


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
all_ds = load_and_scale(
    train_files + val_files, scaler, SEQ_LEN, do_shuffle=False
)
print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")
print(f"All (train+val): {len(all_ds)} sequences")

# ── Build RTDBN ───────────────────────────────────────────────────────────────
model = RTDBN(
    model=("variance_gaussian",),
    n_visible=N_VISIBLE,
    n_hidden=(N_HIDDEN,),
    steps=(1,),
    learning_rate=(ENCODER_LR,),
    momentum=(0.0,),
    decay=(0.0,),
    temperature=(1.0,),
    use_gpu=False,
    n_clusters=N_CLUSTERS,
)
print(f"\nModel: RTDBN(n_visible={N_VISIBLE}, n_hidden={N_HIDDEN}, "
      f"n_clusters={N_CLUSTERS})")

# ── Phase 1: Train encoder ────────────────────────────────────────────────────
print(f"\nPhase 1: Training encoder "
      f"({WARMUP_EPOCHS} warmup + {FULL_EPOCHS} full epochs)...")

model.models[0].sigma.requires_grad_(False)
model.models[0].fit(train_ds, batch_size=BATCH_SIZE, epochs=WARMUP_EPOCHS)
warmup_history = model.models[0].history["mse"].copy()
print(f"Warmup final MSE: {warmup_history[-1]:.4f}")

model.models[0].sigma.requires_grad_(True)
model.models[0].fit(train_ds, batch_size=BATCH_SIZE, epochs=FULL_EPOCHS)
all_encoder_history = model.models[0].history["mse"].copy()
print(f"Encoder final MSE: {all_encoder_history[-1]:.4f}")

# ── Phase 2: Extract embeddings and cluster with k-means ──────────────────────
print(f"\nPhase 2: Extracting embeddings and clustering with k-means "
      f"(n_clusters={N_CLUSTERS})...")

all_embeddings, _ = model.get_cluster_assignments(
    all_ds, batch_size=BATCH_SIZE)
all_embeddings_np = all_embeddings.numpy()

print(f"Embedding shape: {all_embeddings_np.shape}")
print(
    f"Embedding range: [{all_embeddings_np.min():.4f}, {all_embeddings_np.max():.4f}]")
print(f"Embedding std: {all_embeddings_np.std():.4f}")

# K-means clustering
km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42)
all_assignments_np = km.fit_predict(all_embeddings_np)
print(f"K-means inertia: {km.inertia_:.4f}")

# Cluster distribution
unique, counts = np.unique(all_assignments_np, return_counts=True)
print("\nCluster distribution:")
for c, n in zip(unique, counts):
    print(
        f"  Cluster {c:2d}: {n:4d} sequences ({100*n/len(all_assignments_np):.1f}%)")

# ── Plot 1: Encoder training MSE ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(all_encoder_history)+1), all_encoder_history,
        color="green", linewidth=2)
ax.axvline(x=WARMUP_EPOCHS, color="gray", linestyle="--", alpha=0.7,
           label=f"sigma unfrozen (epoch {WARMUP_EPOCHS})")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("RTDBN Encoder Training MSE — myoArmReachFixed-v0")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "encoder_mse.png"), dpi=120)
plt.close()
print("\nSaved: encoder_mse.png")

# ── Plot 2: Cluster distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(unique, counts, color="steelblue", alpha=0.8)
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of sequences")
ax.set_title(f"K-means Cluster Distribution — RTDBN on myoArmReachFixed-v0\n"
             f"({len(all_assignments_np)} total sequences, {N_CLUSTERS} clusters)")
ax.set_xticks(range(N_CLUSTERS))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cluster_distribution.png"), dpi=120)
plt.close()
print("Saved: cluster_distribution.png")

# ── Plot 3: PCA of embeddings colored by cluster ─────────────────────────────
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(all_embeddings_np)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1],
    c=all_assignments_np, cmap="tab10",
    alpha=0.6, s=15, vmin=0, vmax=N_CLUSTERS-1
)
plt.colorbar(scatter, ax=ax, label="Cluster", ticks=range(N_CLUSTERS))
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA of RTDBN Temporal Embeddings — colored by k-means cluster\n"
             "Well-separated clusters = distinct movement patterns learned")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "embeddings_pca.png"), dpi=120)
plt.close()
print("Saved: embeddings_pca.png")

# ── Plot 4: Example sequences per cluster (3 largest clusters) ───────────────
top_clusters = unique[np.argsort(counts)[::-1]][:3]

fig, axes = plt.subplots(3, 4, figsize=(14, 9))
fig.suptitle("Example sequences per cluster — first 4 features\n"
             "Each row = one cluster, lines = different sequences in that cluster",
             fontsize=10)

colors = plt.cm.Set2(np.linspace(0, 1, 5))

for row_idx, cluster_id in enumerate(top_clusters):
    cluster_mask = all_assignments_np == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    n_examples = min(5, len(cluster_indices))
    example_indices = cluster_indices[:n_examples]

    for feat_idx in range(4):
        ax = axes[row_idx, feat_idx]
        for ex_idx, seq_idx in enumerate(example_indices):
            seq, _ = all_ds[int(seq_idx)]
            ax.plot(seq[:, feat_idx], color=colors[ex_idx],
                    alpha=0.7, linewidth=1.2)
        ax.set_title(f"Cluster {cluster_id}, feat {feat_idx}", fontsize=8)
        ax.tick_params(labelsize=7)
        if feat_idx == 0:
            ax.set_ylabel(
                f"Cluster {cluster_id}\n({counts[unique==cluster_id][0]} seqs)",
                fontsize=7
            )

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cluster_examples.png"), dpi=120)
plt.close()
print("Saved: cluster_examples.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"RTDBN end-to-end experiment complete.")
print(f"  Encoder MSE (epoch 1):        {all_encoder_history[0]:.4f}")
print(f"  Encoder MSE (final):          {all_encoder_history[-1]:.4f}")
print(f"  Clustering method:            K-means (n_clusters={N_CLUSTERS})")
print(f"  K-means inertia:              {km.inertia_:.4f}")
print(f"  Total sequences clustered:    {len(all_assignments_np)}")
print(f"  Clusters used:                {len(unique)}/{N_CLUSTERS}")
print(f"  Largest cluster:              {counts.max()} sequences")
print(f"  Smallest cluster:             {counts.min()} sequences")
print(f"  PCA variance explained (2D):  "
      f"{sum(pca.explained_variance_ratio_[:2])*100:.1f}%")
print(f"  Plots saved to:               {OUT_DIR}/")
print(f"{'='*55}")

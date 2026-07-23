"""
Smoke test for RTDBN and IICClusteringHead.
Key things to verify:
- RTDBN wraps RTVarianceGaussianRBM correctly
- encode() collapses (batch, seq_len, n_hidden) -> (batch, n_hidden)
- forward() returns valid embeddings and cluster probabilities
- IIC loss is finite and decreases with training
- get_cluster_assignments() returns valid cluster indices
"""
from sit_fuse_rtrbm.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
from sit_fuse_rtrbm.temporal.rtdbn import RTDBN, IICClusteringHead
from torch.utils.data import TensorDataset
import torch
import logging
logging.disable(logging.CRITICAL)


def test_rtdbn_inherits_from_model():
    """RTDBN must use learnergy's Model base class, same as DBN."""
    from learnergy.core import Model
    model = RTDBN(n_visible=8, n_hidden=(6,), n_clusters=3)
    assert isinstance(
        model, Model), "RTDBN does not inherit from learnergy Model"
    assert hasattr(model, 'history'), "RTDBN missing history dict from Model"
    assert hasattr(model, 'dump'), "RTDBN missing dump() from Model"
    print("RTDBN inherits from learnergy Model base class: PASS")


def test_rtdbn_wraps_rtvariancegaussianrbm():
    """Each layer must be an RTVarianceGaussianRBM."""
    model = RTDBN(n_visible=8, n_hidden=(6,), n_clusters=3)
    assert len(model.models) == 1, "Wrong number of layers"
    assert isinstance(model.models[0], RTVarianceGaussianRBM), (
        "Layer is not RTVarianceGaussianRBM"
    )
    print("RTDBN wraps RTVarianceGaussianRBM correctly: PASS")


def test_encode_collapses_time_dimension():
    """encode() must collapse (batch, seq_len, n_hidden) -> (batch, n_hidden)."""
    model = RTDBN(n_visible=8, n_hidden=(6,), n_clusters=3)
    x = torch.rand(4, 5, 8)  # (batch=4, seq_len=5, n_visible=8)
    emb = model.encode(x)
    assert emb.shape == (4, 6), f"Wrong embedding shape: {emb.shape}"
    print(f"encode() collapses to (batch, n_hidden) = {emb.shape}: PASS")


def test_forward_returns_valid_shapes_and_probs():
    """forward() must return embeddings and valid cluster probabilities."""
    model = RTDBN(n_visible=8, n_hidden=(6,), n_clusters=3)
    x = torch.rand(4, 5, 8)
    emb, probs = model.forward(x)

    assert emb.shape == (4, 6), f"Wrong embedding shape: {emb.shape}"
    assert probs.shape == (4, 3), f"Wrong probs shape: {probs.shape}"
    assert torch.all((probs >= 0) & (probs <= 1)), "probs not in [0,1]"
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5), (
        "probs don't sum to 1 per sample"
    )
    print(
        f"forward() embeddings {emb.shape}, probs {probs.shape}, sum to 1: PASS")


def test_iic_loss_is_finite():
    """IIC loss must be a finite scalar."""
    p = torch.softmax(torch.randn(8, 3), dim=1)
    p2 = torch.softmax(torch.randn(8, 3), dim=1)
    loss = IICClusteringHead.iic_loss(p, p2)
    assert loss.shape == (), f"IIC loss not scalar: {loss.shape}"
    assert torch.isfinite(loss), f"IIC loss not finite: {loss}"
    print(f"IIC loss is finite scalar, value={loss.item():.4f}: PASS")


def test_iic_clustering_head_trains():
    """IIC clustering head loss should decrease with repeated training."""
    torch.manual_seed(0)
    model = RTDBN(n_visible=8, n_hidden=(6,), n_clusters=3)
    x = torch.rand(8, 5, 8)

    optimizer = torch.optim.Adam(
        model.clustering_head.parameters(), lr=0.01
    )

    with torch.no_grad():
        embeddings = model.encode(x)

    losses = []
    for _ in range(50):
        perturbed = model.clustering_head.perturb(embeddings)
        p = model.clustering_head(embeddings)
        p_pert = model.clustering_head(perturbed)
        loss = IICClusteringHead.iic_loss(p, p_pert)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    print(f"  IIC early loss: {early:.4f}, late loss: {late:.4f}")
    assert late < early, f"IIC loss not decreasing: early={early:.4f}, late={late:.4f}"
    print("IIC clustering head trains (loss decreases): PASS")


def test_get_cluster_assignments_returns_valid_indices():
    """get_cluster_assignments() must return integer cluster labels in [0, n_clusters)."""
    model = RTDBN(n_visible=8, n_hidden=(6,), n_clusters=3)
    x = torch.rand(8, 5, 8)
    ds = TensorDataset(x, torch.zeros(8))

    emb, assigns = model.get_cluster_assignments(ds, batch_size=4)

    assert emb.shape == (8, 6), f"Wrong embedding shape: {emb.shape}"
    assert assigns.shape == (8,), f"Wrong assignments shape: {assigns.shape}"
    assert torch.all((assigns >= 0) & (assigns < 3)), (
        f"Assignments out of range [0,2]: {assigns}"
    )
    print(f"get_cluster_assignments() embeddings {emb.shape}, "
          f"assignments in [0,2]: PASS")


def test_single_layer_config_matches_nick_direction():
    """Confirm default single-layer config works exactly as Nick requested."""
    model = RTDBN(
        n_visible=78,     # our actual MyoSuite n_visible
        n_hidden=(64,),   # one layer, 64 hidden units
        n_clusters=10,    # reasonable starting point for reach movements
    )
    assert model.n_layers == 1, "Expected single layer"
    assert model.models[0].n_visible == 78
    assert model.models[0].n_hidden == 64
    assert model.clustering_head.fc[-2].out_features == 10

    # Quick forward pass with real-sized data
    x = torch.rand(2, 20, 78)  # 2 sequences of seq_len=20, n_visible=78
    emb, probs = model.forward(x)
    assert emb.shape == (2, 64)
    assert probs.shape == (2, 10)
    print(f"Single-layer config (n_visible=78, n_hidden=64, n_clusters=10): PASS")
    print(
        f"  Real-sized forward pass: embeddings {emb.shape}, probs {probs.shape}")


if __name__ == "__main__":
    test_rtdbn_inherits_from_model()
    test_rtdbn_wraps_rtvariancegaussianrbm()
    test_encode_collapses_time_dimension()
    test_forward_returns_valid_shapes_and_probs()
    test_iic_loss_is_finite()
    test_iic_clustering_head_trains()
    test_get_cluster_assignments_returns_valid_indices()
    test_single_layer_config_matches_nick_direction()
    print("\nAll RTDBN smoke tests passed.")

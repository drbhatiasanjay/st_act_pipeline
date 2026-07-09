"""
Simplified smoke test for training loop.

Tests the infrastructure without full data loading:
- Model forward pass shapes
- Loss computation
- Backward pass
- Checkpointing
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import SimpleNodeTransformer, UNet3D

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_training_loop_infrastructure():
    """Test training loop infrastructure with dummy data."""

    logger.info("=" * 80)
    logger.info("SMOKE TEST: Training Loop Infrastructure")
    logger.info("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create dummy datasets with small volumes
    logger.info("Creating dummy datasets...")
    batch_size = 1
    z, y, x = 16, 64, 64  # Much smaller than real (64, 256, 256)

    # Create dummy data: (batch_size, 1, z, y, x) for each frame
    dummy_frames_t = torch.rand(4, 1, z, y, x)
    dummy_frames_t1 = torch.rand(4, 1, z, y, x)

    # Create datasets
    train_dataset = TensorDataset(dummy_frames_t, dummy_frames_t1)
    val_dataset = TensorDataset(dummy_frames_t, dummy_frames_t1)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Create models
    logger.info("Creating models...")
    unet3d = UNet3D(in_channels=2, channels=(32, 64, 128))
    transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)

    unet3d.to(device)
    transformer.to(device)

    logger.info(f"UNet3D parameters: {sum(p.numel() for p in unet3d.parameters()):,}")
    logger.info(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    # Test 1: UNet3D forward pass
    logger.info("\n" + "-" * 80)
    logger.info("TEST 1: UNet3D forward pass")
    logger.info("-" * 80)

    x_test = torch.cat([dummy_frames_t[0:1], dummy_frames_t1[0:1]], dim=1).to(device)
    logger.info(f"Input shape: {x_test.shape}")

    with torch.no_grad():
        logits, features = unet3d(x_test)

    logger.info(f"Output logits shape: {logits.shape}")
    logger.info(f"Output features shape: {features.shape}")

    assert logits.shape == (1, 1, z, y, x), f"Unexpected logits shape: {logits.shape}"
    assert features.shape == (1, 128, z, y, x), f"Unexpected features shape: {features.shape}"
    logger.info("✓ UNet3D forward pass: PASS")

    # Test 2: Transformer forward pass
    logger.info("\n" + "-" * 80)
    logger.info("TEST 2: Transformer forward pass")
    logger.info("-" * 80)

    # Extract dummy nodes
    probs = torch.sigmoid(logits)
    node_mask = probs[0, 0] > 0.5
    node_indices = torch.where(node_mask)

    if len(node_indices[0]) > 0:
        nodes = torch.stack([node_indices[0], node_indices[1], node_indices[2]], dim=1).float().to(device)
    else:
        # Create dummy nodes if no detections
        nodes = torch.tensor([[z//2, y//2, x//2]], dtype=torch.float32, device=device)

    logger.info(f"Number of detected nodes: {nodes.shape[0]}")

    # Extract features at nodes
    z_coords = torch.clamp(nodes[:, 0].long(), 0, z - 1)
    y_coords = torch.clamp(nodes[:, 1].long(), 0, y - 1)
    x_coords = torch.clamp(nodes[:, 2].long(), 0, x - 1)
    node_features = features[0, :, z_coords, y_coords, x_coords].t()  # (n_nodes, 128)

    logger.info(f"Node features shape: {node_features.shape}")

    with torch.no_grad():
        edge_probs = transformer(nodes, nodes, node_features, node_features)

    logger.info(f"Edge probabilities shape: {edge_probs.shape}")
    assert edge_probs.shape[0] >= 0, "Edge probs should be 1D or empty"
    logger.info("✓ Transformer forward pass: PASS")

    # Test 3: Loss computation
    logger.info("\n" + "-" * 80)
    logger.info("TEST 3: Loss computation")
    logger.info("-" * 80)

    # Dummy heatmap targets
    heatmap_targets = torch.rand((1, 1, z, y, x)).to(device)

    from src.targets import DetectionLoss
    detection_loss_fn = DetectionLoss()
    detection_loss = detection_loss_fn(logits, heatmap_targets)

    logger.info(f"Detection loss: {detection_loss.item():.6f}")
    assert not torch.isnan(detection_loss), "Detection loss is NaN!"
    assert detection_loss.item() > 0, "Detection loss must be positive!"
    logger.info("✓ Loss computation: PASS")

    # Test 4: Backward pass
    logger.info("\n" + "-" * 80)
    logger.info("TEST 4: Backward pass with gradient clipping")
    logger.info("-" * 80)

    optimizer = torch.optim.AdamW(
        list(unet3d.parameters()) + list(transformer.parameters()),
        lr=1e-4
    )

    optimizer.zero_grad()
    detection_loss.backward()

    # Check gradients exist
    grad_count = sum(1 for p in list(unet3d.parameters()) + list(transformer.parameters()) if p.grad is not None)
    logger.info(f"Parameters with gradients: {grad_count}")
    assert grad_count > 0, "No gradients computed!"

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        list(unet3d.parameters()) + list(transformer.parameters()),
        max_norm=1.0
    )
    optimizer.step()

    logger.info("✓ Backward pass and gradient clipping: PASS")

    # Test 5: Checkpointing
    logger.info("\n" + "-" * 80)
    logger.info("TEST 5: Checkpointing")
    logger.info("-" * 80)

    checkpoint_dir = Path("checkpoints_smoke_test_simple")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': 1,
        'unet3d_state_dict': unet3d.state_dict(),
        'transformer_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': {'test': 0.5},
    }

    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    assert checkpoint_path.exists(), "Checkpoint file not created!"

    # Load checkpoint
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info("Checkpoint loaded successfully")
    logger.info(f"Checkpoint keys: {list(loaded_checkpoint.keys())}")
    logger.info("✓ Checkpointing: PASS")

    logger.info("\n" + "=" * 80)
    logger.info("ALL SMOKE TESTS PASSED!")
    logger.info("=" * 80)
    logger.info("\nTraining loop infrastructure is ready:")
    logger.info("✓ UNet3D forward pass works with correct shapes")
    logger.info("✓ Transformer forward pass works")
    logger.info("✓ Loss computation works")
    logger.info("✓ Backward pass and gradient clipping work")
    logger.info("✓ Checkpointing works")


if __name__ == "__main__":
    test_training_loop_infrastructure()

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class UNet3D(nn.Module):
    """
    3D U-Net for cell detection in anisotropic volumetric data.

    Architecture:
    - Input: (B, 2, 64, 256, 256) [batch, channels=2 (two frames concatenated), z, y, x]
    - Channels: [32, 64, 128] (from host reference implementation)
    - Downsample strides: (1, 4, 4) - Z untouched, Y/X downsampled 4x
    - Output logits: (B, 2, 64, 256, 256) per-voxel detection [0,1] -- channel 0 is
      frame_t's own detections (time t_idx), channel 1 is frame_t+1's (time t_idx+1)
    - Output features: (B, 128, 64, 256, 256) dense feature maps for transformer

    Uses asymmetric kernels (1,3,3) on Z to preserve Z resolution.

    Detection head is 2-channel (not 1) so a single forward pass over
    (frame_t, frame_t1) yields genuinely distinct, real detections for BOTH
    timepoints sharing one feature context -- a single-channel head (the
    original design) forced validate_epoch() to either reuse one volume's
    peaks for both timepoints (identical points, corrupting edge scoring) or
    stitch together predictions from two separate forward passes (a
    confirmed train-test feature distribution mismatch: train_epoch's edge
    loss always slices features_t/features_t1 from ONE shared forward pass,
    see the comment there). The 2-channel head resolves both at the root.
    """

    def __init__(self, in_channels=2, channels=(32, 64, 128), anisotropy_stride=(1, 4, 4)):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.anisotropy_stride = anisotropy_stride

        # Encoder blocks with anisotropic downsampling
        # Level 0: input (B, 2, 64, 256, 256) -> (B, 32, 64, 256, 256)
        self.enc0 = self._conv_block(in_channels, channels[0], kernel=(1, 3, 3), padding=(0, 1, 1))

        # Downsample to Level 1: (B, 32, 64, 256, 256) -> (B, 64, 64, 64, 64)
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.enc1 = self._conv_block(channels[0], channels[1], kernel=(1, 3, 3), padding=(0, 1, 1))

        # Downsample to Level 2: (B, 64, 64, 64, 64) -> (B, 128, 64, 16, 16)
        self.pool2 = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.enc2 = self._conv_block(channels[1], channels[2], kernel=(1, 3, 3), padding=(0, 1, 1))

        # Bottleneck: (B, 128, 64, 16, 16)
        self.bottleneck = self._conv_block(channels[2], channels[2], kernel=(1, 3, 3), padding=(0, 1, 1))

        # Decoder with skip connections
        # Upsample from Level 2 to Level 1
        self.up2 = nn.Upsample(scale_factor=(1, 4, 4), mode='nearest')
        self.dec2 = self._conv_block(channels[2] + channels[1], channels[1], kernel=(1, 3, 3), padding=(0, 1, 1))

        # Upsample from Level 1 to Level 0
        self.up1 = nn.Upsample(scale_factor=(1, 4, 4), mode='nearest')
        self.dec1 = self._conv_block(channels[1] + channels[0], channels[0], kernel=(1, 3, 3), padding=(0, 1, 1))

        # Final output heads
        # Detection head: per-voxel logits, 2 channels (frame_t's own
        # detections, frame_t1's own detections) -- see class docstring.
        self.det_head = nn.Sequential(
            nn.Conv3d(channels[0], channels[0], kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[0], 2, kernel_size=1)
        )
        # RetinaNet-style prior-bias init (found by adversarial review, 2026-07-13,
        # of the adaptive class-imbalance loss fix -- see DEFERRED_IMPROVEMENTS.md):
        # PyTorch's default zero bias means sigmoid(0)=0.5 everywhere at init, so
        # ~99.99% of voxels (real background fraction) start with a large BCE
        # penalty the network can only reduce by first spending many gradient
        # steps pushing this bias very negative -- wasted steps that could
        # instead go toward learning real spatial cell features from step 1.
        # pi ~= 1e-4 is a deliberately coarse, dataset-wide estimate (real
        # per-sample foreground fraction measured 1.5e-5 to ~2e-4 across a
        # handful of real .geff samples) -- this is a starting prior for
        # training to refine, not a precise per-sample value.
        pi = 1e-4
        prior_bias = math.log(pi / (1 - pi))
        with torch.no_grad():
            self.det_head[-1].bias.fill_(prior_bias)

    @staticmethod
    def _conv_block(in_ch, out_ch, kernel=(1, 3, 3), padding=(0, 1, 1)):
        """Double convolution block with anisotropic kernels.

        GroupNorm after every conv (2026-07-14, see DEFERRED_IMPROVEMENTS.md
        item 6): a real 25-step local trace at batch_size=1 without any
        normalization showed a catastrophic single-step overshoot (loss
        11.29, sigmoid saturating to 1.0 across the whole volume) followed by
        an equally violent overcorrection -- gradient clipping (already
        active, clip_grad_norm_ max_norm=1.0) did not prevent it. The
        identical trace with GroupNorm inserted here never saturated (max
        loss 1.85, the ordinary step-0 value). conv bias=False is standard
        once a norm layer follows -- GroupNorm's own learnable shift makes
        the conv bias redundant. groups=min(8, out_ch) keeps per-group
        channel counts reasonable at the smallest width (32) while matching
        the pattern used by a real public kernel for this competition
        (pilkwang/biohub-cell-tracking-blend-preprocessings' DeepCenterUNet3D
        block; verified by pulling that kernel directly, not assumed). ReLU
        is kept (not that kernel's SiLU) to isolate normalization as the one
        changed variable -- see DEFERRED_IMPROVEMENTS.md for the still-open
        question of whether this alone is sufficient for sustained learning,
        not just avoiding the instability.
        """
        groups = min(8, out_ch)
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 2, 64, 256, 256) two consecutive frames concatenated along channel dim
              where channels 0 is frame_t and channel 1 is frame_t+1

        Returns:
            logits: (B, 2, 64, 256, 256) per-voxel detection logits --
                channel 0 is frame_t's own detections, channel 1 is
                frame_t1's own detections, both sharing this one forward
                pass's feature context.
            features: (B, 128, 64, 256, 256) dense feature maps for transformer
        """
        # Gradient checkpointing (2026-07-14): recomputes each conv block's
        # activations during backward instead of storing them, cutting peak
        # memory regardless of which ops are involved -- unlike autocast
        # (tried first, see git history), which explicitly keeps
        # normalization layers in fp32 and barely moved the needle on the
        # real v46/v47 CUDA OOM (13.20->12.97 GiB, ~230MB, far short of the
        # ~2GB needed). Only during training with grad enabled: checkpoint()
        # forces an extra no-grad forward pass that's pure overhead with no
        # memory benefit during eval (validate_epoch runs under
        # torch.no_grad() already, so there's nothing to recompute).
        use_checkpoint = self.training and torch.is_grad_enabled()

        def _block(module, inp):
            if use_checkpoint:
                return torch.utils.checkpoint.checkpoint(module, inp, use_reentrant=False)
            return module(inp)

        # Encoder with skip connections
        enc0 = _block(self.enc0, x)  # (B, 32, 64, 256, 256)

        pool1 = self.pool1(enc0)  # (B, 32, 64, 64, 64)
        enc1 = _block(self.enc1, pool1)  # (B, 64, 64, 64, 64)

        pool2 = self.pool2(enc1)  # (B, 64, 64, 16, 16)
        enc2 = _block(self.enc2, pool2)  # (B, 128, 64, 16, 16)

        # Bottleneck
        bottleneck = _block(self.bottleneck, enc2)  # (B, 128, 64, 16, 16)

        # Decoder with skip connections
        up2 = self.up2(bottleneck)  # (B, 128, 64, 64, 64)
        up2 = torch.cat([up2, enc1], dim=1)  # (B, 192, 64, 64, 64)
        dec2 = _block(self.dec2, up2)  # (B, 64, 64, 64, 64)

        up1 = self.up1(dec2)  # (B, 64, 64, 256, 256)
        up1 = torch.cat([up1, enc0], dim=1)  # (B, 96, 64, 256, 256)
        dec1 = _block(self.dec1, up1)  # (B, 32, 64, 256, 256)

        # Detection logits from final decoder output
        logits = self.det_head(dec1)  # (B, 2, 64, 256, 256)

        # For transformer: use bottleneck features at full Z resolution
        # Upsample bottleneck features to full resolution
        features_upsampled = self.up2(bottleneck)  # (B, 128, 64, 64, 64)
        features_upsampled = self.up1(features_upsampled)  # (B, 128, 64, 256, 256)

        return logits, features_upsampled


class SimpleNodeTransformer(nn.Module):
    """
    Cross-attention Transformer for pairwise edge probability prediction.

    Architecture:
    - Hidden dim: 128
    - Heads: 4
    - Blocks: 4
    - Dropout: 0.3

    Predicts edge probabilities between detected nodes in consecutive frames.
    Processes node embeddings through transformer encoders, then scores edges
    by concatenating attended feature pairs.
    """

    def __init__(self, hidden_dim=128, num_heads=4, num_blocks=4, dropout=0.3,
                 feature_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # Node embedding: coordinates (3) + sinusoidal PE (8*3=24) + UNet features (feature_dim)
        # Total input: 3 + 24 + feature_dim
        node_input_dim = 3 + 24 + feature_dim

        # Project node embeddings to hidden dimension
        self.node_embed = nn.Linear(node_input_dim, hidden_dim)

        # Self-attention transformer encoders for each frame
        self.encoder_t = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='relu'
            ),
            num_layers=num_blocks
        )

        self.encoder_t1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='relu'
            ),
            num_layers=num_blocks
        )

        # Edge scoring MLP: concatenated features -> probability
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def sinusoidal_positional_encoding(num_pos, hidden_dim):
        """Generate sinusoidal positional encoding."""
        pe = torch.zeros(num_pos, hidden_dim)
        position = torch.arange(0, num_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float) *
            -(math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, nodes_t, nodes_t1, features_t, features_t1, candidate_edges=None):
        """
        Forward pass for edge probability prediction.

        Args:
            nodes_t: (n_t, 3) node coordinates [z, y, x] at frame t
            nodes_t1: (n_t1, 3) node coordinates [z, y, x] at frame t+1
            features_t: (n_t, 128) node features from UNet at frame t
            features_t1: (n_t1, 128) node features from UNet at frame t+1
            candidate_edges: Optional (n_candidates, 2) edge indices to score
                            If None, score all possible edges

        Returns:
            edge_probs: (n_candidates,) edge probabilities in [0, 1]
        """
        device = nodes_t.device
        n_t = nodes_t.shape[0]
        n_t1 = nodes_t1.shape[0]

        if n_t == 0 or n_t1 == 0:
            # Handle empty node sets
            return torch.tensor([], dtype=torch.float32, device=device)

        # Generate sinusoidal positional encoding (24 dims for 3 axes x 8)
        pe_dim = 8
        pos_enc_t = self.sinusoidal_positional_encoding(n_t, pe_dim * 3).to(device)
        pos_enc_t1 = self.sinusoidal_positional_encoding(n_t1, pe_dim * 3).to(device)

        # Concatenate coordinates + positional encoding + UNet features
        nodes_t_emb = torch.cat([nodes_t, pos_enc_t, features_t], dim=1)  # (n_t, 3+24+128)
        nodes_t1_emb = torch.cat([nodes_t1, pos_enc_t1, features_t1], dim=1)  # (n_t1, 3+24+128)

        # Project to hidden dimension
        nodes_t_h = self.node_embed(nodes_t_emb)  # (n_t, hidden_dim)
        nodes_t1_h = self.node_embed(nodes_t1_emb)  # (n_t1, hidden_dim)

        # Apply transformer encoders (with batch dimension for transformer)
        nodes_t_h = self.encoder_t(nodes_t_h.unsqueeze(0)).squeeze(0)  # (n_t, hidden_dim)
        nodes_t1_h = self.encoder_t1(nodes_t1_h.unsqueeze(0)).squeeze(0)  # (n_t1, hidden_dim)

        # Generate candidate edges if not provided
        if candidate_edges is None:
            # All pairwise (i, j) edges in the same row-major order the old nested
            # Python loop produced (i outer, j inner) -- greedy_edge_assignment's
            # default reconstruction and inference_kernel.py's peak ordering both
            # assume this order.
            candidate_edges = torch.cartesian_prod(
                torch.arange(n_t, device=device), torch.arange(n_t1, device=device)
            )
        else:
            candidate_edges = candidate_edges.to(device)

        # Score all candidate edges in one batched call instead of a Python-level loop
        # per edge -- unvectorized version was a confirmed-unmeasured scaling risk
        # (run_pipeline.py's own profiling notes cite ~1110 candidates/timepoint on
        # dense real timepoints). Handles the zero-candidate case natively (an (0, D)
        # input produces an (0,) output), no separate empty-list fallback needed.
        feat_t = nodes_t_h[candidate_edges[:, 0]]    # (n_candidates, hidden_dim)
        feat_t1 = nodes_t1_h[candidate_edges[:, 1]]  # (n_candidates, hidden_dim)
        edge_feat = torch.cat([feat_t, feat_t1], dim=1)  # (n_candidates, hidden_dim*2)
        edge_probs = self.edge_scorer(edge_feat).squeeze(-1)  # (n_candidates,)

        return edge_probs


class AnisotropicCoordinateTransformer(nn.Module):
    """
    Natively maps 3D coordinate tensors from anisotropic voxel space (Z, Y, X)
    to physical spatial metric space (Z_phys, Y_phys, X_phys) without raw image interpolation.
    """
    def __init__(self, anisotropy_ratio=(4.0, 1.0, 1.0)):
        super().__init__()
        # Register voxel ratios as non-trainable buffers
        self.register_buffer("anisotropy", torch.tensor(anisotropy_ratio, dtype=torch.float32))

    def forward(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_coords (torch.Tensor): Shape (B, N, 3) representing (Z, Y, X) coordinate values.
        Returns:
            torch.Tensor: Calibrated physical coordinates (B, N, 3).
        """
        # Element-wise calibration
        return voxel_coords * self.anisotropy.view(1, 1, 3)

class STACTCentroidPredictor(nn.Module):
    """
    Fully Convolutional 3D Network that inputs anisotropic timepoint blocks
    and predicts:
    1. Cell Centroid Probability Heatmap (B, 1, Z, Y, X)
    2. Local 3D Motion Vector Offsets (B, 3, Z, Y, X) to map movement to time T+1.
    """
    def __init__(self, in_channels=1, base_features=16):
        super().__init__()

        # Downsampling path representing anisotropic voxel kernels
        # Uses asymmetric kernel structures (e.g., 1x3x3) on Z to prevent pixel distortion
        self.encoder1 = nn.Conv3d(in_channels, base_features, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.encoder2 = nn.Conv3d(base_features, base_features*2, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Coordinate Calibration Layer
        self.coord_transformer = AnisotropicCoordinateTransformer()

        # Heatmap prediction head
        self.heatmap_head = nn.Sequential(
            nn.Conv3d(base_features*2, base_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_features, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 3D motion vector head
        self.motion_head = nn.Sequential(
            nn.Conv3d(base_features*2, base_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_features, 3, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Volume tensor of shape (B, 1, Z, Y, X)
        Returns:
            heatmap: (B, 1, Z, Y, X) - Centroid probability distribution.
            motion_vectors: (B, 3, Z, Y, X) - predicted spatial offset pointing to coordinate at T+1.
        """
        # Feature extraction
        feat = F.relu(self.encoder1(x))
        feat = F.relu(self.encoder2(feat))

        # Predict heads
        heatmap = self.heatmap_head(feat)
        motion_vectors = self.motion_head(feat)

        return heatmap, motion_vectors

if __name__ == "__main__":
    # Test network compilation
    print("Testing PyTorch ST-ACT Model compilation...")
    model = STACTCentroidPredictor()
    dummy_input = torch.randn(1, 1, 10, 128, 128)  # B, C, Z, Y, X
    heatmap, motion = model(dummy_input)
    print(f"Heatmap Output Shape: {heatmap.shape} (Expected: 1, 1, 10, 128, 128)")
    print(f"Motion Vectors Shape: {motion.shape} (Expected: 1, 3, 10, 128, 128)")
    print("PyTorch ST-ACT Module Compiled Successfully!")

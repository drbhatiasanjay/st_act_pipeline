import torch
import torch.nn as nn
import torch.nn.functional as F

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

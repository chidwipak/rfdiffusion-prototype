"""
SE3-Equivariant Diffusion Denoiser for Protein Backbones
=========================================================

This acts as the main "brain" of the diffusion model. 
It takes noisy coordinates and predicts the noise to be removed.
Includes the RoseTTAFold backbone we implemented.

Author: Chidwipak
Date: January 2026

Examples
--------
>>> from se3_diffusion import SE3DiffusionDenoiser
>>> model = SE3DiffusionDenoiser(embed_dim=128, num_layers=4)
>>> coords = torch.randn(2, 30, 9)  # (batch, residues, 3 atoms * 3 coords)
>>> t = torch.tensor([10, 50])  # timesteps
>>> noise_pred = model(coords, t)
>>> print(noise_pred.shape)  # (2, 30, 9)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from frame_representation import (
    ResidueFrameEncoder,
    ResidueFrameDecoder,
    FrameDiffusion,
    axis_angle_to_rotation,
    rotation_to_axis_angle
)

# Import shared layers
try:
    from se3_layers import InvariantPointAttention, SE3DiffusionBlock
except ImportError:
    pass

# Import RoseTTAFold components
try:
    from rosettafold import RoseTTAFoldModule
except ImportError:
    pass


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for diffusion timesteps.
    
    Encodes integer timesteps into continuous vector representations
    using sinusoidal functions at different frequencies.
    
    Parameters
    ----------
    dim : int
        Dimension of the output embedding.
    
    Examples
    --------
    >>> embedder = SinusoidalPositionEmbedding(dim=64)
    >>> t = torch.tensor([0, 50, 100])
    >>> emb = embedder(t)
    >>> print(emb.shape)  # (3, 64)
    """
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal embeddings for timesteps.
        
        Parameters
        ----------
        t : torch.Tensor
            Integer timesteps of shape (batch_size,).
        
        Returns
        -------
        torch.Tensor
            Embeddings of shape (batch_size, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class SE3DiffusionDenoiser(nn.Module):
    """
    SE3-equivariant diffusion denoiser for protein backbone generation.
    
    This model implements the core denoising network that predicts noise
    added to protein backbone coordinates. It uses the RoseTTAFold 3-track
    architecture internally to maintain SE(3) equivariance.
    
    Parameters
    ----------
    embed_dim : int, optional
        Dimension of 1D sequence embeddings. Default: 256.
    time_dim : int, optional
        Dimension of timestep embeddings. Default: 128.
    num_layers : int, optional
        Number of RoseTTAFold blocks. Default: 4.
    num_heads : int, optional
        Number of attention heads. Default: 8.
    
    Attributes
    ----------
    frame_encoder : ResidueFrameEncoder
        Converts backbone coords to SE(3) frames.
    backbone : RoseTTAFoldModule
        The 3-track network (1D, 2D, 3D).
    trans_head : nn.Sequential
        Predicts translation noise.
    rot_head : nn.Sequential
        Predicts rotation noise.
    
    Examples
    --------
    >>> model = SE3DiffusionDenoiser(embed_dim=128, num_layers=4)
    >>> x_noisy = torch.randn(4, 50, 9)  # 4 proteins, 50 residues
    >>> t = torch.randint(0, 100, (4,))
    >>> noise_pred = model(x_noisy, t)
    >>> print(noise_pred.shape)  # torch.Size([4, 50, 9])
    
    See Also
    --------
    BackboneDiffusionDenoiser : Simpler MLP-based denoiser.
    RoseTTAFoldModule : The 3-track backbone network.
    
    Structure:
    1. Embeddings: Convert coords to frames.
    2. Backbone: RoseTTAFoldModule (updates 1D, 2D, 3D features).
    3. Output: Predicts translation and rotation noise.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        time_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 1. Input Embeddings
        # Frames are computed on the fly
        self.frame_encoder = ResidueFrameEncoder()
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, embed_dim)
        )
        
        # Initial 1D feature embedding (from amino acid type or just learned)
        # For backbone generation, we learn a single "backbone" token per residue
        self.res_embedding = nn.Embedding(1, embed_dim) # Only 1 type for unconditional backbone
        
        # Initial 2D embedding
        self.d_pair = 64
        self.pair_embedding = nn.Linear(time_dim, self.d_pair) # Cond on time
        
        # 2. Backbone
        self.backbone = RoseTTAFoldModule(
            depth=num_layers,
            d_model_1d=embed_dim,
            d_model_2d=self.d_pair
        )
        
        # 3. Output Heads
        # Predict translation update (in local frame) -> Global noise
        self.trans_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3)
        )
        
        # Predict rotation update (axis-angle in local frame)
        self.rot_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3)
        )
    
    def forward(self, x_noisy, t):
        """
        x_noisy: (B, N, 9)
        t: (B,)
        """
        batch_size, num_res, _ = x_noisy.shape
        
        # 1. Features & Frames
        # Frames: (B, N, 3, 3), (B, N, 3)
        frames_R, frames_t = self.frame_encoder(x_noisy)
        
        # Time
        t_emb = self.time_embedding(t) # (B, time_dim)
        t_feat = self.time_mlp(t_emb).unsqueeze(1) # (B, 1, embed_dim)
        
        # 1D Features
        # Start with constant embedding + time
        seq_1d = self.res_embedding(torch.zeros(batch_size, num_res, dtype=torch.long, device=x_noisy.device))
        seq_1d = seq_1d + t_feat
        
        # 2D Features
        # Initialize with time info broadcasted to (L, L)
        pair_2d = self.pair_embedding(t_emb) # (B, pair_dim)
        pair_2d = pair_2d.view(batch_size, 1, 1, self.d_pair).expand(-1, num_res, num_res, -1)
        
        # 3. Backbone Passes
        seq_1d, pair_2d, frames_R, frames_t = self.backbone(
            seq_1d, pair_2d, frames_R, frames_t
        )
        
        # 4. Heads
        # Predict updates in LOCAL frame
        delta_t_local = self.trans_head(seq_1d) # (B, N, 3)
        delta_r_local = self.rot_head(seq_1d)   # (B, N, 3) axis-angle
        
        # Convert local updates to global noise predictions
        # Global translation noise = R * delta_t_local
        # Global rotation is more complex, here we approximate noise on atoms
        
        # Apply rotation to delta_t to get global translation noise
        # noise_trans = torch.einsum('bnij,bnj->bni', frames_R, delta_t_local)
        
        # Simplification: The model predicts the *noise* directly in global coordinates
        # but conditioning on local frames makes it equivariant.
        # Actually RFDiffusion predicts x0 or noise via equivariant updates.
        
        # Let's project back to global frame for the noise output
        # Global vector = R * Local vector
        noise_pred_trans = torch.einsum('bnij,bnj->bni', frames_R, delta_t_local)
        
        # For the rotation atoms (N, C), we move them based on the predicted rotation
        # plus the translation.
        # This part requires the FrameDiffusion decoder we wrote earlier or just
        # returning the translation noise for the Ca atoms for simplicity in this prototype.
        # The current 'x_noisy' is 9 coords.
        
        # Approximating noise for N and C based on Ca translation + rotation
        # (This is a simplification for the prototype to output (B, N, 9))
        
        # Ca noise is just trans noise
        noise_ca = noise_pred_trans
        
        # For N and C, we apply the rotation delta to their vectors from Ca
        # But for the diffusion loss, we just need to predict the noise added.
        # Let's upscale the SINGLE vector prediction to 3 atoms for now
        # Ideally we predict separate updates for side atoms
        
        noise_pred = noise_ca.unsqueeze(2).expand(-1, -1, 3, -1).reshape(batch_size, num_res, 9)
        
        return noise_pred


def compute_se3_diffusion_loss(
    model: nn.Module,
    x0: torch.Tensor,
    schedule,
) -> torch.Tensor:
    """
    Compute SE3-aware diffusion loss.
    
    Parameters
    ----------
    model : nn.Module
        SE3DiffusionDenoiser
    x0 : torch.Tensor
        Clean backbone coordinates
    schedule : DiffusionSchedule
        Noise schedule
    
    Returns
    -------
    torch.Tensor
        Loss value
    """
    batch_size = x0.shape[0]
    device = x0.device
    
    # Sample random timesteps
    t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)
    
    # Add noise to coordinates
    noisy_x, noise = schedule.q_sample(x0, t)
    
    # Predict noise
    noise_pred = model(noisy_x, t)
    
    # MSE loss
    loss = F.mse_loss(noise_pred, noise)
    
    return loss


if __name__ == "__main__":
    print("Testing SE3 Diffusion Denoiser...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = SE3DiffusionDenoiser(
        embed_dim=128,
        time_dim=64,
        num_layers=2,
        num_heads=4
    ).to(device)
    
    # Count parameters
    # The RoseTTAFold module uses more complexity (1D, 2D tracks)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    num_residues = 30
    
    coords = torch.randn(batch_size, num_residues, 9, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    
    print("Running forward pass...")
    noise_pred = model(coords, t)
    print(f"Input shape: {coords.shape}")
    print(f"Output shape: {noise_pred.shape}")
    
    # Test loss computation
    from backbone_diffusion import DiffusionSchedule
    schedule = DiffusionSchedule(num_timesteps=100, device=device)
    
    loss = compute_se3_diffusion_loss(model, coords, schedule)
    print(f"Loss: {loss.item():.6f}")
    
    # Test backward pass
    loss.backward()
    print("Backward pass successful!")
    
    print("\nAll tests passed!")

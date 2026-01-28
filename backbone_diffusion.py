"""
Backbone Diffusion Model for Protein Structure Generation

This module implements a simple denoising diffusion probabilistic model (DDPM)
for protein backbone coordinates. It serves as a proof-of-concept prototype
for integrating RFDiffusion-style models into DeepChem.

Author: Chidwipak
Date: January 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for diffusion timesteps.
    This is the standard embedding used in DDPM and similar models.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of timesteps of shape (batch_size,)
        
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Simple residual block with time conditioning."""
    
    def __init__(self, in_dim: int, hidden_dim: int, time_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.time_mlp = nn.Linear(time_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(in_dim)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch, num_residues, feature_dim)
            t_emb: Time embeddings of shape (batch, time_dim)
        
        Returns:
            Output of shape (batch, num_residues, feature_dim)
        """
        h = self.fc1(x)
        h = self.norm1(h)
        # Add time embedding (broadcast across residues)
        h = h + self.time_mlp(t_emb).unsqueeze(1)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return x + h


class BackboneDiffusionDenoiser(nn.Module):
    """
    Neural network that predicts noise added to protein backbone coordinates.
    
    Architecture:
    - Input: Noisy backbone coordinates (N, Ca, C per residue = 9 values)
    - Process: MLP with time conditioning and residual connections
    - Output: Predicted noise (same shape as input)
    
    This is a simplified version. For production, use SE3-equivariant networks.
    """
    
    def __init__(
        self,
        coord_dim: int = 9,  # 3 atoms (N, Ca, C) x 3 coordinates (x, y, z)
        hidden_dim: int = 256,
        time_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(coord_dim, hidden_dim)
        
        # Residual blocks with time conditioning
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, time_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, coord_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise from noisy coordinates at timestep t.
        
        Args:
            x: Noisy backbone coordinates, shape (batch, num_residues, 9)
            t: Timesteps, shape (batch,)
        
        Returns:
            Predicted noise, shape (batch, num_residues, 9)
        """
        # Get time embeddings
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        # Project input
        h = self.input_proj(x)
        
        # Apply residual blocks with time conditioning
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Project to output
        noise_pred = self.output_proj(h)
        
        return noise_pred


class DiffusionSchedule:
    """
    Variance schedule for the diffusion process.
    Implements linear beta schedule as in original DDPM.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = 'cpu'
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute useful quantities
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )
    
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to x0.
        
        Args:
            x0: Original coordinates, shape (batch, num_residues, 9)
            t: Timesteps, shape (batch,)
            noise: Optional noise tensor
        
        Returns:
            Tuple of (noisy_x, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        
        noisy_x = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return noisy_x, noise
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise x_t by one step.
        
        Args:
            model: Denoiser network
            x_t: Noisy coordinates at timestep t
            t: Current timestep
        
        Returns:
            Denoised coordinates at timestep t-1
        """
        # Predict noise
        noise_pred = model(x_t, t)
        
        # Compute mean for t-1
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t][:, None, None]
        beta_t = self.betas[t][:, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        
        mean = sqrt_recip_alpha_t * (x_t - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)
        
        # Add noise if t > 0
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])[:, None, None]
            x_prev = mean + variance * noise
        else:
            x_prev = mean
        
        return x_prev
    
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate samples by full reverse diffusion.
        
        Args:
            model: Trained denoiser
            shape: (batch_size, num_residues, 9)
            device: Device to run on
        
        Returns:
            Generated coordinates
        """
        model.eval()
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        
        return x


def compute_diffusion_loss(
    model: nn.Module,
    x0: torch.Tensor,
    schedule: DiffusionSchedule
) -> torch.Tensor:
    """
    Compute the simplified DDPM training loss.
    
    Args:
        model: Denoiser network
        x0: Clean backbone coordinates
        schedule: Diffusion schedule
    
    Returns:
        MSE loss between predicted and actual noise
    """
    batch_size = x0.shape[0]
    device = x0.device
    
    # Sample random timesteps
    t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)
    
    # Forward diffusion
    noisy_x, noise = schedule.q_sample(x0, t)
    
    # Predict noise
    noise_pred = model(noisy_x, t)
    
    # MSE loss
    loss = F.mse_loss(noise_pred, noise)
    
    return loss


if __name__ == "__main__":
    # Quick test
    print("Testing BackboneDiffusionDenoiser...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = BackboneDiffusionDenoiser().to(device)
    schedule = DiffusionSchedule(num_timesteps=100, device=device)  # Small for testing
    
    # Test forward pass
    batch_size = 4
    num_residues = 50
    x0 = torch.randn(batch_size, num_residues, 9, device=device)
    
    # Compute loss
    loss = compute_diffusion_loss(model, x0, schedule)
    print(f"Test loss: {loss.item():.4f}")
    
    # Test sampling
    print("Testing sampling...")
    samples = schedule.sample(model, (1, 10, 9), device=device)
    print(f"Sample shape: {samples.shape}")
    
    print("All tests passed!")

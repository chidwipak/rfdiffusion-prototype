"""
SE3-Equivariant Diffusion Denoiser for Protein Backbones

This module implements a more sophisticated denoiser that operates on
residue frames and uses equivariant-inspired architecture patterns.

Note: Full SE3-equivariance would require DGL graphs and SE3PairwiseConv.
This version approximates it with frame-aware operations.

Author: Chidwipak
Date: January 2026
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


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class InvariantPointAttention(nn.Module):
    """
    Simplified Invariant Point Attention (IPA) inspired by AlphaFold2.
    
    Operates on residue embeddings and frames to produce rotationally
    invariant attention patterns.
    
    This is a simplified version - full IPA uses query/key/value points
    transformed through residue frames.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_query_points: int = 4,
        num_value_points: int = 8
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Point query/key/value projections (for geometric attention)
        self.q_point_proj = nn.Linear(embed_dim, num_heads * num_query_points * 3)
        self.k_point_proj = nn.Linear(embed_dim, num_heads * num_query_points * 3)
        self.v_point_proj = nn.Linear(embed_dim, num_heads * num_value_points * 3)
        
        # Output projection
        self.out_proj = nn.Linear(
            embed_dim + num_heads * num_value_points * 3 + num_heads,
            embed_dim
        )
        
        # Learnable weights for attention combination
        self.head_weights = nn.Parameter(torch.ones(num_heads))
    
    def forward(
        self,
        x: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of IPA.
        
        Parameters
        ----------
        x : torch.Tensor
            Residue embeddings, shape (batch, num_residues, embed_dim)
        rotations : torch.Tensor
            Residue frame rotations, shape (batch, num_residues, 3, 3)
        translations : torch.Tensor
            Residue frame translations, shape (batch, num_residues, 3)
        mask : torch.Tensor, optional
            Attention mask
        
        Returns
        -------
        torch.Tensor
            Updated embeddings, shape (batch, num_residues, embed_dim)
        """
        batch_size, num_residues, _ = x.shape
        
        # Standard attention
        q = self.q_proj(x).view(batch_size, num_residues, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_residues, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_residues, self.num_heads, self.head_dim)
        
        # Standard attention scores
        attn_scores = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(self.head_dim)
        
        # Point-based attention (geometric)
        q_points = self.q_point_proj(x).view(
            batch_size, num_residues, self.num_heads, self.num_query_points, 3
        )
        k_points = self.k_point_proj(x).view(
            batch_size, num_residues, self.num_heads, self.num_query_points, 3
        )
        v_points = self.v_point_proj(x).view(
            batch_size, num_residues, self.num_heads, self.num_value_points, 3
        )
        
        # Transform points to global frame
        # q_global = R @ q_local + t
        q_global = torch.einsum(
            'bijk,bihlk->bihlj',
            rotations,
            q_points
        ) + translations[:, :, None, None, :]
        
        k_global = torch.einsum(
            'bijk,bihlk->bihlj',
            rotations,
            k_points
        ) + translations[:, :, None, None, :]
        
        v_global = torch.einsum(
            'bijk,bihlk->bihlj',
            rotations,
            v_points
        ) + translations[:, :, None, None, :]
        
        # Simplified distance-based attention using Ca positions (translations)
        # translations: (batch, num_res, 3)
        trans_i = translations[:, :, None, :]  # (batch, num_res, 1, 3)
        trans_j = translations[:, None, :, :]  # (batch, 1, num_res, 3)
        dist_sq = torch.sum((trans_i - trans_j) ** 2, dim=-1)  # (batch, num_res, num_res)
        
        # Broadcast to heads dimension
        point_attn = -0.5 * dist_sq[:, None, :, :]  # (batch, 1, i, j)

        
        # Combine attention scores
        weights = F.softmax(self.head_weights, dim=0)
        combined_attn = attn_scores + point_attn * weights[None, :, None, None]
        
        if mask is not None:
            combined_attn = combined_attn.masked_fill(~mask[:, None, None, :], -1e9)
        
        attn_probs = F.softmax(combined_attn, dim=-1)
        
        # Aggregate values
        attn_output = torch.einsum('bhij,bjhd->bihd', attn_probs, v)
        attn_output = attn_output.reshape(batch_size, num_residues, self.embed_dim)
        
        # Simplified geometric aggregation using Ca translations
        # Instead of full IPA point aggregation, use Ca positions weighted by attention
        # translations: (batch, num_res, 3)
        trans_weighted = torch.einsum('bhij,bj...->bhi...', attn_probs, translations)
        # trans_weighted: (batch, heads, num_res, 3) - aggregate neighbor positions
        trans_weighted = trans_weighted.permute(0, 2, 1, 3)  # (batch, num_res, heads, 3)
        
        # Relative positions (local reference)
        trans_local = trans_weighted - translations[:, :, None, :]  # (batch, num_res, heads, 3)
        point_output_flat = trans_local.reshape(batch_size, num_residues, self.num_heads * 3)
        
        # Pad to expected size
        expected_size = self.num_heads * self.num_value_points * 3
        if point_output_flat.shape[-1] < expected_size:
            padding = torch.zeros(
                batch_size, num_residues, 
                expected_size - point_output_flat.shape[-1],
                device=x.device
            )
            point_output_flat = torch.cat([point_output_flat, padding], dim=-1)

        
        # Pairwise distance features for output
        mean_dist = torch.sqrt(dist_sq.mean(dim=-1, keepdim=True) + 1e-8)  # (batch, num_res, 1)
        pair_features = mean_dist.expand(-1, -1, self.num_heads)

        
        # Concatenate and project
        output = torch.cat([attn_output, point_output_flat, pair_features], dim=-1)
        output = self.out_proj(output)
        
        return output


class SE3DiffusionBlock(nn.Module):
    """
    Single block of SE3-aware diffusion denoiser.
    
    Components:
    1. Invariant Point Attention (geometric)
    2. Feed-forward network
    3. Time conditioning
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        time_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 1024
    ):
        super().__init__()
        
        self.ipa = InvariantPointAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of SE3 diffusion block.
        
        Parameters
        ----------
        x : torch.Tensor
            Residue embeddings, (batch, num_res, embed_dim)
        rotations : torch.Tensor
            Frame rotations, (batch, num_res, 3, 3)
        translations : torch.Tensor
            Frame translations, (batch, num_res, 3)
        time_emb : torch.Tensor
            Time embeddings, (batch, time_dim)
        
        Returns
        -------
        torch.Tensor
            Updated embeddings
        """
        # Time conditioning
        time_cond = self.time_mlp(time_emb)[:, None, :]  # (batch, 1, embed_dim)
        
        # IPA with residual
        h = self.norm1(x)
        h = self.ipa(h, rotations, translations)
        x = x + h + time_cond
        
        # Feed-forward with residual
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        
        return x


class SE3DiffusionDenoiser(nn.Module):
    """
    Full SE3-aware diffusion denoiser.
    
    Architecture
    ------------
    1. Embed noisy coordinates to frames
    2. Create residue embeddings
    3. Apply SE3DiffusionBlocks with time conditioning
    4. Predict translation noise and rotation noise
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
        
        # Frame encoder/decoder
        self.frame_encoder = ResidueFrameEncoder()
        self.frame_decoder = ResidueFrameDecoder()
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Initial embedding from coords
        self.coord_embed = nn.Linear(9, embed_dim)  # N, Ca, C flattened
        
        # Positional embedding for sequence position
        self.pos_embed = nn.Embedding(1024, embed_dim)  # Max 1024 residues
        
        # SE3 diffusion blocks
        self.blocks = nn.ModuleList([
            SE3DiffusionBlock(
                embed_dim=embed_dim,
                time_dim=time_dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.translation_head = nn.Linear(embed_dim, 3)  # Predict translation noise
        self.rotation_head = nn.Linear(embed_dim, 3)  # Predict rotation noise (axis-angle)
    
    def forward(
        self,
        noisy_coords: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise from noisy coordinates.
        
        Parameters
        ----------
        noisy_coords : torch.Tensor
            Noisy backbone coordinates, (batch, num_res, 9)
        t : torch.Tensor
            Timesteps, (batch,)
        
        Returns
        -------
        torch.Tensor
            Predicted noise, (batch, num_res, 9)
        """
        batch_size, num_residues, _ = noisy_coords.shape
        device = noisy_coords.device
        
        # Get time embeddings
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)
        
        # Convert to frames
        rotations, translations = self.frame_encoder(noisy_coords)
        
        # Create residue embeddings
        x = self.coord_embed(noisy_coords)
        
        # Add positional embeddings
        positions = torch.arange(num_residues, device=device)
        x = x + self.pos_embed(positions)[None, :, :]
        
        # Apply SE3 blocks
        for block in self.blocks:
            x = block(x, rotations, translations, time_emb)
        
        # Predict noise components
        translation_noise = self.translation_head(x)  # (batch, num_res, 3)
        rotation_noise = self.rotation_head(x)  # (batch, num_res, 3)
        
        # Convert back to coordinate noise
        # This is approximate - proper version would use frame operations
        noise_pred = torch.cat([
            rotation_noise,  # N atom noise (approximate)
            translation_noise,  # Ca atom noise (translation)
            rotation_noise  # C atom noise (approximate)
        ], dim=-1)
        
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

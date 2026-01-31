"""
SE3-Equivariant Layers for generic usage.
Contains InvariantPointAttention and SE3DiffusionBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class InvariantPointAttention(nn.Module):
    """
    Simplified Invariant Point Attention (IPA) + Point Updates.
    Inspired by AlphaFold2 / RFDiffusion structure module.
    
    This module updates standard 1D features (s) using geometry (frames).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_value_points: int = 4,
        num_query_points: int = 4,
        scalar_key_dim: int = 16,
        scalar_value_dim: int = 16,
        point_key_dim: int = 4,
        point_value_dim: int = 4
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # Dimensions
        self.head_dim = embed_dim // num_heads
        self.scalar_key_dim = scalar_key_dim
        self.scalar_value_dim = scalar_value_dim
        self.point_key_dim = point_key_dim
        self.point_value_dim = point_value_dim
        
        # Projections for scalar attention (standard Transformer stuff)
        self.project_q_scalar = nn.Linear(embed_dim, num_heads * scalar_key_dim, bias=False)
        self.project_k_scalar = nn.Linear(embed_dim, num_heads * scalar_key_dim, bias=False)
        self.project_v_scalar = nn.Linear(embed_dim, num_heads * scalar_value_dim, bias=False)
        
        # Projections for point attention (Geometric part)
        # We generate points in local frame of each residue
        self.project_q_point = nn.Linear(embed_dim, num_heads * num_query_points * 3, bias=False)
        self.project_k_point = nn.Linear(embed_dim, num_heads * num_query_points * 3, bias=False) # K and Q should match points
        self.project_v_point = nn.Linear(embed_dim, num_heads * num_value_points * 3, bias=False)
        
        # Output projection
        # We concatenate scalar values, point values (norm), and point values (in local frame)
        # For simplicity in this prototype, we just use the scalar output + norm of point outputs
        out_dim = num_heads * scalar_value_dim + num_heads * num_value_points # + simplified point features
        self.out_proj = nn.Linear(out_dim, embed_dim)
        
        self.softplus = nn.Softplus()
        
    def forward(
        self,
        x: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (batch, num_res, embed_dim)
        rotations: (batch, num_res, 3, 3) Global frames
        translations: (batch, num_res, 3) Global positions (Ca)
        """
        b, n, c = x.shape
        h = self.num_heads
        
        # 1. Scalar Attention terms
        q_scalar = self.project_q_scalar(x).view(b, n, h, -1) # (b, n, h, c_s)
        k_scalar = self.project_k_scalar(x).view(b, n, h, -1)
        v_scalar = self.project_v_scalar(x).view(b, n, h, -1)
        
        # 2. Point Attention terms
        # Generate points in LOCAL frame
        q_pts_local = self.project_q_point(x).view(b, n, h, -1, 3) # (b, n, h, p_q, 3)
        k_pts_local = self.project_k_point(x).view(b, n, h, -1, 3)
        v_pts_local = self.project_v_point(x).view(b, n, h, -1, 3)
        
        # Transform points to GLOBAL frame
        # x_global = R * x_local + t
        # R: (b, n, 3, 3) -> broadcast to (b, n, h, p, 3, 3)
        # t: (b, n, 3) -> broadcast to (b, n, h, p, 3)
        
        # We need to apply rigid transform for each head/point.
        # Efficient implementation: (b, n, 3, 3) @ (b, n, ..., 3, 1) + t
        
        # Reshape for broadcast
        R_expand = rotations.view(b, n, 1, 1, 3, 3)
        t_expand = translations.view(b, n, 1, 1, 3)
        
        # Check simple broadcast via verify: q_pts_local (b, n, h, p, 3)
        # We treat last dim as vector
        # (b, n, 1, 1, 3, 3) x (b, n, h, p, 3, 1) -> (b, n, h, p, 3)
        
        q_pts_global = torch.einsum('bnij,bnhpj->bnhpi', rotations, q_pts_local) + t_expand
        k_pts_global = torch.einsum('bnij,bnhpj->bnhpi', rotations, k_pts_local) + t_expand
        v_pts_global = torch.einsum('bnij,bnhpj->bnhpi', rotations, v_pts_local) + t_expand
        
        # 3. Attention Scores
        # Scalar score: q^T k
        attn_scalar = torch.einsum('bhid,bhjd->bhij', q_scalar.permute(0, 2, 1, 3), k_scalar.permute(0, 2, 1, 3))
        # (b, h, n, n)
        
        # Distance score: -w/2 * ||q - k||^2
        # Start by computing squared distances between all pairs of residues (n, n) for all heads/points
        # (b, n_q, h, p, 3) vs (b, n_k, h, p, 3)
        
        # Naively: (n, 1, ...) - (1, n, ...)
        # Memory expensive for large proteins, but okay for prototype
        
        # Shapes: q (b, n, h, p, 3), k (b, n, h, p, 3)
        # We want dest (n), src (m)
        diff = q_pts_global.unsqueeze(2) - k_pts_global.unsqueeze(1) # (b, n, n, h, p, 3)
        dist_sq = (diff ** 2).sum(dim=-1).sum(dim=-1) # (b, n, n, h) sum over coord and points
        
        # Gamma weights for distance term (usually learned, here fixed or simplified)
        gamma = np.sqrt(2.0 / (9 * self.point_key_dim)) # Heuristic scaling
        attn_point = -0.5 * dist_sq * gamma
        
        # Combine scores
        attn_logits = attn_scalar + attn_point.permute(0, 3, 1, 2) # (b, h, n, n)
        
        # Bias from scalar term scaling
        attn_logits = attn_logits * (1.0 / np.sqrt(self.scalar_key_dim))
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # 4. Aggregate Values
        # Scalar values
        out_scalar = torch.matmul(attn_weights, v_scalar.permute(0, 2, 1, 3)) # (b, h, n, v_dim)
        out_scalar = out_scalar.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        
        # Point values: We attend to global points, then project back to LOCAL frame of receiving residue
        # out_pts_global = weights * v_pts_global
        
        # Flatten points for attention matmul: (b, h, n, p*3)
        v_pts_flat = v_pts_global.view(b, n, h, -1).permute(0, 2, 1, 3)
        out_pts_global = torch.matmul(attn_weights, v_pts_flat) # (b, h, n, p*3)
        out_pts_global = out_pts_global.permute(0, 2, 1, 3).view(b, n, h, -1, 3)
        
        # Transform back to local frame of residue i
        # x_local = R^T * (x_global - t)
        t_expand_i = translations.view(b, n, 1, 1, 3)
        R_expand_i = rotations.view(b, n, 1, 1, 3, 3)
        
        out_pts_centered = out_pts_global - t_expand_i
        # R^T @ v
        out_pts_local = torch.einsum('bnji,bnhpj->bnhpi', rotations, out_pts_centered)
        
        # Features from points: norms and flattened coords
        out_pts_norm = torch.norm(out_pts_local, dim=-1) # (b, n, h, p)
        out_pts_features = out_pts_norm.view(b, n, -1)
        
        # Concatenate everything
        out = torch.cat([out_scalar, out_pts_features], dim=-1)
        
        return self.out_proj(out)


class SE3DiffusionBlock(nn.Module):
    """
    Single block of SE3-equivariant transformer.
    """
    def __init__(self, embed_dim: int, num_heads: int, time_dim: int = 128):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ipa = InvariantPointAttention(embed_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, embed_dim)
        )
        
    def forward(self, x, rotations, translations, time_emb):
        # Time conditioning
        t = self.time_mlp(time_emb).unsqueeze(1)
        x = x + t
        
        # Attention
        res = x
        x = self.norm1(x)
        x = self.ipa(x, rotations, translations)
        x = res + x
        
        # Feedforward
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        
        return x

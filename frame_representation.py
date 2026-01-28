"""
Frame Representation Module for Protein Structures

Implements residue-level local frames from N-Ca-C backbone atoms,
following the approach from AlphaFold2 and RFDiffusion.

Author: Chidwipak
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def gram_schmidt(v1: torch.Tensor, v2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gram-Schmidt orthonormalization to create orthonormal basis.
    
    Parameters
    ----------
    v1 : torch.Tensor
        First vector, shape (..., 3)
    v2 : torch.Tensor
        Second vector (not necessarily orthogonal to v1), shape (..., 3)
    
    Returns
    -------
    Tuple
        (e1, e2, e3) orthonormal basis vectors
    """
    # Normalize v1 to get e1
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    
    # Make v2 orthogonal to e1
    u2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + 1e-8)
    
    # e3 is cross product
    e3 = torch.cross(e1, e2, dim=-1)
    
    return e1, e2, e3


def coords_to_frames(
    n_coords: torch.Tensor,
    ca_coords: torch.Tensor,
    c_coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert backbone atom coordinates to local frames.
    
    Following AlphaFold2/RFDiffusion convention:
    - Origin at Ca
    - x-axis: Ca -> C direction
    - y-axis: in N-Ca-C plane, perpendicular to x
    - z-axis: perpendicular to plane (right-hand rule)
    
    Parameters
    ----------
    n_coords : torch.Tensor
        N atom coordinates, shape (..., 3)
    ca_coords : torch.Tensor
        Ca atom coordinates, shape (..., 3)
    c_coords : torch.Tensor
        C atom coordinates, shape (..., 3)
    
    Returns
    -------
    Tuple
        (rotation_matrices, translations)
        rotation: (..., 3, 3), translation: (..., 3)
    """
    # Translation is Ca position
    translation = ca_coords
    
    # Build local frame
    v1 = c_coords - ca_coords  # Ca -> C
    v2 = n_coords - ca_coords  # Ca -> N
    
    e1, e2, e3 = gram_schmidt(v1, v2)
    
    # Rotation matrix: columns are basis vectors
    rotation = torch.stack([e1, e2, e3], dim=-1)
    
    return rotation, translation


def frames_to_coords(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    ideal_n: torch.Tensor = None,
    ideal_ca: torch.Tensor = None,
    ideal_c: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert local frames back to backbone coordinates.
    
    Uses ideal bond geometry if not provided:
    - N-Ca bond: 1.46 Å
    - Ca-C bond: 1.52 Å
    - N-Ca-C angle: ~111°
    
    Parameters
    ----------
    rotation : torch.Tensor
        Rotation matrices, shape (..., 3, 3)
    translation : torch.Tensor
        Translation vectors (Ca positions), shape (..., 3)
    
    Returns
    -------
    Tuple
        (n_coords, ca_coords, c_coords)
    """
    # Ideal local coordinates (in local frame)
    if ideal_n is None:
        # N is ~1.46 Å from Ca, in -y direction (after Gram-Schmidt)
        ideal_n = torch.tensor([-0.5, -1.2, 0.0], device=rotation.device)
    if ideal_ca is None:
        ideal_ca = torch.tensor([0.0, 0.0, 0.0], device=rotation.device)
    if ideal_c is None:
        # C is ~1.52 Å from Ca, along x-axis
        ideal_c = torch.tensor([1.52, 0.0, 0.0], device=rotation.device)
    
    # Transform from local to global: R @ local + t
    n_coords = torch.einsum('...ij,...j->...i', rotation, ideal_n) + translation
    ca_coords = translation  # Ca is at origin of local frame
    c_coords = torch.einsum('...ij,...j->...i', rotation, ideal_c) + translation
    
    return n_coords, ca_coords, c_coords


class ResidueFrameEncoder(nn.Module):
    """
    Encode backbone coordinates as frames (rotation + translation).
    
    This representation is SE(3) equivariant:
    - Applying a rotation R to all coordinates rotates all frames by R
    - Applying a translation t to all coordinates shifts all frames by t
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert flattened backbone coords to frames.
        
        Parameters
        ----------
        coords : torch.Tensor
            Backbone coordinates, shape (batch, num_residues, 9)
            where 9 = 3 atoms * 3 coordinates
        
        Returns
        -------
        Tuple
            (rotations, translations)
            rotations: (batch, num_residues, 3, 3)
            translations: (batch, num_residues, 3)
        """
        batch_size, num_residues, _ = coords.shape
        
        # Reshape: (batch, num_res, 3, 3) for N, Ca, C
        coords_reshaped = coords.view(batch_size, num_residues, 3, 3)
        
        n_coords = coords_reshaped[:, :, 0, :]  # N atoms
        ca_coords = coords_reshaped[:, :, 1, :]  # Ca atoms
        c_coords = coords_reshaped[:, :, 2, :]  # C atoms
        
        rotations, translations = coords_to_frames(n_coords, ca_coords, c_coords)
        
        return rotations, translations


class ResidueFrameDecoder(nn.Module):
    """
    Decode frames back to backbone coordinates.
    """
    
    def __init__(self):
        super().__init__()
        # Learnable ideal coordinates
        self.ideal_n = nn.Parameter(torch.tensor([-0.5, -1.2, 0.0]))
        self.ideal_c = nn.Parameter(torch.tensor([1.52, 0.0, 0.0]))
    
    def forward(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert frames to flattened backbone coords.
        
        Parameters
        ----------
        rotations : torch.Tensor
            Frame rotations, shape (batch, num_residues, 3, 3)
        translations : torch.Tensor
            Frame translations, shape (batch, num_residues, 3)
        
        Returns
        -------
        torch.Tensor
            Backbone coordinates, shape (batch, num_residues, 9)
        """
        n_coords, ca_coords, c_coords = frames_to_coords(
            rotations, translations,
            ideal_n=self.ideal_n,
            ideal_c=self.ideal_c
        )
        
        # Stack and flatten
        coords = torch.stack([n_coords, ca_coords, c_coords], dim=2)
        return coords.view(rotations.shape[0], rotations.shape[1], 9)


class FrameDiffusion(nn.Module):
    """
    Diffusion process on rigid body frames.
    
    Instead of adding noise to coordinates directly, we add noise to:
    - Translation: Gaussian noise
    - Rotation: Noise on tangent space (so(3))
    
    This respects SE(3) structure better than coordinate-space diffusion.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))
    
    def add_translation_noise(
        self,
        translations: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to translations.
        
        Parameters
        ----------
        translations : torch.Tensor
            Original translations, shape (batch, num_residues, 3)
        t : torch.Tensor
            Timesteps, shape (batch,)
        
        Returns
        -------
        Tuple
            (noisy_translations, noise)
        """
        noise = torch.randn_like(translations)
        
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        
        noisy = sqrt_alpha * translations + sqrt_one_minus_alpha * noise
        
        return noisy, noise
    
    def add_rotation_noise(
        self,
        rotations: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to rotations using tangent space noise.
        
        We sample noise in so(3) (tangent space) and apply it via
        exponential map to get noisy rotation.
        
        Parameters
        ----------
        rotations : torch.Tensor
            Original rotations, shape (batch, num_residues, 3, 3)
        t : torch.Tensor
            Timesteps, shape (batch,)
        
        Returns
        -------
        Tuple
            (noisy_rotations, rotation_noise)
            rotation_noise is in tangent space (axis-angle)
        """
        batch_size, num_residues = rotations.shape[:2]
        device = rotations.device
        
        # Sample axis-angle noise (3D vector in so(3))
        noise_scale = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        axis_angle_noise = torch.randn(batch_size, num_residues, 3, device=device)
        axis_angle_noise = axis_angle_noise * noise_scale
        
        # Convert to rotation matrix via exponential map
        noise_rotation = axis_angle_to_rotation(axis_angle_noise)
        
        # Apply noise: R_noisy = R_noise @ R
        noisy_rotations = torch.einsum('...ij,...jk->...ik', noise_rotation, rotations)
        
        return noisy_rotations, axis_angle_noise


def axis_angle_to_rotation(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix.
    Uses Rodrigues' formula.
    
    Parameters
    ----------
    axis_angle : torch.Tensor
        Axis-angle vectors, shape (..., 3)
    
    Returns
    -------
    torch.Tensor
        Rotation matrices, shape (..., 3, 3)
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    
    # Rodrigues' formula
    cos_angle = torch.cos(angle)[..., None]
    sin_angle = torch.sin(angle)[..., None]
    
    # Skew-symmetric matrix
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    
    K = torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1)
    ], dim=-2)
    
    # R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=axis_angle.device).expand_as(K)
    R = I + sin_angle * K + (1 - cos_angle) * torch.einsum('...ij,...jk->...ik', K, K)
    
    return R


def rotation_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation.
    
    Parameters
    ----------
    rotation : torch.Tensor
        Rotation matrices, shape (..., 3, 3)
    
    Returns
    -------
    torch.Tensor
        Axis-angle vectors, shape (..., 3)
    """
    # Angle from trace
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    
    # Axis from skew-symmetric part
    axis = torch.stack([
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1]
    ], dim=-1)
    
    axis = axis / (2 * torch.sin(angle)[..., None] + 1e-8)
    
    return axis * angle[..., None]


if __name__ == "__main__":
    print("Testing frame representation module...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create test data
    batch_size = 4
    num_residues = 20
    
    # Random backbone coordinates
    coords = torch.randn(batch_size, num_residues, 9, device=device)
    
    # Test frame encoder
    encoder = ResidueFrameEncoder().to(device)
    rotations, translations = encoder(coords)
    print(f"Rotations shape: {rotations.shape}")  # (4, 20, 3, 3)
    print(f"Translations shape: {translations.shape}")  # (4, 20, 3)
    
    # Verify rotations are orthonormal
    RtR = torch.einsum('...ij,...ik->...jk', rotations, rotations)
    I = torch.eye(3, device=device)
    ortho_error = torch.mean(torch.abs(RtR - I))
    print(f"Orthonormality error: {ortho_error:.6f}")
    
    # Test frame decoder
    decoder = ResidueFrameDecoder().to(device)
    reconstructed = decoder(rotations, translations)
    print(f"Reconstructed coords shape: {reconstructed.shape}")
    
    # Test frame diffusion
    diffusion = FrameDiffusion(num_timesteps=100).to(device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    
    noisy_trans, trans_noise = diffusion.add_translation_noise(translations, t)
    noisy_rot, rot_noise = diffusion.add_rotation_noise(rotations, t)
    
    print(f"Noisy translations shape: {noisy_trans.shape}")
    print(f"Noisy rotations shape: {noisy_rot.shape}")
    print(f"Rotation noise (axis-angle) shape: {rot_noise.shape}")
    
    # Test axis-angle conversion
    axis_angle = torch.randn(batch_size, num_residues, 3, device=device) * 0.1
    R = axis_angle_to_rotation(axis_angle)
    axis_angle_back = rotation_to_axis_angle(R)
    conversion_error = torch.mean(torch.abs(axis_angle - axis_angle_back))
    print(f"Axis-angle roundtrip error: {conversion_error:.6f}")
    
    print("\nAll tests passed!")

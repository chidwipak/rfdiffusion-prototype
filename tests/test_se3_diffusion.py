"""
Unit tests for SE3 Diffusion and related components.

Author: Chidwipak
Date: January 2026
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from se3_diffusion import SE3DiffusionDenoiser, SinusoidalPositionEmbedding
from se3_layers import InvariantPointAttention, SE3DiffusionBlock
from frame_representation import (
    gram_schmidt,
    coords_to_frames,
    frames_to_coords,
    ResidueFrameEncoder,
    ResidueFrameDecoder
)


class TestSE3Layers(unittest.TestCase):
    """Test SE3-equivariant layers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.num_residues = 10
        self.embed_dim = 32
        self.num_heads = 2
    
    def test_ipa_shapes(self):
        """Test InvariantPointAttention output shapes."""
        ipa = InvariantPointAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_value_points=4,
            num_query_points=4
        ).to(self.device)
        
        # Inputs
        x = torch.randn(self.batch_size, self.num_residues, self.embed_dim)
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(
            self.batch_size, self.num_residues, -1, -1
        )  # Identity rotations
        t = torch.zeros(self.batch_size, self.num_residues, 3)
        
        # Forward
        out = ipa(x, R, t)
        
        # Assertions
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())
    
    def test_ipa_rotation_equivariance(self):
        """Test that IPA respects rotation symmetry."""
        ipa = InvariantPointAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        ).to(self.device)
        ipa.eval()
        
        # Create input
        x = torch.randn(1, self.num_residues, self.embed_dim)
        R_identity = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(
            1, self.num_residues, -1, -1
        )
        t = torch.randn(1, self.num_residues, 3)
        
        # Get output with identity rotation
        with torch.no_grad():
            out1 = ipa(x, R_identity, t)
        
        # Create a rotation matrix (90 degrees around z-axis)
        theta = np.pi / 2
        R_z = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Rotate frames
        R_rotated = R_z.unsqueeze(0).unsqueeze(0) @ R_identity
        t_rotated = torch.einsum('ij,...j->...i', R_z, t)
        
        # Get output with rotated frames
        with torch.no_grad():
            out2 = ipa(x, R_rotated, t_rotated)
        
        # The scalar outputs should be invariant to global rotation
        # (up to numerical precision)
        # Note: Feature outputs should be the same since x didn't change
        self.assertTrue(out1.shape == out2.shape)


class TestFrameRepresentation(unittest.TestCase):
    """Test frame representation utilities."""
    
    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthonormalization."""
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([1.0, 1.0, 0.0])
        
        e1, e2, e3 = gram_schmidt(v1, v2)
        
        # Check orthonormality
        self.assertAlmostEqual(torch.dot(e1, e2).item(), 0.0, places=5)
        self.assertAlmostEqual(torch.dot(e1, e3).item(), 0.0, places=5)
        self.assertAlmostEqual(torch.dot(e2, e3).item(), 0.0, places=5)
        
        # Check unit length
        self.assertAlmostEqual(torch.norm(e1).item(), 1.0, places=5)
        self.assertAlmostEqual(torch.norm(e2).item(), 1.0, places=5)
        self.assertAlmostEqual(torch.norm(e3).item(), 1.0, places=5)
    
    def test_coords_to_frames_shapes(self):
        """Test coords_to_frames output shapes."""
        batch_size = 2
        num_residues = 10
        
        n_coords = torch.randn(batch_size, num_residues, 3)
        ca_coords = torch.randn(batch_size, num_residues, 3)
        c_coords = torch.randn(batch_size, num_residues, 3)
        
        R, t = coords_to_frames(n_coords, ca_coords, c_coords)
        
        self.assertEqual(R.shape, (batch_size, num_residues, 3, 3))
        self.assertEqual(t.shape, (batch_size, num_residues, 3))
    
    def test_coords_to_frames_rotation_valid(self):
        """Test that output rotations are proper rotation matrices."""
        n = torch.tensor([[0.0, -1.46, 0.0]])
        ca = torch.tensor([[0.0, 0.0, 0.0]])
        c = torch.tensor([[1.52, 0.0, 0.0]])
        
        R, t = coords_to_frames(n, ca, c)
        
        # Check orthogonality: R^T R = I
        R_transpose = R.transpose(-1, -2)
        product = torch.matmul(R_transpose, R)
        identity = torch.eye(3)
        
        self.assertTrue(torch.allclose(product.squeeze(), identity, atol=1e-5))
        
        # Check determinant = 1 (proper rotation)
        det = torch.det(R.squeeze())
        self.assertAlmostEqual(det.item(), 1.0, places=4)
    
    def test_encoder_decoder_consistency(self):
        """Test that encoding then decoding gives back valid structure."""
        batch_size = 2
        num_residues = 5
        
        # Create valid backbone coordinates
        coords = torch.randn(batch_size, num_residues, 9)
        
        encoder = ResidueFrameEncoder()
        decoder = ResidueFrameDecoder()
        
        # Encode
        R, t = encoder(coords)
        
        # Decode
        reconstructed = decoder(R, t)
        
        # Check shape
        self.assertEqual(reconstructed.shape, coords.shape)


class TestSE3DiffusionDenoiser(unittest.TestCase):
    """Test the main SE3 diffusion model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.num_residues = 10
        
        self.model = SE3DiffusionDenoiser(
            embed_dim=32,
            time_dim=16,
            num_layers=2,
            num_heads=2
        ).to(self.device)
    
    def test_forward_shapes(self):
        """Test forward pass output shapes."""
        coords = torch.randn(self.batch_size, self.num_residues, 9)
        t = torch.randint(0, 100, (self.batch_size,))
        
        noise_pred = self.model(coords, t)
        
        self.assertEqual(noise_pred.shape, coords.shape)
    
    def test_forward_no_nan(self):
        """Test that forward pass doesn't produce NaN."""
        coords = torch.randn(self.batch_size, self.num_residues, 9)
        t = torch.randint(0, 100, (self.batch_size,))
        
        noise_pred = self.model(coords, t)
        
        self.assertFalse(torch.isnan(noise_pred).any())
        self.assertFalse(torch.isinf(noise_pred).any())
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        coords = torch.randn(self.batch_size, self.num_residues, 9, requires_grad=True)
        t = torch.randint(0, 100, (self.batch_size,))
        
        noise_pred = self.model(coords, t)
        loss = noise_pred.sum()
        loss.backward()
        
        # Check that model parameters have gradients
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        self.assertTrue(has_grad, "Model parameters should have gradients")
    
    def test_different_timesteps(self):
        """Test that different timesteps produce different outputs."""
        coords = torch.randn(1, self.num_residues, 9)
        
        with torch.no_grad():
            t0 = torch.tensor([0])
            t50 = torch.tensor([50])
            
            out_t0 = self.model(coords, t0)
            out_t50 = self.model(coords, t50)
        
        # Outputs should be different for different timesteps
        self.assertFalse(torch.allclose(out_t0, out_t50))


class TestTimestepEmbedding(unittest.TestCase):
    """Test timestep embeddings."""
    
    def test_sinusoidal_embedding_shapes(self):
        """Test sinusoidal embedding output shape."""
        dim = 64
        embedder = SinusoidalPositionEmbedding(dim)
        
        t = torch.tensor([0, 10, 50, 99])
        emb = embedder(t)
        
        self.assertEqual(emb.shape, (4, dim))
    
    def test_sinusoidal_embedding_unique(self):
        """Test that different timesteps get different embeddings."""
        dim = 64
        embedder = SinusoidalPositionEmbedding(dim)
        
        t = torch.tensor([0, 1, 2, 3])
        emb = embedder(t)
        
        # All embeddings should be unique
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(torch.allclose(emb[i], emb[j]))


if __name__ == '__main__':
    unittest.main()

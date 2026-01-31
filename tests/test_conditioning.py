"""
Unit tests for conditioning modules.

Author: Chidwipak
Date: January 2026
"""

import sys
import os
import unittest
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conditioning import (
    MotifConditioner,
    LengthConditioner,
    BinderConditioner
)


class TestMotifConditioner(unittest.TestCase):
    """Test motif conditioning for scaffolding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.num_residues = 20
        self.embed_dim = 64
        
        self.conditioner = MotifConditioner(embed_dim=self.embed_dim)
    
    def test_forward_with_conditioning(self):
        """Test forward pass with motif mask."""
        x = torch.randn(self.batch_size, self.num_residues, self.embed_dim)
        motif_coords = torch.randn(self.batch_size, self.num_residues, 9)
        motif_mask = torch.zeros(self.batch_size, self.num_residues, dtype=torch.bool)
        motif_mask[:, 5:10] = True  # Residues 5-9 are motif
        
        out = self.conditioner(x, motif_coords, motif_mask)
        
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())
    
    def test_forward_without_conditioning(self):
        """Test forward pass without conditioning (passthrough)."""
        x = torch.randn(self.batch_size, self.num_residues, self.embed_dim)
        
        out = self.conditioner(x)
        
        # Should return input unchanged
        self.assertTrue(torch.equal(out, x))
    
    def test_motif_residues_receive_different_embedding(self):
        """Test that motif residues get different embeddings than scaffold."""
        x = torch.randn(self.batch_size, self.num_residues, self.embed_dim)
        motif_coords = torch.randn(self.batch_size, self.num_residues, 9)
        motif_mask = torch.zeros(self.batch_size, self.num_residues, dtype=torch.bool)
        motif_mask[:, 5:10] = True
        
        out = self.conditioner(x, motif_coords, motif_mask)
        
        # Motif and scaffold residues should be processed differently
        self.assertFalse(torch.allclose(out[:, 5:10, :], x[:, 5:10, :]))


class TestLengthConditioner(unittest.TestCase):
    """Test length conditioning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.num_residues = 20
        self.embed_dim = 64
        
        self.conditioner = LengthConditioner(embed_dim=self.embed_dim, max_length=100)
    
    def test_forward_with_length(self):
        """Test forward pass with target length."""
        x = torch.randn(self.batch_size, self.num_residues, self.embed_dim)
        target_length = torch.tensor([15, 25])
        
        out = self.conditioner(x, target_length)
        
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())
    
    def test_forward_without_length(self):
        """Test forward pass with default length (num_residues)."""
        x = torch.randn(self.batch_size, self.num_residues, self.embed_dim)
        
        out = self.conditioner(x)
        
        self.assertEqual(out.shape, x.shape)
    
    def test_different_lengths_give_different_outputs(self):
        """Test that different target lengths produce different outputs."""
        x = torch.randn(1, self.num_residues, self.embed_dim)
        
        length_20 = torch.tensor([20])
        length_50 = torch.tensor([50])
        
        out_20 = self.conditioner(x, length_20)
        out_50 = self.conditioner(x, length_50)
        
        self.assertFalse(torch.allclose(out_20, out_50))
    
    def test_length_clamping(self):
        """Test that lengths beyond max_length are clamped."""
        x = torch.randn(1, self.num_residues, self.embed_dim)
        
        # Length beyond max should be clamped
        out = self.conditioner(x, torch.tensor([200]))
        
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())


class TestBinderConditioner(unittest.TestCase):
    """Test binder conditioning for protein-protein interactions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.binder_len = 20
        self.target_len = 30
        self.embed_dim = 64
        
        self.conditioner = BinderConditioner(embed_dim=self.embed_dim)
    
    def test_forward_with_target(self):
        """Test forward pass with target protein."""
        x = torch.randn(self.batch_size, self.binder_len, self.embed_dim)
        target_coords = torch.randn(self.batch_size, self.target_len, 9)
        hotspot_mask = torch.zeros(self.batch_size, self.target_len, dtype=torch.bool)
        hotspot_mask[:, 10:15] = True
        
        out = self.conditioner(x, target_coords, hotspot_mask)
        
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())
    
    def test_forward_without_target(self):
        """Test forward pass without target (passthrough)."""
        x = torch.randn(self.batch_size, self.binder_len, self.embed_dim)
        
        out = self.conditioner(x)
        
        self.assertTrue(torch.equal(out, x))
    
    def test_cross_attention_integration(self):
        """Test that binder features are influenced by target."""
        x = torch.randn(self.batch_size, self.binder_len, self.embed_dim)
        target_coords = torch.randn(self.batch_size, self.target_len, 9)
        
        out = self.conditioner(x, target_coords)
        
        # Output should be different from input
        self.assertFalse(torch.allclose(out, x))


class TestCombinedConditioning(unittest.TestCase):
    """Test combining multiple conditioning types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.num_residues = 20
        self.embed_dim = 64
        
        self.motif_cond = MotifConditioner(embed_dim=self.embed_dim)
        self.length_cond = LengthConditioner(embed_dim=self.embed_dim)
        self.binder_cond = BinderConditioner(embed_dim=self.embed_dim)
    
    def test_sequential_conditioning(self):
        """Test applying multiple conditioners sequentially."""
        x = torch.randn(self.batch_size, self.num_residues, self.embed_dim)
        
        # Apply motif conditioning
        motif_mask = torch.zeros(self.batch_size, self.num_residues, dtype=torch.bool)
        motif_mask[:, 5:10] = True
        motif_coords = torch.randn(self.batch_size, self.num_residues, 9)
        x = self.motif_cond(x, motif_coords, motif_mask)
        
        # Apply length conditioning
        target_length = torch.tensor([self.num_residues, self.num_residues])
        x = self.length_cond(x, target_length)
        
        self.assertEqual(x.shape, (self.batch_size, self.num_residues, self.embed_dim))
        self.assertFalse(torch.isnan(x).any())
    
    def test_gradient_flow_through_conditioners(self):
        """Test that gradients flow through all conditioners."""
        x = torch.randn(self.batch_size, self.num_residues, self.embed_dim, 
                        requires_grad=True)
        
        # Apply conditioning chain
        motif_mask = torch.zeros(self.batch_size, self.num_residues, dtype=torch.bool)
        motif_mask[:, 5:10] = True
        motif_coords = torch.randn(self.batch_size, self.num_residues, 9)
        
        out = self.motif_cond(x, motif_coords, motif_mask)
        out = self.length_cond(out, torch.tensor([20, 20]))
        
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)


if __name__ == '__main__':
    unittest.main()

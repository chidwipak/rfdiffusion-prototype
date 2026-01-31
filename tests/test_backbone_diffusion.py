"""
Unit tests for backbone_diffusion.py

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

from backbone_diffusion import (
    SinusoidalPositionEmbedding,
    ResidualBlock,
    BackboneDiffusionDenoiser,
    DiffusionSchedule,
    compute_diffusion_loss
)

class TestBackboneDiffusion(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cpu'
        self.batch_size = 4
        self.num_residues = 10
        self.coord_dim = 9
    
    def test_sinusoidal_embedding(self):
        dim = 32
        embedder = SinusoidalPositionEmbedding(dim)
        t = torch.tensor([0, 10, 100], device=self.device)
        emb = embedder(t)
        
        self.assertEqual(emb.shape, (3, dim))
        # Check that output is not all zeros
        self.assertFalse(torch.allclose(emb, torch.zeros_like(emb)))
        
    def test_residual_block(self):
        in_dim = 16
        hidden_dim = 32
        time_dim = 16
        block = ResidualBlock(in_dim, hidden_dim, time_dim).to(self.device)
        
        x = torch.randn(self.batch_size, self.num_residues, in_dim, device=self.device)
        t_emb = torch.randn(self.batch_size, time_dim, device=self.device)
        
        out = block(x, t_emb)
        self.assertEqual(out.shape, x.shape)
        
    def test_denoiser_shapes(self):
        model = BackboneDiffusionDenoiser(
            coord_dim=self.coord_dim,
            hidden_dim=32,
            time_dim=16,
            num_layers=2
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.num_residues, self.coord_dim, device=self.device)
        t = torch.randint(0, 100, (self.batch_size,), device=self.device)
        
        noise_pred = model(x, t)
        self.assertEqual(noise_pred.shape, x.shape)
        
    def test_schedule_q_sample(self):
        schedule = DiffusionSchedule(num_timesteps=100, device=self.device)
        x0 = torch.randn(self.batch_size, self.num_residues, self.coord_dim, device=self.device)
        t = torch.tensor([0, 50, 99], device=self.device)
        
        # Test shape
        # Note: q_sample expects t to match batch size if not broadcast?
        # Creating a batch where t matches first dim
        if x0.shape[0] != t.shape[0]:
             t = torch.randint(0, 100, (self.batch_size,), device=self.device)
             
        noisy_x, noise = schedule.q_sample(x0, t)
        self.assertEqual(noisy_x.shape, x0.shape)
        self.assertEqual(noise.shape, x0.shape)
        
        # Test noise scaling (t=0 should be closer to x0 than t=99)
        t0 = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        t99 = torch.full((self.batch_size,), 99, dtype=torch.long, device=self.device)
        
        noisy_0, _ = schedule.q_sample(x0, t0)
        noisy_99, _ = schedule.q_sample(x0, t99)
        
        mse_0 = torch.mean((noisy_0 - x0)**2).item()
        mse_99 = torch.mean((noisy_99 - x0)**2).item()
        
        self.assertLess(mse_0, mse_99)

    def test_loss_computation(self):
        model = BackboneDiffusionDenoiser(
            coord_dim=self.coord_dim,
            hidden_dim=32, 
            time_dim=16
        ).to(self.device)
        schedule = DiffusionSchedule(num_timesteps=100, device=self.device)
        
        x0 = torch.randn(self.batch_size, self.num_residues, self.coord_dim, device=self.device)
        loss = compute_diffusion_loss(model, x0, schedule)
        
        self.assertTrue(loss.item() > 0)
        self.assertFalse(torch.isnan(loss))

if __name__ == '__main__':
    unittest.main()

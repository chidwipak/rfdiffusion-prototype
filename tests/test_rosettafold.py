"""
Unit tests for RoseTTAFold architecture

Author: Chidwipak
Date: January 2026
"""

import sys
import os
import unittest
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosettafold import RoseTTAFoldBlock, RoseTTAFoldModule

class TestRoseTTAFold(unittest.TestCase):
    
    def setUp(self):
        self.d_1d = 32
        self.d_2d = 16
        self.batch_size = 2
        self.num_res = 10
        self.device = 'cpu'
        
    def test_block_shapes(self):
        block = RoseTTAFoldBlock(
            d_model_1d=self.d_1d,
            d_model_2d=self.d_2d,
            n_heads_1d=2,
            n_heads_2d=2,
            n_heads_3d=2
        )
        
        seq = torch.randn(self.batch_size, self.num_res, self.d_1d)
        pair = torch.randn(self.batch_size, self.num_res, self.num_res, self.d_2d)
        R = torch.eye(3).view(1, 1, 3, 3).expand(self.batch_size, self.num_res, -1, -1)
        t = torch.zeros(self.batch_size, self.num_res, 3)
        
        seq_new, pair_new, R_new, t_new = block(seq, pair, R, t)
        
        self.assertEqual(seq_new.shape, seq.shape)
        self.assertEqual(pair_new.shape, pair.shape)
        self.assertEqual(R_new.shape, R.shape)
        self.assertEqual(t_new.shape, t.shape)
        
    def test_module_forward(self):
        model = RoseTTAFoldModule(
            depth=2,
            d_model_1d=self.d_1d,
            d_model_2d=self.d_2d
        )
        
        seq = torch.randn(self.batch_size, self.num_res, self.d_1d)
        pair = torch.randn(self.batch_size, self.num_res, self.num_res, self.d_2d)
        R = torch.eye(3).view(1, 1, 3, 3).expand(self.batch_size, self.num_res, -1, -1)
        t = torch.zeros(self.batch_size, self.num_res, 3)
        
        seq_out, pair_out, R_out, t_out = model(seq, pair, R, t)
        
        self.assertEqual(seq_out.shape, seq.shape)
        
if __name__ == '__main__':
    unittest.main()

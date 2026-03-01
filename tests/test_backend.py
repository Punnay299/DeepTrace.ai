import unittest
import numpy as np

# Adjust imports to be absolute or relative correctly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline.aggregator import merge_flagged_windows, calculate_verdict
from backend.pipeline.class4_heuristic import compute_class4_heuristic
from backend.pipeline.video_branch import compile_tensor
import torch

class TestVideoBranch(unittest.TestCase):
    def test_compile_tensor_valid(self):
        frames = np.random.randint(0, 255, (20, 224, 224, 3), dtype=np.uint8)
        device = torch.device('cpu')
        tensor = compile_tensor(frames, device)
        
        self.assertEqual(tensor.shape, (1, 20, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        
    def test_compile_tensor_invalid_frame_count(self):
        frames = np.zeros((19, 224, 224, 3), dtype=np.uint8)
        device = torch.device('cpu')
        with self.assertRaises(AssertionError):
            compile_tensor(frames, device)
            
    def test_compile_tensor_invalid_resolution(self):
        frames = np.zeros((20, 380, 380, 3), dtype=np.uint8)
        device = torch.device('cpu')
        with self.assertRaises(AssertionError):
            compile_tensor(frames, device)

class TestAggregator(unittest.TestCase):
    def test_empty_windows(self):
        self.assertEqual(merge_flagged_windows([]), [])
        
    def test_no_flagged_windows(self):
        scores = [{'start': 0, 'end': 4, 'score': 0.1}, {'start': 4, 'end': 8, 'score': 0.2}]
        self.assertEqual(merge_flagged_windows(scores, threshold=0.5), [])
        
    def test_single_flagged_window(self):
        scores = [{'start': 0, 'end': 4, 'score': 0.8}]
        merged = merge_flagged_windows(scores, threshold=0.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['start'], 0)
        self.assertEqual(merged[0]['end'], 4)
        self.assertEqual(merged[0]['peak_score'], 0.8)
        
    def test_merge_within_tolerance(self):
        scores = [
            {'start': 0.0, 'end': 4.0, 'score': 0.8},
            {'start': 2.0, 'end': 6.0, 'score': 0.9} # Overlaps
        ]
        merged = merge_flagged_windows(scores, threshold=0.5, tolerance_sec=2.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['start'], 0.0)
        self.assertEqual(merged[0]['end'], 6.0)
        self.assertEqual(merged[0]['peak_score'], 0.9)
        
    def test_no_merge_outside_tolerance(self):
        scores = [
            {'start': 0.0, 'end': 4.0, 'score': 0.8},
            {'start': 8.0, 'end': 12.0, 'score': 0.9} # Gap of 4.0 > 2.0
        ]
        merged = merge_flagged_windows(scores, threshold=0.5, tolerance_sec=2.0)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]['start'], 0.0)
        self.assertEqual(merged[0]['end'], 4.0)
        
    def test_calculate_verdict(self):
        self.assertEqual(calculate_verdict([], 50.0)[0], 0) # Class 0
        self.assertEqual(calculate_verdict([{'start':0}], 50.0)[0], 1) # Class 1
        self.assertEqual(calculate_verdict([], 200.0, heuristic_threshold=150.0)[0], 4) # Class 4
        self.assertEqual(calculate_verdict([{'start':0}], 200.0, heuristic_threshold=150.0)[0], 4) # Class 4 takes precedence

class TestClass4Heuristic(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(compute_class4_heuristic([]), 0.0)
        
    def test_insufficient_frames(self):
        # 1 frame
        window = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        self.assertEqual(compute_class4_heuristic([window]), 0.0)
        
    def test_normal_sequence(self):
        # 20 frames, static
        window = np.ones((20, 224, 224, 3), dtype=np.uint8) * 128
        self.assertEqual(compute_class4_heuristic([window]), 0.0)
        
        # 20 frames, alternating noise
        window2 = np.random.randint(0, 255, (20, 224, 224, 3), dtype=np.uint8)
        score = compute_class4_heuristic([window2])
        self.assertGreater(score, 0.0)

if __name__ == '__main__':
    unittest.main()

import os
import unittest
import pandas as pd
import numpy as np

class TestStage3Outputs(unittest.TestCase):
    def setUp(self):
         # Create a dummy test file simulating the Windows output
         self.test_csv = 'dataset/labels/windows_index_test.csv'
         os.makedirs('dataset/labels', exist_ok=True)
         
         data = {
             'video_id': ['v1', 'v1', 'v2', 'v3'],
             'window_id': ['v1_w0', 'v1_w1', 'v2_w0', 'v3_w0'],
             'start_sec': [0.0, 2.0, 0.0, 0.0],
             'end_sec': [4.0, 6.0, 4.0, 4.0],
             'face_detected': [1, 1, 0, 1],
             'x1': [0.1000, 0.1500, np.nan, 0.0000],
             'y1': [0.1000, 0.1500, np.nan, 0.0000],
             'x2': [0.4000, 0.4500, np.nan, 1.0000],
             'y2': [0.4000, 0.4500, np.nan, 1.0000],
             'conf_mean': [0.99, 0.98, np.nan, 0.95],
             'frame_count_checked': [20, 20, 20, 20],
             'injection_label': [0, 1, 0, 0],
             'video_path': ['dummy1.mp4', 'dummy1.mp4', 'dummy2.mp4', 'dummy3.mp4'],
             'notes': ['', '', '', '']
         }
         df = pd.DataFrame(data)
         df.to_csv(self.test_csv, index=False)
         
    def test_csv_integrity(self):
         df = pd.read_csv(self.test_csv)
         
         # Check NaN in start/end sec
         self.assertFalse(df['start_sec'].isnull().any(), "NaN found in start_sec")
         self.assertFalse(df['end_sec'].isnull().any(), "NaN found in end_sec")
         
         # Check face_detected only contains 0 and 1
         valid_faces = df['face_detected'].isin([0, 1]).all()
         self.assertTrue(valid_faces, "face_detected must be only 0 or 1")
         
         # Check bounding box normalization bound between 0 and 1
         faces_df = df[df['face_detected'] == 1]
         for col in ['x1', 'y1', 'x2', 'y2']:
             self.assertTrue((faces_df[col] >= 0.0).all(), f"{col} has values < 0.0")
             self.assertTrue((faces_df[col] <= 1.0).all(), f"{col} has values > 1.0")
             
         # Check inversion validity
         self.assertTrue((faces_df['x2'] >= faces_df['x1']).all(), "x2 is smaller than x1")
         self.assertTrue((faces_df['y2'] >= faces_df['y1']).all(), "y2 is smaller than y1")
         
    def tearDown(self):
         if os.path.exists(self.test_csv):
              os.remove(self.test_csv)

if __name__ == '__main__':
    unittest.main()

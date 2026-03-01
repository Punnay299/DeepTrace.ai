import numpy as np

def compute_class4_heuristic(windows_list_224):
    """
    windows_list_224: list of np.ndarray(20, 224, 224, 3) 
    
    Calculates Class 4 heuristic. Fully Generated AI videos (Sora/Runway)
    exhibit non-human temporal structural variance (morphing/smearing).
    
    This computes sequence-wide pixel absolute difference variation over luma channels.
    """
    if not windows_list_224:
        return 0.0
        
    variance_scores = []
    
    for window in windows_list_224:
        # Convert to roughly grayscale by naive luma preservation (.299 R, .587 G, .114 B)
        # Window shape: (20, 224, 224, 3)
        gray = np.dot(window[...,:3], [0.2989, 0.5870, 0.1140]) # (20, 224, 224)
        
        # Temporal difference between consecutive frames
        diffs = np.abs(np.diff(gray, axis=0)) # (T-1, 224, 224)
        
        # Calculate variance across time dimension (this highlights flickering/morphing vs tracking)
        if diffs.shape[0] == 0:
            var_score = 0.0
        else:
            var_score = np.var(diffs)
            
        variance_scores.append(var_score)
        
    return float(np.mean(variance_scores))

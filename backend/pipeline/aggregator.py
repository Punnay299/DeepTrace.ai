def merge_flagged_windows(window_scores, threshold=0.6, tolerance_sec=2.0):
    """
    Merges a list of contiguous windows into defined 'suspicious ranges'.
    
    window_scores format: [{'start': start_sec, 'end': end_sec, 'score': aggregate_score}, ...]
    If the gap between two flagged windows is <= tolerance_sec, they are merged.
    """
    flagged = [w for w in window_scores if w['score'] > threshold]
    if not flagged:
        return []
        
    # Sort by start time just in case
    flagged.sort(key=lambda x: x['start'])
    
    merged = []
    current_range = None
    
    for w in flagged:
        if current_range is None:
            current_range = {
                'start': w['start'],
                'end': w['end'],
                'peak_score': w['score']
            }
        else:
            # Check overlap or proximity
            gap = w['start'] - current_range['end']
            if gap <= tolerance_sec:
                # Merge
                current_range['end'] = max(current_range['end'], w['end'])
                current_range['peak_score'] = max(current_range['peak_score'], w['score'])
            else:
                # Push and reset
                merged.append(current_range)
                current_range = {
                    'start': w['start'],
                    'end': w['end'],
                    'peak_score': w['score']
                }
                
    if current_range is not None:
        merged.append(current_range)
        
    # Filter out blips (e.g. less than 1.5 seconds) - but windows themselves are 4s
    # so any single window is inherently large enough.
        
    return merged

def calculate_verdict(aggregated_ranges, heuristic_score, heuristic_threshold=150.0):
    """
    Performs the decision tree from deepfake.md:
    - If heuristic_score > Class 4 Threshold -> Class 4 (Fully AI)
    - Else if len(aggregated_ranges) > 0 -> Class 1 (AI Video Swap)
    - Else -> Class 0 (Real)
    """
    if heuristic_score > heuristic_threshold:
        return 4, "FULLY_AI_VIDEO"
        
    if aggregated_ranges:
        return 1, "AI_VIDEO"
        
    return 0, "REAL"

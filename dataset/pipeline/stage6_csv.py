import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='dataset/labels/windows_index.csv')
    parser.add_argument('--out_dir', type=str, default='dataset/labels/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} not found.")
        return

    df = pd.read_csv(args.input_csv)
    
    # Save as master.csv for reference
    master_path = os.path.join(args.out_dir, 'master.csv')
    df.to_csv(master_path, index=False)
    print(f"Saved master dataset: {master_path} ({len(df)} windows)")
    
    # We must split by video_id to avoid data leakage
    video_ids = df['video_id'].unique()
    
    if len(video_ids) < 3:
        print("Warning: Not enough unique videos to split properly. Saving everything to train.")
        df.to_csv(os.path.join(args.out_dir, 'train.csv'), index=False)
        return
        
    # Split 80/10/10
    train_ids, temp_ids = train_test_split(video_ids, test_size=0.2, random_state=args.seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=args.seed)
    
    train_df = df[df['video_id'].isin(train_ids)]
    val_df = df[df['video_id'].isin(val_ids)]
    test_df = df[df['video_id'].isin(test_ids)]
    
    train_path = os.path.join(args.out_dir, 'train.csv')
    val_path = os.path.join(args.out_dir, 'val.csv')
    test_path = os.path.join(args.out_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\nSplit Details (Window Counts):")
    print(f"Train: {len(train_df)} ({len(train_ids)} videos)")
    print(f"Val:   {len(val_df)} ({len(val_ids)} videos)")
    print(f"Test:  {len(test_df)} ({len(test_ids)} videos)")
    print("Done!")

if __name__ == "__main__":
    main()

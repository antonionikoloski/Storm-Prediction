import pandas as pd
import numpy as np

def split_storms(csv_path, output_dir="data/splits", train_ratio=0.70, val_ratio=0.15):
    """
    Reads events.csv (assumed to be chronologically ordered), splits storm IDs 
    into train/val/test sets sequentially based on time. 
    """
    df = pd.read_csv(csv_path)
    unique_storms = df['id'].unique() 
    print(f"Found {len(unique_storms)} unique storms in {csv_path} (ordered chronologically).")


    n_total = len(unique_storms)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    n_test  = n_total - n_train - n_val

    train_storms = unique_storms[:n_train]             
    val_storms   = unique_storms[n_train:n_train + n_val] 
    test_storms  = unique_storms[n_train + n_val:]   

    print(f"Train: {len(train_storms)}, Val: {len(val_storms)}, Test: {len(test_storms)}")
    print(f"Train time range: First {train_ratio*100:.0f}% of storms (oldest)")
    print(f"Test time range: Last {(1 - train_ratio - val_ratio)*100:.0f}% of storms (newest)")


    import os
    os.makedirs(output_dir, exist_ok=True)

    for split_name, storms in zip(["train", "val", "test"], [train_storms, val_storms, test_storms]):
        with open(f"{output_dir}/{split_name}_ids.txt", "w") as f:
            for sid in storms:
                f.write(f"{sid}\n")

    print(f"Saved time-based splits to {output_dir}/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/split_data.py <events_csv> [output_dir]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/splits"
    split_storms(csv_path, out_dir)
"""
Hindsight Labeling Script
=====================================
Processes raw .pkl logs from DivergenceLogger and adds future-conditioned 
ground truth labels for analysis.

Labels added to each timestep:
- future_collision: bool (True if a collision occurs within horizon)
- future_near_miss: bool (True if a near_miss occurs within horizon)
- time_to_collision: float (seconds until the first collision event)

Usage:
    python Bench2Drive/tests/process_logs.py --input raw_logs/*.pkl --out processed_logs/
"""

import argparse
import glob
import os
import pickle
import numpy as np
from pathlib import Path


def process_route(input_path, output_dir, horizon_seconds=3.0):
    print(f"Processing: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    timesteps = data['timesteps']
    config = data.get('config', {})
    fps = config.get('fps', 20)
    horizon_frames = int(horizon_seconds * fps)
    
    num_steps = len(timesteps)
    
    # Extract raw event arrays for fast lookahead
    collisions = np.array([t.get('collision', False) for t in timesteps], dtype=bool)
    near_misses = np.array([t.get('near_miss', False) for t in timesteps], dtype=bool)
    
    for i in range(num_steps):
        # Lookahead window excludes current frame — measure prediction, not detection
        start = min(num_steps, i + 1)
        end = min(num_steps, i + 1 + horizon_frames)

        window_collisions = collisions[start:end]
        window_near_misses = near_misses[start:end]

        timesteps[i]['future_collision'] = bool(np.any(window_collisions))
        timesteps[i]['future_near_miss'] = bool(np.any(window_near_misses))

        # Time from frame i+1 to first collision; +1 converts slice-relative to frame-relative
        future_collision_indices = np.where(collisions[i + 1:])[0]
        if len(future_collision_indices) > 0:
            timesteps[i]['time_to_collision'] = float((future_collision_indices[0] + 1) / fps)
        else:
            timesteps[i]['time_to_collision'] = None

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    out_name = Path(input_path).stem + "_processed.pkl"
    out_path = Path(output_dir) / out_name
    
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"  → Saved: {out_path} ({num_steps} steps labeled)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Glob pattern for raw pkl files')
    parser.add_argument('--out', required=True, help='Output directory for processed pkl files')
    parser.add_argument('--horizon', type=float, default=3.0, help='Future horizon in seconds')
    args = parser.parse_args()
    
    input_files = glob.glob(args.input)
    if not input_files:
        print(f"No files found matching: {args.input}")
        return
    
    print(f"Found {len(input_files)} files to process.")
    for f in input_files:
        process_route(f, args.out, horizon_seconds=args.horizon)
    
    print("\n✓ Hindsight labeling complete.")


if __name__ == "__main__":
    main()

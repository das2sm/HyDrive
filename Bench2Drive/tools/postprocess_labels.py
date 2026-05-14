"""
postprocess_labels.py

Offline future-conditioned label generation for temporal divergence research.

This script:
1. Loads a raw logged route file from DivergenceLogger
2. Computes future-conditioned labels:
    - future_collision
    - time_to_collision
3. Saves a processed dataset file

Usage:
    python postprocess_labels.py \
        --input logs/route_01.pkl \
        --output processed/route_01_processed.pkl
"""

import argparse
import pickle
from pathlib import Path


FPS = 20
HORIZON_SECONDS = 3.0
FUTURE_WINDOW = int(FPS * HORIZON_SECONDS)


def compute_future_labels(timesteps):
    """
    Adds future-conditioned labels to each timestep.

    For timestep t:
        future_collision:
            True if collision occurs within next 3 seconds

        time_to_collision:
            Seconds until collision (None if no collision)
    """

    total_steps = len(timesteps)

    for i in range(total_steps):

        end_idx = min(i + FUTURE_WINDOW, total_steps)

        future_collision = False
        collision_frames = None

        # Search future horizon
        for j in range(i, end_idx):
            step_data = timesteps[j]

            # Collision detection
            if step_data["collision"]:
                future_collision = True
                if collision_frames is None:
                    collision_frames = j - i
                break  # stop at first collision

        # Convert frame counts to seconds
        time_to_collision = collision_frames / FPS if collision_frames is not None else None

        # Attach labels
        timesteps[i]["future_collision"] = future_collision
        timesteps[i]["time_to_collision"] = time_to_collision

    return timesteps


def print_statistics(timesteps):

    num_collision = sum(t["future_collision"] for t in timesteps)
    num_safe = len(timesteps) - num_collision

    print("\n========== Dataset Statistics ==========")
    print(f"Total timesteps:      {len(timesteps)}")
    print(f"Future collisions:    {num_collision}")
    print(f"Safe:                 {num_safe}")

    if len(timesteps) > 0:
        print(f"Collision ratio:      {num_collision / len(timesteps):.4f}")

    print("========================================\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw route pickle file"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to processed output pickle file"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"\nLoading: {input_path}")

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    timesteps = data["timesteps"]

    print(f"Loaded {len(timesteps)} timesteps")

    # Compute future-conditioned labels (only collision)
    timesteps = compute_future_labels(timesteps)

    # Print statistics
    print_statistics(timesteps)

    # Save processed dataset
    processed_data = {
        "timesteps": timesteps,
        "config": data.get("config", {}),
        "label_config": {
            "fps": FPS,
            "future_window_frames": FUTURE_WINDOW,
            "future_window_seconds": HORIZON_SECONDS,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)

    print(f"\nSaved processed dataset:")
    print(output_path)


if __name__ == "__main__":
    main()
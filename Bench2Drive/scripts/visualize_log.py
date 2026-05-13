import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('close_loop_log/save/f2d_eval/divergence_logs/route_f2d_eval.pkl', 'rb') as f:
    data = pickle.load(f)

timesteps = data['timesteps']

# Plot basic signals over time
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

steps = [t['step'] for t in timesteps]
speeds = [t['ego_speed'] for t in timesteps]
ttcs = [t['ttc'] for t in timesteps]
min_dists = [t['min_distance'] for t in timesteps]

axes[0].plot(steps, speeds)
axes[0].set_ylabel('Speed (m/s)')
axes[0].grid(True)

axes[1].plot(steps, ttcs)
axes[1].set_ylabel('TTC (s)')
axes[1].set_ylim(0, 10)
axes[1].grid(True)

axes[2].plot(steps, min_dists)
axes[2].set_ylabel('Min Distance (m)')
axes[2].set_xlabel('Step')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('pilot_signals.png')
print("Saved pilot_signals.png")
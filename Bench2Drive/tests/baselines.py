"""
Baseline safety signal computation.

Signals:
  - ttc_signal:      raw TTC (lower = more dangerous); sentinel 999 → clipped to max
  - dist_signal:     raw min_distance; sentinel 999 → clipped to max
  - rss_signal:      binary RSS proxy: TTC < 2.0s AND dist < 3.0m
  - ttc_inv:         1/TTC (higher = more dangerous, comparable to divergence direction)
  - dist_inv:        1/dist

All signals are returned as arrays of length N.
Sentinel values (999) are treated as "no hazard" and clipped to a safe maximum.
"""

import numpy as np

TTC_SENTINEL = 90.0   # Guardian uses 99.0; treat anything ≥90 as "no hazard"
DIST_SENTINEL = 90.0  # Same threshold covers both 99.0 and 999.0 sentinels
TTC_CLIP = 10.0    # cap TTC at 10s for inversion
DIST_CLIP = 20.0   # cap distance at 20m for inversion

RSS_TTC_THRESH = 2.0   # seconds
RSS_DIST_THRESH = 3.0  # meters


def compute_baseline_series(timesteps):
    """
    Extract and derive baseline signals from logged timesteps.

    Returns dict with arrays of length N.
    """
    N = len(timesteps)
    ttc_raw = np.array([t['ttc'] for t in timesteps], dtype=np.float64)
    dist_raw = np.array([t['min_distance'] for t in timesteps], dtype=np.float64)
    speed = np.array([t['ego_speed'] for t in timesteps], dtype=np.float64)

    # Replace sentinels with clip values (= no hazard)
    ttc_clipped = np.where(ttc_raw >= TTC_SENTINEL, TTC_CLIP, np.clip(ttc_raw, 0.01, TTC_CLIP))
    dist_clipped = np.where(dist_raw >= DIST_SENTINEL, DIST_CLIP, np.clip(dist_raw, 0.01, DIST_CLIP))

    ttc_inv = 1.0 / ttc_clipped      # higher = more dangerous
    dist_inv = 1.0 / dist_clipped    # higher = more dangerous

    # RSS proxy: both conditions must hold
    rss = ((ttc_raw < RSS_TTC_THRESH) & (dist_raw < RSS_DIST_THRESH)).astype(np.float64)

    return {
        'ttc_raw': ttc_raw,
        'dist_raw': dist_raw,
        'ttc_inv': ttc_inv,
        'dist_inv': dist_inv,
        'rss': rss,
        'speed': speed,
    }

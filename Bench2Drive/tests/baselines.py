"""
Baseline safety signal computation.

Signals:
  - ttc_raw:         raw TTC (lower = more dangerous); >= 99s → clipped to max (10s)
  - dist_raw:        raw min_distance (lower = more dangerous); >= 99m → clipped to max (20m)
  - rss_signal:      binary RSS proxy: TTC < 2.0s AND dist < 3.0m
  - ttc_inv:         1/TTC (higher = more dangerous, comparable to divergence direction)
  - dist_inv:        1/dist

All signals are returned as arrays of length N.
Sentinel values are treated as "no hazard" and clipped to a safe maximum.
"""

import numpy as np

TTC_SENTINEL = 99.0   
DIST_SENTINEL = 99.0  
TTC_CLIP = 10.0    
DIST_CLIP = 20.0   
RSS_TTC_THRESH = 2.0   
RSS_DIST_THRESH = 3.0  
EPSILON = 0.01  

def compute_baseline_series(timesteps):
    """
    Extract and derive baseline signals with robust boundary validation.
    """
    keys = ['ttc_raw', 'dist_raw', 'ttc_inv', 'dist_inv', 'rss', 'speed', 'valid_mask']
    if not timesteps:
        return {k: np.array([], dtype=np.float64) for k in keys}

    # Field extraction handling dicts, classes, and explicit Nones
    def extract_field(key):
        vals = []
        for t in timesteps:
            if t is None:
              vals.append(np.nan)
              continue
            val = t.get(key) if hasattr(t, 'get') else getattr(t, key, np.nan)
            vals.append(val if val is not None else np.nan)
        return np.array(vals, dtype=np.float64)

    ttc_raw = extract_field('ttc')
    dist_raw = extract_field('min_distance')
    speed = extract_field('ego_speed')

    # 1. Check for truly invalid/missing data (NaNs, negative numbers or 0)
    ttc_data_present = np.isfinite(ttc_raw) & (ttc_raw > 0)
    dist_data_present = np.isfinite(dist_raw) & (dist_raw > 0)
    valid_mask = ttc_data_present & dist_data_present

    # 2. Identify active hazard zones (below sentinel thresholds)
    ttc_under_sentinel = ttc_raw < TTC_SENTINEL
    dist_under_sentinel = dist_raw < DIST_SENTINEL

    hazard_ttc = ttc_data_present & ttc_under_sentinel   # True when a real obstacle is close
    hazard_dist = dist_data_present & dist_under_sentinel

    # 3. Clip logic: True hazards get clipped normally. Sentinels/Invalids get safe max.
    ttc_clipped = np.where(hazard_ttc, np.clip(ttc_raw, EPSILON, TTC_CLIP), TTC_CLIP)  
    dist_clipped = np.where(hazard_dist, np.clip(dist_raw, EPSILON, DIST_CLIP), DIST_CLIP)  

    # Base inverses (still safe, never infinite)
    ttc_inv = np.where(hazard_ttc, 1.0 / ttc_clipped, 0.0)
    dist_inv = np.where(hazard_dist, 1.0 / dist_clipped, 0.0)

    ttc_inv[~valid_mask] = np.nan
    dist_inv[~valid_mask] = np.nan

    # RSS check evaluation
    rss = ((ttc_clipped < RSS_TTC_THRESH) & (dist_clipped < RSS_DIST_THRESH)).astype(np.float64)
    
    # Missing data yields NaN, but valid sentinel values safely yield 0.0
    rss[~valid_mask] = np.nan

    return {
        'ttc_raw': ttc_raw,
        'dist_raw': dist_raw,
        'ttc_inv': ttc_inv,
        'dist_inv': dist_inv,
        'rss': rss,
        'speed': speed,
        'valid_mask': valid_mask
    }

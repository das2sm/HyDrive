"""
Transparent heuristic safety baselines for temporal failure prediction.

The TTC and distance signals logged by Guardian are proxy measurements. Values
above the configured evaluation horizon are treated as right-censored "no
hazard within horizon" observations. Missing/error measurements are marked
invalid and returned as NaN so metric helpers can exclude them.
"""

import numpy as np


# Default analysis constants. Treat these as reported hyperparameters, not
# standards-compliant safety thresholds.
TTC_HORIZON = 5.0
DIST_HORIZON = 30.0
TTC_TAU = 1.5
DIST_TAU = 10.0
SMOOTHING_WINDOW = 5
MAX_REALISTIC_SPEED = 45.0
TTC_RIGHT_CENSOR = 90.0
DIST_RIGHT_CENSOR = 90.0
RSS_TTC_THRESH = 2.0
RSS_DIST_THRESH = 3.0
INVALID_OCCUPANCY_SOURCES = {"error", "none", "init"}


def _as_float(value, default=np.nan):
    if value is None:
        return default
    try:
        arr = np.asarray(value).reshape(-1)
        if len(arr) == 0:
            return default
        return float(arr[0])
    except (TypeError, ValueError):
        return default


def _get_field(timestep, key, default=None):
    if timestep is None:
        return default
    if hasattr(timestep, "get"):
        return timestep.get(key, default)
    return getattr(timestep, key, default)


def _extract_float(timesteps, key, default=np.nan):
    return np.array(
        [_as_float(_get_field(t, key, default), default=default) for t in timesteps],
        dtype=np.float64,
    )


def _extract_optional_bool(timesteps, keys):
    values = []
    seen = False
    for t in timesteps:
        value = None
        for key in keys:
            candidate = _get_field(t, key, None)
            if candidate is not None:
                value = candidate
                seen = True
                break
        values.append(bool(value))
    if not seen:
        return None
    return np.array(values, dtype=bool)


def _extract_metadata_string(timesteps, key, default=""):
    vals = []
    for t in timesteps:
        metadata = _get_field(t, "metadata", {}) or {}
        val = _get_field(metadata, key, default)
        vals.append(str(val) if val is not None else default)
    return np.array(vals, dtype=object)


def _causal_smooth(data, window):
    """Causal moving average that preserves NaN on frames with invalid current data."""
    if data is None or len(data) == 0:
        return np.array([], dtype=np.float64)
    
    data = np.array(data, dtype=np.float64)
    
    if window <= 1:
        return data

    smoothed = np.full_like(data, np.nan)
    for i in range(len(data)):
        if not np.isfinite(data[i]):
            continue
        start = max(0, i - window + 1)
        valid = np.isfinite(data[start : i + 1])
        if valid.any():
            smoothed[i] = np.mean(data[start : i + 1][valid])
    return smoothed


def _infer_valid(raw, explicit_valid=None, source_invalid=None):
    if explicit_valid is not None:
        valid = np.asarray(explicit_valid, dtype=bool)
    else:
        valid = np.isfinite(raw) & (raw >= 0.0)
    if source_invalid is not None:
        valid &= ~source_invalid
    return valid


def _right_censor(raw, valid, horizon, censor_value):
    raw = np.asarray(raw, dtype=np.float64)
    clean = np.full_like(raw, np.nan)

    censored = valid & (raw >= censor_value)
    observed = valid & ~censored

    clean[censored] = horizon
    clean[observed] = np.clip(raw[observed], 0.0, horizon)
    return clean


def _exp_risk(clean, tau, smoothing_window):
    risk = np.full_like(clean, np.nan)
    valid = np.isfinite(clean)
    risk[valid] = np.exp(-clean[valid] / tau)
    return _causal_smooth(risk, smoothing_window)


def compute_baseline_series(
    timesteps,
    *,
    ttc_horizon=TTC_HORIZON,
    dist_horizon=DIST_HORIZON,
    ttc_tau=TTC_TAU,
    dist_tau=DIST_TAU,
    smoothing_window=SMOOTHING_WINDOW,
):
    """
    Compute causal heuristic baseline signals.

    Score keys:
      - ttc_risk: Guardian path-occupancy TTC risk (min_dist / speed)
      - dist_risk: Guardian path-occupancy distance risk
      - collision_cls: SparseDrive learned collision classifier
      - point_collision_cls_mean: mean per-waypoint collision score
    """
    n = len(timesteps)
    if n == 0:
        keys = [
            "ttc_raw", "ttc_risk", "dist_risk",
            "collision_cls", "point_collision_cls_mean",
            "speed",
        ]
        return {k: np.array([], dtype=np.float64) for k in keys}

    ttc_raw = _extract_float(timesteps, "ttc")
    dist_raw = _extract_float(timesteps, "min_distance")
    speed = _extract_float(timesteps, "ego_speed")

    ttc_explicit = _extract_optional_bool(timesteps, ["ttc_valid"])
    dist_explicit = _extract_optional_bool(timesteps, ["min_distance_valid"])
    speed_valid = np.isfinite(speed) & (speed >= -5.0) & (speed <= MAX_REALISTIC_SPEED)

    # Learned collision prediction
    collision_cls = _extract_float(timesteps, "collision_cls")
    point_collision_cls_mean = _extract_float(timesteps, "point_collision_cls_mean")

    occupancy_source = _extract_metadata_string(timesteps, "occupancy_source")
    source_invalid = np.isin(occupancy_source, INVALID_OCCUPANCY_SOURCES)

    ttc_valid = _infer_valid(ttc_raw, ttc_explicit, source_invalid) & speed_valid
    dist_valid = _infer_valid(dist_raw, dist_explicit, source_invalid)

    ttc_clean = _right_censor(
        ttc_raw,
        ttc_valid,
        horizon=ttc_horizon,
        censor_value=TTC_RIGHT_CENSOR,
    )
    dist_clean = _right_censor(
        dist_raw,
        dist_valid,
        horizon=dist_horizon,
        censor_value=DIST_RIGHT_CENSOR,
    )

    ttc_risk = _exp_risk(ttc_clean, ttc_tau, smoothing_window)
    dist_risk = _exp_risk(dist_clean, dist_tau, smoothing_window)

    return {
        "ttc_raw": ttc_raw,
        "ttc_risk": ttc_risk,
        "dist_risk": dist_risk,
        "collision_cls": collision_cls,
        "point_collision_cls_mean": point_collision_cls_mean,
        "speed": speed,
    }

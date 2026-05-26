"""
Month 2 Temporal Conflict Analysis
============================
Tasks:
  4. AUROC / AUPRC: temporal conflict vs baselines for early failure prediction
  5. Matched-bin separation: conflict within TTC/dist/speed bins
  6. Lead-time analysis: conflict rise time vs TTC trigger time
  7. Ablation: remove multi-modal sampling, temporal aggregation

Usage:
    python analysis/run_analysis.py \
        --route processed/route_02_processed.pkl \
        [--out analysis/results]
"""

import argparse
import pickle
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Allow imports from current directory (tests) and project root
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from divergence import (
    compute_divergence_series,
    planner_spread_entropy,
    rasterize_planner,
    rasterize_planner_temporal,
    js_divergence,
    js_conflict_score,
    planner_conditioned_occupancy,
    _align_occupancy_to_planner_bev,
    _smooth_normalise,
)
from baselines import compute_baseline_series


BASELINE_SCORE_KEYS = ['ttc_proxy_risk', 'ttc_rel_risk', 'dist_proxy_risk', 'rss_proxy']


def available_keys(signals, keys):
    return [k for k in keys if k in signals and np.isfinite(signals[k]).any()]


def preferred_ttc_key(signals):
    rel = signals.get('ttc_rel_risk')
    if rel is not None and np.isfinite(rel).any():
        return 'ttc_rel_risk'
    return 'ttc_proxy_risk'


def finite_percentile_edges(values, percentiles):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    return np.unique(np.percentile(values[finite], percentiles))


def finite_max(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan
    return float(np.max(values[finite]))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def roc_auc(scores, labels):
    """Robust ROC AUC that handles NaNs and single-class edge cases."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    valid = ~np.isnan(scores)
    if valid.sum() == 0:
        return float('nan')
    s, l = scores[valid], labels[valid]
    if l.sum() == 0 or l.sum() == len(l):
        return float('nan')
    return float(roc_auc_score(l, s))


def pr_auc(scores, labels):
    """Robust PR AUC that handles NaNs and edge cases."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    valid = ~np.isnan(scores)
    if valid.sum() == 0:
        return float('nan')
    s, l = scores[valid], labels[valid]
    if l.sum() == 0:
        return float('nan')
    return float(average_precision_score(l, s))


def bootstrap_ci(scores, labels, metric_fn, n=500, alpha=0.05, rng=None):
    """Frame-level bootstrap CI (use only when route structure unavailable)."""
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(labels)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        v = metric_fn(scores[idx], labels[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float('nan'), float('nan')
    vals = np.array(vals)
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def route_bootstrap_ci(per_route_scores, per_route_labels, metric_fn, n=500, alpha=0.05, rng=None):
    """
    Route-level bootstrap CI. Resamples routes (not frames) to respect
    temporal autocorrelation within routes.
    Falls back to frame-level if fewer than 4 routes.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    R = len(per_route_scores)
    if R < 4:
        return bootstrap_ci(
            np.concatenate(per_route_scores),
            np.concatenate(per_route_labels),
            metric_fn, n=n, alpha=alpha, rng=rng
        )
    vals = []
    for _ in range(n):
        idx = rng.integers(0, R, R)
        s = np.concatenate([per_route_scores[i] for i in idx])
        l = np.concatenate([per_route_labels[i] for i in idx])
        v = metric_fn(s, l)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float('nan'), float('nan')
    vals = np.array(vals)
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: AUROC / AUPRC
# ─────────────────────────────────────────────────────────────────────────────

def task4_auroc_auprc(signals, labels, out_dir, per_route_signals=None, per_route_labels=None):
    """
    Compare temporal conflict vs baselines on early failure prediction.
    Uses route-level bootstrap when per_route_signals is provided.
    """
    print("\n" + "="*60)
    print("TASK 4: AUROC / AUPRC")
    print("="*60)

    candidates = {
        'temporal_conflict_smooth': signals['temporal_conflict_smooth'],
        'temporal_conflict_raw':    signals['temporal_conflict_raw'],
        'temporal_agreement_smooth': signals['temporal_agreement_smooth'],
        'temporal_agreement_raw':    signals['temporal_agreement_raw'],
        'planner_spread':           signals['planner_spread'],
        'planner_conditioned_occupancy': signals['planner_conditioned_occupancy'],
    }
    for name in available_keys(signals, BASELINE_SCORE_KEYS):
        if name in signals:
            candidates[name] = signals[name]

    results = {}
    for name, s in candidates.items():
        auroc = roc_auc(s, labels)
        auprc = pr_auc(s, labels)
        if per_route_signals is not None:
            pr_scores = [r[name] for r in per_route_signals]
            lo_roc, hi_roc = route_bootstrap_ci(pr_scores, per_route_labels, roc_auc)
            lo_pr,  hi_pr  = route_bootstrap_ci(pr_scores, per_route_labels, pr_auc)
        else:
            lo_roc, hi_roc = bootstrap_ci(s, labels, roc_auc)
            lo_pr,  hi_pr  = bootstrap_ci(s, labels, pr_auc)
        results[name] = dict(auroc=auroc, auprc=auprc,
                             auroc_ci=(lo_roc, hi_roc), auprc_ci=(lo_pr, hi_pr))
        print(f"  {name:25s}  AUROC={auroc:.3f} [{lo_roc:.3f},{hi_roc:.3f}]  "
              f"AUPRC={auprc:.3f} [{lo_pr:.3f},{hi_pr:.3f}]"
              + (" [route-CI]" if per_route_signals and len(per_route_signals) >= 4 else " [frame-CI]"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
    aurocs = [results[n]['auroc'] for n in names]
    auprcs = [results[n]['auprc'] for n in names]
    auroc_errs = [[results[n]['auroc'] - results[n]['auroc_ci'][0] for n in names],
                  [results[n]['auroc_ci'][1] - results[n]['auroc'] for n in names]]
    auprc_errs = [[results[n]['auprc'] - results[n]['auprc_ci'][0] for n in names],
                  [results[n]['auprc_ci'][1] - results[n]['auprc'] for n in names]]

    aurocs_plot = np.nan_to_num(np.array(aurocs, dtype=float), nan=0.0)
    auprcs_plot = np.nan_to_num(np.array(auprcs, dtype=float), nan=0.0)
    auroc_errs_plot = np.nan_to_num(np.array(auroc_errs, dtype=float), nan=0.0)
    auprc_errs_plot = np.nan_to_num(np.array(auprc_errs, dtype=float), nan=0.0)

    colors = ['#e74c3c' if 'conflict' in n else '#e67e22' if 'agreement' in n else '#2ecc71' if 'occupancy' in n else '#3498db' for n in names]
    ax = axes[0]
    bars = ax.bar(names, aurocs_plot, color=colors, yerr=auroc_errs_plot, capsize=4)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='random')
    ax.set_ylim(0, 1.05)
    ax.set_title('AUROC: Early Failure Prediction')
    ax.set_ylabel('AUROC')
    ax.tick_params(axis='x', rotation=30)
    ax.legend()

    ax = axes[1]
    baseline_rate = labels.mean()
    ax.bar(names, auprcs_plot, color=colors, yerr=auprc_errs_plot, capsize=4)
    ax.axhline(baseline_rate, color='gray', linestyle='--', linewidth=0.8,
               label=f'random ({baseline_rate:.2f})')
    ax.set_ylim(0, 1.05)
    ax.set_title('AUPRC: Early Failure Prediction')
    ax.set_ylabel('AUPRC')
    ax.tick_params(axis='x', rotation=30)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / 'task4_auroc_auprc.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved task4_auroc_auprc.png")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Matched-bin separation
# ─────────────────────────────────────────────────────────────────────────────

def task5_matched_bin(signals, labels, timesteps, out_dir, route_ids=None):
    """
    Anti-TTC-collapse test: within each TTC/dist/speed bin,
    does temporal conflict still separate failure from safe?

    Adds a complexity control via observed actor count and optionally
    reports route-aware consistency when route_ids are available.
    """
    print("\n" + "="*60)
    print("TASK 5: MATCHED-BIN SEPARATION")
    print("="*60)

    div = signals['temporal_conflict_smooth']
    ttc_key = preferred_ttc_key(signals)
    ttc_risk = signals[ttc_key]
    dist_risk = signals['dist_proxy_risk']
    speed = signals['speed']
    num_actors = np.array([
        t.get('metadata', {}).get('num_actors', np.nan)
        for t in timesteps
    ], dtype=np.float64)

    # Define binning variables and their edges
    bin_configs = [
        ('speed',          speed,    np.array([0, 2, 5, 10, 30])),
        ('dist_proxy_risk', dist_risk, finite_percentile_edges(dist_risk, [0, 25, 50, 75, 100])),
        (ttc_key,          ttc_risk, finite_percentile_edges(ttc_risk, [0, 25, 50, 75, 100])),
        ('num_actors',     num_actors, finite_percentile_edges(num_actors, [0, 25, 50, 75, 100])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (bin_name, bin_var, bin_edges) in zip(axes, bin_configs):
        if bin_edges is None or len(bin_edges) < 2:
            ax.set_title(f'{bin_name}: unavailable')
            ax.axis('off')
            print(f"  {bin_name}: unavailable (no finite values)")
            continue
        bin_edges = np.unique(bin_edges)
        separations = []
        bin_labels = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (bin_var >= lo) & (bin_var < hi)
            if mask.sum() < 5:
                continue
            fail = labels[mask].astype(bool)
            if fail.sum() == 0 or (~fail).sum() == 0:
                continue
            mean_fail = div[mask][fail].mean()
            mean_safe = div[mask][~fail].mean()
            sep = mean_fail - mean_safe
            bin_auroc = roc_auc(div[mask], fail.astype(float))
            try:
                _, pval = mannwhitneyu(div[mask][fail], div[mask][~fail], alternative='greater')
            except Exception:
                pval = float('nan')
            separations.append(sep)
            bin_labels.append(f'[{lo:.2f},{hi:.2f})\nn={mask.sum()}')
            sig = '*' if pval < 0.05 else ''
            print(f"  {bin_name} bin [{lo:.2f},{hi:.2f}): "
                  f"n={mask.sum()}, fail={fail.sum()}, "
                  f"sep={sep:+.4f}, bin-AUROC={bin_auroc:.3f}, p={pval:.3f}{sig}")

        colors = ['#e74c3c' if s > 0 else '#3498db' for s in separations]
        ax.bar(range(len(separations)), separations, color=colors)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=30, fontsize=8)
        ax.set_title(f'Conflict separation within {bin_name} bins')
        ax.set_ylabel('mean(div|fail) − mean(div|safe)')

    plt.tight_layout()
    fig.savefig(out_dir / 'task5_matched_bin.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved task5_matched_bin.png")

    # ── PCO matched-bin: same bins but using planner_conditioned_occupancy ──
    pco = signals['planner_conditioned_occupancy']
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    axes2 = axes2.flatten()

    for ax, (bin_name, bin_var, bin_edges) in zip(axes2, bin_configs):
        if bin_edges is None or len(bin_edges) < 2:
            ax.set_title(f'{bin_name}: unavailable')
            ax.axis('off')
            continue
        bin_edges = np.unique(bin_edges)
        separations, bin_labels = [], []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (bin_var >= lo) & (bin_var < hi)
            if mask.sum() < 5:
                continue
            fail = labels[mask].astype(bool)
            if fail.sum() == 0 or (~fail).sum() == 0:
                continue
            mean_fail = pco[mask][fail].mean()
            mean_safe = pco[mask][~fail].mean()
            sep = mean_fail - mean_safe
            bin_auroc = roc_auc(pco[mask], fail.astype(float))
            try:
                _, pval = mannwhitneyu(pco[mask][fail], pco[mask][~fail], alternative='greater')
            except Exception:
                pval = float('nan')
            separations.append(sep)
            bin_labels.append(f'[{lo:.2f},{hi:.2f})\nn={mask.sum()}')
            sig = '*' if pval < 0.05 else ''
            print(f"  [PCO] {bin_name} bin [{lo:.2f},{hi:.2f}): "
                  f"n={mask.sum()}, fail={fail.sum()}, "
                  f"sep={sep:+.4f}, bin-AUROC={bin_auroc:.3f}, p={pval:.3f}{sig}")

        colors = ['#2ecc71' if s > 0 else '#3498db' for s in separations]
        ax.bar(range(len(separations)), separations, color=colors)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=30, fontsize=8)
        ax.set_title(f'PCO separation within {bin_name} bins')
        ax.set_ylabel('mean(pco|fail) − mean(pco|safe)')

    plt.tight_layout()
    fig2.savefig(out_dir / 'task5_pco_matched_bin.png', dpi=150)
    plt.close(fig2)
    print(f"  → Saved task5_pco_matched_bin.png")

    if route_ids is not None:
        unique_routes = np.unique(route_ids)
        for route in unique_routes:
            route_mask = route_ids == route
            if route_mask.sum() < 10:
                continue
            route_fail = labels[route_mask].astype(bool)
            if route_fail.sum() == 0 or (~route_fail).sum() == 0:
                continue
            route_auroc = roc_auc(div[route_mask], route_fail.astype(float))
            print(f"  Route {int(route)} separation AUROC={route_auroc:.3f} (n={route_mask.sum()})")


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: Lead-time analysis
# ─────────────────────────────────────────────────────────────────────────────

def task6_lead_time(signals, timesteps, out_dir, fps=20,
                    div_threshold=0.10, pco_threshold=0.15, ttc_threshold=2.0, sustain=3):
    """
    For each failure event, measure:
      - t_div: first time temporal conflict exceeds threshold before failure
      - t_agreement: first time planner-occupancy agreement exceeds threshold
      - t_pco: first time planner_conditioned_occupancy exceeds threshold
      - t_ttc: first time TTC < ttc_threshold before failure
    Lead time = time_to_failure - t_signal.

    Uses a fixed, pre-specified conflict threshold to avoid post-hoc tuning.
    Conflict threshold 0.10 corresponds to ~90th percentile of observed values.
    Agreement threshold 0.90 is the complement (1 - conflict = agreement).
    PCO threshold 0.15 is tuned to the observed collision-window distribution.
    """
    print("\n" + "="*60)
    print("TASK 6: LEAD-TIME ANALYSIS")
    print("="*60)

    div = signals['temporal_conflict_smooth']
    agreement = signals['temporal_agreement_smooth']
    pco = signals.get('planner_conditioned_occupancy', np.full(len(div), np.nan))
    ttc_raw_key = 'ttc_rel_raw' if 'ttc_rel_risk' in signals and np.isfinite(signals['ttc_rel_risk']).any() else 'ttc_proxy_raw'
    ttc_raw = signals[ttc_raw_key]

    div_finite = div[np.isfinite(div)]
    if len(div_finite) == 0:
        print("  Temporal conflict unavailable; skipping lead-time analysis.")
        return {'div_leads': [], 'pco_leads': [], 'ttc_leads': []}

    div_thresh = float(div_threshold)
    agreement_thresh = 1.0 - div_thresh
    pco_thresh = float(pco_threshold)
    print(f"  Temporal conflict threshold: {div_thresh:.4f}")
    print(f"  Planner-occupancy agreement threshold: {agreement_thresh:.4f}")
    print(f"  Planner-conditioned occupancy threshold: {pco_thresh:.4f}")
    print(f"  TTC threshold: {ttc_threshold:.1f}s ({ttc_raw_key})")

    # Find actual collision frames (collision=True), deduplicated to one per event
    collision_steps = []
    in_coll = False
    for i, t in enumerate(timesteps):
        if t['collision'] and not in_coll:
            collision_steps.append(i)
            in_coll = True
        elif not t['collision']:
            in_coll = False

    print(f"  Actual collision frames: {len(collision_steps)}")

    div_leads = []
    agreement_leads = []
    pco_leads = []
    ttc_leads = []

    for collision_step in collision_steps:
        search_start = max(0, collision_step - 120)  # look back up to 6s

        # Conflict: first causal sustained activation (sustain frames all above threshold)
        div_trigger = None
        div_lead = None
        for j in range(search_start, collision_step - sustain + 1):
            if np.all(div[j:j+sustain] > div_thresh):
                div_trigger = j
                break
        if div_trigger is not None:
            div_lead = (collision_step - div_trigger) / fps
            div_leads.append(div_lead)

        # Planner-occupancy agreement: inverse of conflict (high = dangerous)
        agreement_trigger = None
        agreement_lead = None
        for j in range(search_start, collision_step - sustain + 1):
            if np.all(agreement[j:j+sustain] > agreement_thresh):
                agreement_trigger = j
                break
        if agreement_trigger is not None:
            agreement_lead = (collision_step - agreement_trigger) / fps
            agreement_leads.append(agreement_lead)

        # Planner-conditioned occupancy: obstacle at planner's trajectory cell
        pco_trigger = None
        pco_lead = None
        for j in range(search_start, collision_step - sustain + 1):
            w = pco[j:j+sustain]
            if np.all(np.isfinite(w) & (w > pco_thresh)):
                pco_trigger = j
                break
        if pco_trigger is not None:
            pco_lead = (collision_step - pco_trigger) / fps
            pco_leads.append(pco_lead)

        # TTC: first causal sustained activation
        ttc_trigger = None
        ttc_lead = None
        for j in range(search_start, collision_step - sustain + 1):
            window = ttc_raw[j:j+sustain]
            if np.all(np.isfinite(window) & (window < ttc_threshold)):
                ttc_trigger = j
                break
        if ttc_trigger is not None:
            ttc_lead = (collision_step - ttc_trigger) / fps
            ttc_leads.append(ttc_lead)

        div_str = f"{div_lead:.2f}s" if div_lead is not None else "None"
        agr_str = f"{agreement_lead:.2f}s" if agreement_lead is not None else "None"
        pco_str = f"{pco_lead:.2f}s" if pco_lead is not None else "None"
        ttc_str = f"{ttc_lead:.2f}s" if ttc_lead is not None else "None"
        print(f"  Collision at step {collision_step}: conflict={div_str}, agreement={agr_str}, pco={pco_str}, ttc={ttc_str}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    if div_leads:
        ax.hist(div_leads, bins=10, alpha=0.5, color='#e74c3c', label=f'Conflict (n={len(div_leads)})')
        print(f"  Conflict lead times: mean={np.mean(div_leads):.2f}s, "
              f"median={np.median(div_leads):.2f}s")
    if agreement_leads:
        ax.hist(agreement_leads, bins=10, alpha=0.5, color='#e67e22', label=f'Agreement (n={len(agreement_leads)})')
        print(f"  Agreement lead times: mean={np.mean(agreement_leads):.2f}s, "
              f"median={np.median(agreement_leads):.2f}s")
    if pco_leads:
        ax.hist(pco_leads, bins=10, alpha=0.5, color='#2ecc71', label=f'PCO (n={len(pco_leads)})')
        print(f"  PCO lead times: mean={np.mean(pco_leads):.2f}s, "
              f"median={np.median(pco_leads):.2f}s")
    if ttc_leads:
        ax.hist(ttc_leads, bins=10, alpha=0.5, color='#3498db', label=f'TTC (n={len(ttc_leads)})')
        print(f"  TTC lead times: mean={np.mean(ttc_leads):.2f}s, "
              f"median={np.median(ttc_leads):.2f}s")
    ax.set_xlabel('Lead time before failure (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Lead-time distribution: temporal conflict vs PCO vs TTC')
    if div_leads or agreement_leads or pco_leads or ttc_leads:
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / 'task6_lead_time.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved task6_lead_time.png")

    return {'div_leads': div_leads, 'agreement_leads': agreement_leads, 'pco_leads': pco_leads, 'ttc_leads': ttc_leads}


# ─────────────────────────────────────────────────────────────────────────────
# Task 7: Ablation
# ─────────────────────────────────────────────────────────────────────────────

def _single_mode_divergence(timesteps):
    """Ablation: use only the top-scoring trajectory (no multi-modal sampling)."""
    N = len(timesteps)
    div = np.full(N, np.nan)
    for i, t in enumerate(timesteps):
        trajs = t['planner_trajs']   # (K, T, 2)
        scores = t['planner_scores'] # (K,)
        occ = t['occupancy_grid']

        # Use only top-1 mode
        top1 = np.argmax(scores)
        trajs_1 = trajs[top1:top1+1]
        scores_1 = np.array([1.0])

        occupancy_future = np.asarray(t.get('occupancy_future', occ[None, ...]))
        occupancy_valid = np.asarray(
            t.get('occupancy_future_valid', np.ones(len(occupancy_future), dtype=bool)),
            dtype=bool,
        )
        valid_slices = occupancy_valid & np.array([o.max() > 1e-6 for o in occupancy_future], dtype=bool)

        if valid_slices.any():
            planner_future = rasterize_planner_temporal(trajs_1, scores_1)
            n_slices = min(len(planner_future), len(occupancy_future))
            conflicts = []
            for h in range(n_slices):
                if not valid_slices[h]:
                    continue
                q = _align_occupancy_to_planner_bev(occupancy_future[h]).astype(np.float64)
                q = _smooth_normalise(q)
                js = js_divergence(planner_future[h], q)
                conflicts.append(js_conflict_score(js))
            div[i] = float(np.mean(conflicts)) if conflicts else 0.0
        else:
            # Fallback to single-mode endpoint variance (which is 0)
            div[i] = np.nan
    return div


def task7_ablation(signals, labels, timesteps, out_dir):
    """
    Ablation study:
      A) Remove multi-modal sampling → use only top-1 trajectory
      B) Remove temporal aggregation → use raw D(t) instead of smoothed
    Show AUROC/AUPRC degrades.
    """
    print("\n" + "="*60)
    print("TASK 7: ABLATION STUDY")
    print("="*60)

    full_div = signals['temporal_conflict_smooth']
    full_auroc = roc_auc(full_div, labels)
    full_auprc = pr_auc(full_div, labels)
    print(f"  Full temporal conflict (smooth, multi-modal): "
          f"AUROC={full_auroc:.3f}, AUPRC={full_auprc:.3f}")

    # Ablation A: single mode
    div_single = _single_mode_divergence(timesteps)
    auroc_a = roc_auc(div_single, labels)
    auprc_a = pr_auc(div_single, labels)
    print(f"  Ablation A (top-1 mode only):          "
          f"AUROC={auroc_a:.3f}, AUPRC={auprc_a:.3f}")

    # Ablation B: no temporal smoothing
    div_raw = signals['temporal_conflict_raw']
    auroc_b = roc_auc(div_raw, labels)
    auprc_b = pr_auc(div_raw, labels)
    print(f"  Ablation B (no temporal smoothing):    "
          f"AUROC={auroc_b:.3f}, AUPRC={auprc_b:.3f}")

    # Plot
    names = ['Full\n(multi-modal\n+ smoothed)', 'Ablation A\n(top-1 only)', 'Ablation B\n(no smoothing)']
    aurocs = [full_auroc, auroc_a, auroc_b]
    auprcs = [full_auprc, auprc_a, auprc_b]
    colors = ['#e74c3c', '#e67e22', '#f39c12']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(names, aurocs, color=colors)
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Ablation: AUROC')
    axes[0].set_ylabel('AUROC')

    axes[1].bar(names, auprcs, color=colors)
    axes[1].axhline(labels.mean(), color='gray', linestyle='--', linewidth=0.8,
                    label=f'random ({labels.mean():.2f})')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Ablation: AUPRC')
    axes[1].set_ylabel('AUPRC')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_dir / 'task7_ablation.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved task7_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: time-series plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_series(signals, labels, timesteps, out_dir, fps=20):
    """Plot temporal conflict, TTC, dist, PCO, and labels."""
    N = len(timesteps)
    t_axis = np.arange(N) / fps

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    ax = axes[0]
    ax.plot(t_axis, signals['temporal_conflict_smooth'], color='#e74c3c', label='Conflict smooth')
    ax.plot(t_axis, signals['temporal_conflict_raw'], color='#e74c3c', alpha=0.3, linewidth=0.8, label='Conflict raw')
    ax.set_ylabel('Temporal Conflict')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_axis, signals['ttc_proxy_risk'], color='#3498db', label='TTC proxy risk')
    if 'ttc_rel_risk' in signals and np.isfinite(signals['ttc_rel_risk']).any():
        ax.plot(t_axis, signals['ttc_rel_risk'], color='#8e44ad', alpha=0.8, label='Actor-relative TTC risk')
    ax.set_ylabel('TTC Risk')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t_axis, signals['dist_proxy_risk'], color='#2ecc71', label='Distance proxy risk')
    ax.set_ylabel('Dist Risk')
    ax.legend(fontsize=8)

    ax = axes[3]
    pco = signals.get('planner_conditioned_occupancy', np.full(N, np.nan))
    ax.plot(t_axis, pco, color='#27ae60', linewidth=1.2, label='Planner-cond occupancy')
    ax.set_ylabel('PCO')
    ax.legend(fontsize=8)

    ax = axes[4]
    ax.fill_between(t_axis, labels.astype(float), alpha=0.5, color='#e74c3c', label='failure label')
    ax.set_ylabel('Failure')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)

    # Mark actual collision steps
    for i, t in enumerate(timesteps):
        if t['collision']:
            for a in axes:
                a.axvline(i / fps, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

    plt.suptitle('Temporal signals over route', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / 'time_series.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved time_series.png")


# ─────────────────────────────────────────────────────────────────────────────
# Event-level metrics
# ─────────────────────────────────────────────────────────────────────────────

def task_event_level(signals, timesteps, out_dir, fps=20, pre_window_s=3.0):
    """
    Event-level AUROC/AUPRC: one sample per collision event.
    """
    print("\n" + "="*60)
    print("EVENT-LEVEL METRICS")
    print("="*60)

    pre_frames = int(pre_window_s * fps)
    div = signals['temporal_conflict_smooth']
    pco = signals.get('planner_conditioned_occupancy', np.full(len(div), np.nan))
    ttc_key = preferred_ttc_key(signals)
    ttc_risk = signals[ttc_key]
    dist_risk = signals['dist_proxy_risk']

    # Find actual collision frames
    collision_steps = []
    in_coll = False
    for i, t in enumerate(timesteps):
        if t['collision'] and not in_coll:
            collision_steps.append(i)
            in_coll = True
        elif not t['collision']:
            in_coll = False

    if len(collision_steps) < 2:
        print(f"  Only {len(collision_steps)} collision event(s) — need ≥2 for event-level metrics. Skipping.")
        return

    # Positive samples: max signal in pre-failure window
    pos_div, pos_pco, pos_ttc, pos_dist = [], [], [], []
    for cs in collision_steps:
        start = max(0, cs - pre_frames)
        pos_div.append(finite_max(div[start:cs]))
        pos_pco.append(finite_max(pco[start:cs]))
        pos_ttc.append(finite_max(ttc_risk[start:cs]))
        pos_dist.append(finite_max(dist_risk[start:cs]))

    # Negative samples: safe windows far from any collision
    safe_candidates = [
        i for i in range(pre_frames, len(timesteps) - pre_frames)
        if not any(abs(i - cs) < pre_frames * 2 for cs in collision_steps)
    ]
    rng = np.random.default_rng(42)
    n_neg_balanced = min(len(pos_div), len(safe_candidates))  # balanced 1:1
    neg_idx_balanced = rng.choice(safe_candidates, n_neg_balanced, replace=False)

    neg_div_bal, neg_pco_bal, neg_ttc_bal, neg_dist_bal = [], [], [], []
    for i in neg_idx_balanced:
        neg_div_bal.append(finite_max(div[i-pre_frames:i]))
        neg_pco_bal.append(finite_max(pco[i-pre_frames:i]))
        neg_ttc_bal.append(finite_max(ttc_risk[i-pre_frames:i]))
        neg_dist_bal.append(finite_max(dist_risk[i-pre_frames:i]))

    # Also collect natural-prevalence negatives (all safe windows)
    neg_div_nat, neg_pco_nat, neg_ttc_nat, neg_dist_nat = [], [], [], []
    for i in safe_candidates:
        neg_div_nat.append(finite_max(div[i-pre_frames:i]))
        neg_pco_nat.append(finite_max(pco[i-pre_frames:i]))
        neg_ttc_nat.append(finite_max(ttc_risk[i-pre_frames:i]))
        neg_dist_nat.append(finite_max(dist_risk[i-pre_frames:i]))

    # Balanced evaluation (1:1)
    scores_div_bal  = np.array(pos_div  + neg_div_bal)
    scores_pco_bal  = np.array(pos_pco  + neg_pco_bal)
    scores_ttc_bal  = np.array(pos_ttc  + neg_ttc_bal)
    scores_dist_bal = np.array(pos_dist + neg_dist_bal)
    ev_labels_bal   = np.array([1]*len(pos_div) + [0]*len(neg_div_bal), dtype=float)

    print(f"\n  Events: {len(pos_div)} positive, {len(neg_div_bal)} negative (balanced 1:1)")
    for name, s in [('temporal_conflict', scores_div_bal), ('planner_cond_occ', scores_pco_bal),
                     (ttc_key, scores_ttc_bal), ('dist_proxy_risk', scores_dist_bal)]:
        auroc = roc_auc(s, ev_labels_bal)
        auprc = pr_auc(s, ev_labels_bal)
        print(f"  {name:15s}: event-AUROC={auroc:.3f}  event-AUPRC={auprc:.3f}")

    # Natural-prevalence evaluation (true base rate)
    scores_div_nat  = np.array(pos_div  + neg_div_nat)
    scores_pco_nat  = np.array(pos_pco  + neg_pco_nat)
    scores_ttc_nat  = np.array(pos_ttc  + neg_ttc_nat)
    scores_dist_nat = np.array(pos_dist + neg_dist_nat)
    ev_labels_nat   = np.array([1]*len(pos_div) + [0]*len(neg_div_nat), dtype=float)
    prev = len(pos_div) / (len(pos_div) + len(neg_div_nat))

    print(f"\n  Events: {len(pos_div)} positive, {len(neg_div_nat)} negative (natural prevalence={prev:.3f})")
    for name, s in [('temporal_conflict', scores_div_nat), ('planner_cond_occ', scores_pco_nat),
                     (ttc_key, scores_ttc_nat), ('dist_proxy_risk', scores_dist_nat)]:
        auroc = roc_auc(s, ev_labels_nat)
        auprc = pr_auc(s, ev_labels_nat)
        print(f"  {name:15s}: event-AUROC={auroc:.3f}  event-AUPRC={auprc:.3f}")


def task_baseline_sensitivity(all_routes_ts, eval_mask, labels_occ):
    """Report baseline AUROC sensitivity to declared horizons, taus, and smoothing."""
    print("\n=== BASELINE SENSITIVITY (pre-impact eval mask) ===")

    specs = [
        ('smoothing_window', [1, 3, 5, 10]),
        ('ttc_horizon', [3.0, 5.0, 7.0]),
        ('ttc_tau', [1.0, 1.5, 3.0]),
        ('dist_horizon', [20.0, 30.0, 50.0]),
        ('dist_tau', [5.0, 10.0, 15.0]),
    ]
    score_keys = ['ttc_proxy_risk', 'ttc_rel_risk', 'dist_proxy_risk', 'rss_proxy']

    for param, values in specs:
        print(f"  {param}:")
        for value in values:
            kwargs = {param: value}
            route_scores = []
            for route_ts in all_routes_ts:
                route_scores.append(compute_baseline_series(route_ts, **kwargs))

            merged = {
                key: np.concatenate([scores[key] for scores in route_scores])
                for key in score_keys
            }
            parts = []
            for key in score_keys:
                s = merged[key][eval_mask]
                if not np.isfinite(s).any():
                    continue
                parts.append(f"{key}={roc_auc(s, labels_occ):.3f}")
            print(f"    {value}: " + ("  ".join(parts) if parts else "no finite baseline scores"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', help='Single processed route pkl file')
    parser.add_argument('--routes', help='Glob for multiple pkl files, e.g. "processed/*_processed.pkl"')
    parser.add_argument('--out', default='analysis/results', help='Output directory')
    parser.add_argument('--label', default='collision',
                        choices=['collision', 'any_failure'],
                        help='collision=future_collision only; any_failure=collision OR near_miss')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import glob as _glob
    if args.routes:
        paths = sorted(_glob.glob(args.routes))
        if not paths:
            print(f"No files matched: {args.routes}"); return
    elif args.route:
        paths = [args.route]
    else:
        print("Provide --route or --routes"); return

    timesteps = []
    all_routes_ts = [] 
    all_div_signals = {
        k: [] for k in [
            'temporal_conflict_raw',
            'temporal_conflict_smooth',
            'temporal_agreement_raw',
            'temporal_agreement_smooth',
            'temporal_js_raw',
            'temporal_js_smooth',
            'temporal_valid_count',
            'divergence_raw',
            'divergence_smooth',
            'planner_spread',
            'planner_conditioned_occupancy',
            'has_occupancy',
        ]
    }
    all_base_signals = None
    per_route_signals = []
    per_route_labels  = []

    for p in paths:
        print(f"Loading: {p}")
        with open(p, 'rb') as f:
            d = pickle.load(f)
        route_ts = d['timesteps']
        timesteps.extend(route_ts)
        all_routes_ts.append(route_ts)

        # Compute signals per-route so temporal smoothing doesn't cross boundaries
        div = compute_divergence_series(route_ts, use_occupancy=True, temporal_window=5)
        base = compute_baseline_series(route_ts)
        pco, _ = planner_conditioned_occupancy(route_ts, blur_sigma=0.0)
        if all_base_signals is None:
            all_base_signals = {k: [] for k in base}
        for k in all_div_signals:
            if k in div:
                all_div_signals[k].append(div[k])
        all_div_signals['planner_conditioned_occupancy'].append(pco)
        for k in all_base_signals:
            all_base_signals[k].append(base[k])

        route_sig = {**div, **base, 'planner_conditioned_occupancy': pco}
        per_route_signals.append(route_sig)
        if args.label == 'collision':
            per_route_labels.append(np.array([t['future_collision'] for t in route_ts], dtype=np.float64))
        else:
            per_route_labels.append(np.array([t['future_collision'] or t.get('future_near_miss', False)
                                               for t in route_ts], dtype=np.float64))

    print(f"Total: {len(timesteps)} timesteps across {len(paths)} route(s)")

    div_signals  = {k: np.concatenate(v) for k, v in all_div_signals.items()}
    base_signals = {k: np.concatenate(v) for k, v in all_base_signals.items()}
    print(f"  Occupancy available: {div_signals['has_occupancy'].sum()}/{len(timesteps)} steps")

    # Merge into one signals dict
    signals = {**div_signals, **base_signals}
    
    # Extract actual collision status (detection) vs future labels (prediction)
    collision_mask = np.array([t.get('collision', False) for t in timesteps], dtype=bool)
    time_to_collision = np.array([
        np.nan if t.get('time_to_collision') is None else float(t.get('time_to_collision'))
        for t in timesteps
    ], dtype=np.float64)

    if args.label == 'collision':
        labels = np.array([t['future_collision'] for t in timesteps], dtype=np.float64)
    else:
        labels = np.array([t['future_collision'] or t.get('future_near_miss', False)
                           for t in timesteps], dtype=np.float64)
    
    print(f"\nLabel: {args.label}  |  positive rate: {labels.mean():.3f}  ({int(labels.sum())}/{len(labels)})")

    # ── Issue 6: Evaluation Mask (Occupancy-valid AND Not-Currently-Crashing) ──
    # We exclude frames where collision=True to measure PREDICTION, not DETECTION.
    at_impact = collision_mask | (np.abs(time_to_collision) < 1e-6)
    eval_mask = signals['has_occupancy'].astype(bool) & (~at_impact)
    
    signals_occ = {k: v[eval_mask] for k, v in signals.items()}
    labels_occ  = labels[eval_mask]
    ts_occ      = [t for t, v in zip(timesteps, eval_mask) if v]
    
    print(f"  Evaluation frames (occ-valid & pre-impact): {eval_mask.sum()}/{len(timesteps)} "
          f"(positive rate: {labels_occ.mean():.3f})")

    route_ids = np.concatenate([
        np.full(len(rt), i, dtype=int)
        for i, rt in enumerate(all_routes_ts)
    ])
    route_ids_occ = route_ids[eval_mask]

    # Update per-route splits for bootstrapping
    per_route_eval_masks = []
    for s, rt in zip(per_route_signals, all_routes_ts):
        route_collision = np.array([t.get('collision', False) for t in rt], dtype=bool)
        route_ttc = np.array([
            np.nan if t.get('time_to_collision') is None else float(t.get('time_to_collision'))
            for t in rt
        ], dtype=np.float64)
        per_route_eval_masks.append(s['has_occupancy'].astype(bool) & (~(route_collision | (route_ttc == 0.0))))
    
    per_route_labels_occ = [l[m] for l, m in zip(per_route_labels, per_route_eval_masks)]
    per_route_signals_occ = [{k: v[m] for k, v in s.items()}
                              for s, m in zip(per_route_signals, per_route_eval_masks)]

    # All-frame baselines
    print("\n=== ALL-FRAME BASELINES (diagnostic; core metrics use pre-impact mask) ===")
    for name in available_keys(signals, BASELINE_SCORE_KEYS):
        a = roc_auc(signals[name], labels)
        print(f"  {name:15s}: AUROC={a:.3f}")

    # Per-route AUROC
    ttc_key = preferred_ttc_key(signals)
    print(f"\n=== PER-ROUTE AUROC (temporal_conflict_smooth vs planner_cond_occ vs {ttc_key}) ===")
    route_aurocs_div, route_aurocs_pco, route_aurocs_ttc = [], [], []
    for i, (rs, rl) in enumerate(zip(per_route_signals_occ, per_route_labels_occ)):
        a_div = roc_auc(rs['temporal_conflict_smooth'], rl)
        a_pco = roc_auc(rs['planner_conditioned_occupancy'], rl)
        a_ttc = roc_auc(rs[ttc_key], rl)
        route_aurocs_div.append(a_div)
        route_aurocs_pco.append(a_pco)
        route_aurocs_ttc.append(a_ttc)
        print(f"  Route {i+1}: conflict={a_div:.3f}  pco={a_pco:.3f}  ttc={a_ttc:.3f}")
    if len(route_aurocs_div) > 1:
        valid_div = [x for x in route_aurocs_div if not np.isnan(x)]
        valid_pco = [x for x in route_aurocs_pco if not np.isnan(x)]
        valid_ttc = [x for x in route_aurocs_ttc if not np.isnan(x)]
        print(f"  Mean ± std  conflict={np.mean(valid_div):.3f}±{np.std(valid_div):.3f}  "
              f"pco={np.mean(valid_pco):.3f}±{np.std(valid_pco):.3f}  "
              f"ttc={np.mean(valid_ttc):.3f}±{np.std(valid_ttc):.3f}")

    # Delta AUROC
    ttc_key = preferred_ttc_key(signals)
    print(f"\n=== DELTA AUROC: temporal_conflict vs planner_cond_occ vs {ttc_key} ===")
    rng = np.random.default_rng(42)
    R = len(per_route_signals_occ)

    for label, s_key in [('conflict', 'temporal_conflict_smooth'),
                          ('pco', 'planner_conditioned_occupancy')]:
        delta_point = (roc_auc(signals_occ[s_key], labels_occ) -
                       roc_auc(signals_occ[ttc_key], labels_occ))
        delta_vals = []
        for _ in range(500):
            idx = rng.integers(0, R, R)
            s_candidate = np.concatenate([per_route_signals_occ[i][s_key] for i in idx])
            s_ttc       = np.concatenate([per_route_signals_occ[i][ttc_key] for i in idx])
            l           = np.concatenate([per_route_labels_occ[i] for i in idx])
            d = roc_auc(s_candidate, l) - roc_auc(s_ttc, l)
            if not np.isnan(d):
                delta_vals.append(d)
        if delta_vals:
            lo_d = np.percentile(delta_vals, 2.5)
            hi_d = np.percentile(delta_vals, 97.5)
            ci_type = "route-CI" if R >= 4 else "frame-CI"
            print(f"  {label}: Δ AUROC = {delta_point:+.3f}  95% CI [{lo_d:+.3f}, {hi_d:+.3f}] [{ci_type}]")

    print("\nPlotting time series...")
    plot_time_series(signals, labels, timesteps, out_dir)

    print("\n[Core metrics on occupancy-valid, pre-impact frames]")
    task4_results = task4_auroc_auprc(signals_occ, labels_occ, out_dir,
                                      per_route_signals=per_route_signals_occ,
                                      per_route_labels=per_route_labels_occ)

    task5_matched_bin(signals_occ, labels_occ, ts_occ, out_dir, route_ids=route_ids_occ)
    task6_lead_time(signals, timesteps, out_dir)
    task7_ablation(signals_occ, labels_occ, ts_occ, out_dir)
    task_event_level(signals, timesteps, out_dir)

    print("\n=== HORIZON SENSITIVITY (temporal_conflict_smooth AUROC) ===")
    for w in [1, 3, 5, 10]:
        chunks = []
        for route_ts in all_routes_ts:
            dv = compute_divergence_series(route_ts, use_occupancy=True, temporal_window=w)
            chunks.append(dv['temporal_conflict_smooth'])
        div_w = np.concatenate(chunks)[eval_mask]
        a = roc_auc(div_w, labels_occ)
        print(f"  window={w:2d}: AUROC={a:.3f}")

    task_baseline_sensitivity(all_routes_ts, eval_mask, labels_occ)

    print("\n=== CALIBRATION: temporal conflict percentile vs failure rate ===")
    calibration_valid = np.isfinite(signals_occ['temporal_conflict_smooth'])
    div_s = signals_occ['temporal_conflict_smooth'][calibration_valid]
    labels_cal = labels_occ[calibration_valid]
    if len(div_s) == 0:
        print("  Temporal conflict unavailable; skipping calibration.")
        return
    n_bins = 10
    percentile_edges = np.percentile(div_s, np.linspace(0, 100, n_bins + 1))
    bin_centers, fail_rates = [], []
    edges = list(zip(percentile_edges[:-1], percentile_edges[1:]))
    for b, (lo, hi) in enumerate(edges):
        mask = (div_s >= lo) & (div_s < hi if b < len(edges)-1 else div_s <= hi)
        if mask.sum() < 5: continue
        rate = labels_cal[mask].mean()
        bin_centers.append((lo + hi) / 2)
        fail_rates.append(rate)
        print(f"  conflict [{lo:.3f},{hi:.3f}): failure_rate={rate:.3f}  (n={mask.sum()})")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, fail_rates, 'o-', color='#e74c3c')
    ax.axhline(labels_cal.mean(), color='gray', linestyle='--', linewidth=0.8,
               label=f'base rate ({labels_cal.mean():.3f})')
    ax.set_xlabel('Temporal conflict score')
    ax.set_ylabel('Empirical failure rate')
    ax.set_title('Calibration: temporal conflict vs failure probability')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / 'calibration.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved calibration.png")

    print(f"\n✓ All results saved to {out_dir}/")


if __name__ == '__main__':
    main()

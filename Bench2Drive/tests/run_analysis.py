"""
Month 2 Divergence Analysis
============================
Tasks:
  4. AUROC / AUPRC: divergence vs baselines for early failure prediction
  5. Matched-bin separation: divergence within TTC/dist/speed bins
  6. Lead-time analysis: divergence rise time vs TTC trigger time
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

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.divergence import compute_divergence_series, planner_spread_entropy, rasterize_planner, js_divergence
from analysis.baselines import compute_baseline_series


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def roc_auc(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float('nan')
    return float(roc_auc_score(labels, scores))


def pr_auc(scores, labels):
    if labels.sum() == 0:
        return float('nan')
    return float(average_precision_score(labels, scores))


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
    Compare divergence vs baselines on early failure prediction.
    Uses route-level bootstrap when per_route_signals is provided.
    """
    print("\n" + "="*60)
    print("TASK 4: AUROC / AUPRC")
    print("="*60)

    candidates = {
        'divergence_smooth': signals['divergence_smooth'],
        'divergence_raw':    signals['divergence_raw'],
        'planner_spread':    signals['planner_spread'],
        'ttc_inv':           signals['ttc_inv'],
        'dist_inv':          signals['dist_inv'],
        'rss':               signals['rss'],
    }

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

    colors = ['#e74c3c' if 'divergence' in n else '#3498db' for n in names]
    ax = axes[0]
    bars = ax.bar(names, aurocs, color=colors, yerr=auroc_errs, capsize=4)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='random')
    ax.set_ylim(0, 1.05)
    ax.set_title('AUROC: Early Failure Prediction')
    ax.set_ylabel('AUROC')
    ax.tick_params(axis='x', rotation=30)
    ax.legend()

    ax = axes[1]
    baseline_rate = labels.mean()
    ax.bar(names, auprcs, color=colors, yerr=auprc_errs, capsize=4)
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

def task5_matched_bin(signals, labels, timesteps, out_dir):
    """
    Anti-TTC-collapse test: within each TTC/dist/speed bin,
    does divergence still separate failure from safe?
    """
    print("\n" + "="*60)
    print("TASK 5: MATCHED-BIN SEPARATION")
    print("="*60)

    div = signals['divergence_smooth']
    ttc_inv = signals['ttc_inv']
    dist_inv = signals['dist_inv']
    speed = signals['speed']

    # Define binning variables and their edges
    bin_configs = [
        ('speed',    speed,    np.array([0, 2, 5, 10, 30])),
        ('dist_inv', dist_inv, np.percentile(dist_inv, [0, 25, 50, 75, 100])),
        ('ttc_inv',  ttc_inv,  np.percentile(ttc_inv,  [0, 25, 50, 75, 100])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (bin_name, bin_var, bin_edges) in zip(axes, bin_configs):
        bin_edges = np.unique(bin_edges)
        separations = []
        bin_labels = []
        n_bins = len(bin_edges) - 1

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
        ax.set_title(f'Divergence separation within {bin_name} bins')
        ax.set_ylabel('mean(div|fail) − mean(div|safe)')

    plt.tight_layout()
    fig.savefig(out_dir / 'task5_matched_bin.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved task5_matched_bin.png")


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: Lead-time analysis
# ─────────────────────────────────────────────────────────────────────────────

def task6_lead_time(signals, timesteps, out_dir, fps=20,
                    div_threshold_pct=75, ttc_threshold=2.0, sustain=3):
    """
    For each failure event, measure:
      - t_div: first time divergence exceeds threshold before failure
      - t_ttc: first time TTC < ttc_threshold before failure
    Lead time = time_to_failure - t_signal.
    """
    print("\n" + "="*60)
    print("TASK 6: LEAD-TIME ANALYSIS")
    print("="*60)

    div = signals['divergence_smooth']
    ttc_raw = signals['ttc_raw']

    div_thresh = np.percentile(div, div_threshold_pct)
    print(f"  Divergence threshold (p{div_threshold_pct}): {div_thresh:.4f}")
    print(f"  TTC threshold: {ttc_threshold:.1f}s")

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
    ttc_leads = []

    for collision_step in collision_steps:
        search_start = max(0, collision_step - 120)  # look back up to 6s

        # Divergence: first causal sustained activation (sustain frames all above threshold)
        div_trigger = None
        for j in range(search_start, collision_step - sustain + 1):
            if np.all(div[j:j+sustain] > div_thresh):
                div_trigger = j
                break
        if div_trigger is not None:
            div_leads.append((collision_step - div_trigger) / fps)

        # TTC: first causal sustained activation
        ttc_trigger = None
        for j in range(search_start, collision_step - sustain + 1):
            window = ttc_raw[j:j+sustain]
            if np.all((window < ttc_threshold) & (window < 999.0)):
                ttc_trigger = j
                break
        if ttc_trigger is not None:
            ttc_leads.append((collision_step - ttc_trigger) / fps)

        div_str = f"{div_leads[-1]:.2f}s" if div_leads else "None"
        ttc_str = f"{ttc_leads[-1]:.2f}s" if ttc_leads else "None"
        print(f"  Collision at step {collision_step}: div_lead={div_str}, ttc_lead={ttc_str}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if div_leads:
        ax.hist(div_leads, bins=10, alpha=0.7, color='#e74c3c', label=f'Divergence (n={len(div_leads)})')
        print(f"  Divergence lead times: mean={np.mean(div_leads):.2f}s, "
              f"median={np.median(div_leads):.2f}s")
    if ttc_leads:
        ax.hist(ttc_leads, bins=10, alpha=0.7, color='#3498db', label=f'TTC (n={len(ttc_leads)})')
        print(f"  TTC lead times: mean={np.mean(ttc_leads):.2f}s, "
              f"median={np.median(ttc_leads):.2f}s")
    ax.set_xlabel('Lead time before failure (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Lead-time distribution: Divergence vs TTC')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / 'task6_lead_time.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved task6_lead_time.png")

    return {'div_leads': div_leads, 'ttc_leads': ttc_leads}


# ─────────────────────────────────────────────────────────────────────────────
# Task 7: Ablation
# ─────────────────────────────────────────────────────────────────────────────

def _single_mode_divergence(timesteps):
    """Ablation: use only the top-scoring trajectory (no multi-modal sampling)."""
    N = len(timesteps)
    div = np.zeros(N)
    for i, t in enumerate(timesteps):
        trajs = t['planner_trajs']   # (K, T, 2)
        scores = t['planner_scores'] # (K,)
        occ = t['occupancy_grid']

        # Use only top-1 mode
        top1 = np.argmax(scores)
        trajs_1 = trajs[top1:top1+1]
        scores_1 = np.array([1.0])

        if occ.max() > 1e-6:
            p = rasterize_planner(trajs_1, scores_1)
            q = occ / (occ.sum() + 1e-10)
            div[i] = js_divergence(p, q)
        else:
            div[i] = 0.0  # no occupancy: undefined, not degenerate spread
    return div


def _no_temporal_smoothing(divergence_raw):
    """Ablation: no temporal aggregation (use raw D(t))."""
    return divergence_raw.copy()


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

    full_div = signals['divergence_smooth']
    full_auroc = roc_auc(full_div, labels)
    full_auprc = pr_auc(full_div, labels)
    print(f"  Full divergence (smooth, multi-modal): "
          f"AUROC={full_auroc:.3f}, AUPRC={full_auprc:.3f}")

    # Ablation A: single mode
    div_single = _single_mode_divergence(timesteps)
    auroc_a = roc_auc(div_single, labels)
    auprc_a = pr_auc(div_single, labels)
    print(f"  Ablation A (top-1 mode only):          "
          f"AUROC={auroc_a:.3f}, AUPRC={auprc_a:.3f}")

    # Ablation B: no temporal smoothing
    div_raw = signals['divergence_raw']
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
    """Plot D(t), TTC_inv, dist_inv, and failure labels over time."""
    N = len(timesteps)
    t_axis = np.arange(N) / fps

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(t_axis, signals['divergence_smooth'], color='#e74c3c', label='D(t) smooth')
    ax.plot(t_axis, signals['divergence_raw'], color='#e74c3c', alpha=0.3, linewidth=0.8, label='D(t) raw')
    ax.set_ylabel('Divergence D(t)')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_axis, signals['ttc_inv'], color='#3498db', label='1/TTC')
    ax.set_ylabel('1/TTC')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t_axis, signals['dist_inv'], color='#2ecc71', label='1/dist')
    ax.set_ylabel('1/dist')
    ax.legend(fontsize=8)

    ax = axes[3]
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
# Event-level metrics (Issue 5)
# ─────────────────────────────────────────────────────────────────────────────

def task_event_level(signals, timesteps, out_dir, fps=20, pre_window_s=3.0):
    """
    Event-level AUROC/AUPRC: one sample per collision event.
    For each collision, take max divergence / min TTC in the pre-failure window.
    For each safe window (same length, randomly sampled), take the same.
    This avoids dense-window inflation.
    """
    print("\n" + "="*60)
    print("EVENT-LEVEL METRICS")
    print("="*60)

    pre_frames = int(pre_window_s * fps)
    div = signals['divergence_smooth']
    ttc_inv = signals['ttc_inv']
    dist_inv = signals['dist_inv']

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
    pos_div, pos_ttc, pos_dist = [], [], []
    for cs in collision_steps:
        start = max(0, cs - pre_frames)
        pos_div.append(div[start:cs].max())
        pos_ttc.append(ttc_inv[start:cs].max())
        pos_dist.append(dist_inv[start:cs].max())

    # Negative samples: balanced (1:1), safe windows far from any collision
    safe_candidates = [
        i for i in range(pre_frames, len(timesteps) - pre_frames)
        if not any(abs(i - cs) < pre_frames * 2 for cs in collision_steps)
    ]
    rng = np.random.default_rng(42)
    n_neg = min(len(pos_div), len(safe_candidates))  # balanced 1:1
    neg_idx = rng.choice(safe_candidates, n_neg, replace=False)

    neg_div, neg_ttc, neg_dist = [], [], []
    for i in neg_idx:
        neg_div.append(div[i-pre_frames:i].max())
        neg_ttc.append(ttc_inv[i-pre_frames:i].max())
        neg_dist.append(dist_inv[i-pre_frames:i].max())

    scores_div  = np.array(pos_div  + neg_div)
    scores_ttc  = np.array(pos_ttc  + neg_ttc)
    scores_dist = np.array(pos_dist + neg_dist)
    ev_labels   = np.array([1]*len(pos_div) + [0]*len(neg_div), dtype=float)

    print(f"  Events: {len(pos_div)} positive, {len(neg_div)} negative (balanced 1:1)")
    for name, s in [('divergence', scores_div), ('ttc_inv', scores_ttc), ('dist_inv', scores_dist)]:
        auroc = roc_auc(s, ev_labels)
        auprc = pr_auc(s, ev_labels)
        print(f"  {name:15s}: event-AUROC={auroc:.3f}  event-AUPRC={auprc:.3f}")


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
    all_routes_ts = []  # store per-route timesteps for horizon sensitivity
    all_div_signals = {k: [] for k in ['divergence_raw', 'divergence_smooth', 'planner_spread', 'has_occupancy']}
    all_base_signals = {k: [] for k in ['ttc_raw', 'dist_raw', 'ttc_inv', 'dist_inv', 'rss', 'speed']}
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
        for k in all_div_signals:
            all_div_signals[k].append(div[k])
        for k in all_base_signals:
            all_base_signals[k].append(base[k])

        route_sig = {**div, **base}
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
    if args.label == 'collision':
        labels = np.array([t['future_collision'] for t in timesteps], dtype=np.float64)
    else:
        labels = np.array([t['future_collision'] or t.get('future_near_miss', False)
                           for t in timesteps], dtype=np.float64)
    print(f"\nLabel: {args.label}  |  positive rate: {labels.mean():.3f}  ({int(labels.sum())}/{len(labels)})")

    # ── Issue 6: Occupancy-valid mask (core paper metrics use this) ──────────
    occ_valid = signals['has_occupancy'].astype(bool)
    signals_occ = {k: v[occ_valid] for k, v in signals.items()}
    labels_occ  = labels[occ_valid]
    ts_occ      = [t for t, v in zip(timesteps, occ_valid) if v]
    print(f"  Occupancy-valid frames: {occ_valid.sum()}/{len(timesteps)} "
          f"(positive rate: {labels_occ.mean():.3f})")
    per_route_labels_occ = [l[s['has_occupancy'].astype(bool)]
                             for s, l in zip(per_route_signals, per_route_labels)]
    per_route_signals_occ = [{k: v[s['has_occupancy'].astype(bool)] for k, v in s.items()}
                              for s in per_route_signals]

    # ── Issue 2: All-frame baselines (supplemental) ──────────────────────────
    print("\n=== ALL-FRAME BASELINES (supplemental, no occupancy mask) ===")
    for name in ['ttc_inv', 'dist_inv', 'rss']:
        a = roc_auc(signals[name], labels)
        print(f"  {name:15s}: AUROC={a:.3f}")

    # ── Issue 5: Per-route AUROC reporting ───────────────────────────────────
    print("\n=== PER-ROUTE AUROC (divergence_smooth vs ttc_inv) ===")
    route_aurocs_div, route_aurocs_ttc = [], []
    for i, (rs, rl) in enumerate(zip(per_route_signals, per_route_labels)):
        a_div = roc_auc(rs['divergence_smooth'], rl)
        a_ttc = roc_auc(rs['ttc_inv'], rl)
        route_aurocs_div.append(a_div)
        route_aurocs_ttc.append(a_ttc)
        print(f"  Route {i+1}: div={a_div:.3f}  ttc={a_ttc:.3f}")
    if len(route_aurocs_div) > 1:
        valid_div = [x for x in route_aurocs_div if not np.isnan(x)]
        valid_ttc = [x for x in route_aurocs_ttc if not np.isnan(x)]
        print(f"  Mean ± std  div={np.mean(valid_div):.3f}±{np.std(valid_div):.3f}  "
              f"ttc={np.mean(valid_ttc):.3f}±{np.std(valid_ttc):.3f}")

    # ── Issue 7: Delta AUROC with route bootstrap CI ─────────────────────────
    print("\n=== DELTA AUROC: divergence_smooth − ttc_inv ===")
    def delta_auroc(scores_pair, labels):
        # scores_pair is concatenation of [div, ttc] — split by half
        half = len(scores_pair) // 2
        return roc_auc(scores_pair[:half], labels[:half]) - roc_auc(scores_pair[half:], labels[half:])

    pr_div = [r['divergence_smooth'] for r in per_route_signals_occ]
    pr_ttc = [r['ttc_inv']           for r in per_route_signals_occ]
    pr_lbl = per_route_labels_occ
    delta_point = (roc_auc(signals_occ['divergence_smooth'], labels_occ) -
                   roc_auc(signals_occ['ttc_inv'], labels_occ))
    # Route bootstrap on delta
    rng = np.random.default_rng(42)
    R = len(pr_div)
    delta_vals = []
    for _ in range(500):
        idx = rng.integers(0, R, R)
        s_div = np.concatenate([pr_div[i] for i in idx])
        s_ttc = np.concatenate([pr_ttc[i] for i in idx])
        l     = np.concatenate([pr_lbl[i] for i in idx])
        d = roc_auc(s_div, l) - roc_auc(s_ttc, l)
        if not np.isnan(d):
            delta_vals.append(d)
    if delta_vals:
        lo_d = np.percentile(delta_vals, 2.5)
        hi_d = np.percentile(delta_vals, 97.5)
        ci_type = "route-CI" if R >= 4 else "frame-CI"
        print(f"  Δ AUROC = {delta_point:+.3f}  95% CI [{lo_d:+.3f}, {hi_d:+.3f}] [{ci_type}]")
    else:
        print(f"  Δ AUROC = {delta_point:+.3f}  (CI unavailable)")

    # ── Diagnostic time-series ───────────────────────────────────────────────
    print("\nPlotting time series...")
    plot_time_series(signals, labels, timesteps, out_dir)

    # ── Task 4: AUROC / AUPRC (occupancy-valid frames) ───────────────────────
    print("\n[Core metrics on occupancy-valid frames]")
    task4_results = task4_auroc_auprc(signals_occ, labels_occ, out_dir,
                                      per_route_signals=per_route_signals_occ,
                                      per_route_labels=per_route_labels_occ)

    # ── Task 5: Matched-bin separation ───────────────────────────────────────
    task5_matched_bin(signals_occ, labels_occ, ts_occ, out_dir)

    # ── Task 6: Lead-time analysis ───────────────────────────────────────────
    task6_lead_time(signals, timesteps, out_dir)

    # ── Task 7: Ablation ─────────────────────────────────────────────────────
    task7_ablation(signals, labels, timesteps, out_dir)

    # ── Event-level metrics ───────────────────────────────────────────────────
    task_event_level(signals_occ, ts_occ, out_dir)

    # ── Horizon sensitivity (issue 1 fix: use stored all_routes_ts) ──────────
    print("\n=== HORIZON SENSITIVITY (divergence_smooth AUROC) ===")
    for w in [1, 3, 5, 10]:
        chunks = []
        for route_ts in all_routes_ts:
            dv = compute_divergence_series(route_ts, use_occupancy=True, temporal_window=w)
            chunks.append(dv['divergence_smooth'])
        div_w = np.concatenate(chunks)[occ_valid]
        a = roc_auc(div_w, labels_occ)
        print(f"  window={w:2d}: AUROC={a:.3f}")

    # ── Calibration (issue 4 fix: use < not <= for bin edges) ────────────────
    print("\n=== CALIBRATION: divergence percentile vs failure rate ===")
    div_s = signals_occ['divergence_smooth']
    n_bins = 10
    percentile_edges = np.percentile(div_s, np.linspace(0, 100, n_bins + 1))
    bin_centers, fail_rates = [], []
    edges = list(zip(percentile_edges[:-1], percentile_edges[1:]))
    for b, (lo, hi) in enumerate(edges):
        # Issue 4: use < hi except for final bin
        mask = (div_s >= lo) & (div_s < hi if b < len(edges)-1 else div_s <= hi)
        if mask.sum() < 5:
            continue
        rate = labels_occ[mask].mean()
        bin_centers.append((lo + hi) / 2)
        fail_rates.append(rate)
        print(f"  D(t) [{lo:.3f},{hi:.3f}): failure_rate={rate:.3f}  (n={mask.sum()})")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, fail_rates, 'o-', color='#e74c3c')
    ax.axhline(labels_occ.mean(), color='gray', linestyle='--', linewidth=0.8,
               label=f'base rate ({labels_occ.mean():.3f})')
    ax.set_xlabel('Divergence D(t)')
    ax.set_ylabel('Empirical failure rate')
    ax.set_title('Calibration: D(t) vs failure probability')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / 'calibration.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved calibration.png")

    print(f"\n✓ All results saved to {out_dir}/")


if __name__ == '__main__':
    main()

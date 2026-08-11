#!/usr/bin/env python3
"""Render the same-state current-versus-temporal occupancy snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle


matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def footprint_polygon(point, yaw, length, width):
    forward = np.array([np.sin(yaw), np.cos(yaw)])
    right = np.array([np.cos(yaw), -np.sin(yaw)])
    return np.asarray([
        point + forward * length / 2 + right * width / 2,
        point + forward * length / 2 - right * width / 2,
        point - forward * length / 2 - right * width / 2,
        point - forward * length / 2 + right * width / 2,
    ])


def waypoint_yaw(trajectory, index):
    if index == 0:
        return 0.0
    if index < len(trajectory) - 1:
        delta = trajectory[index + 1] - trajectory[index]
    else:
        delta = trajectory[index] - trajectory[index - 1]
    if np.linalg.norm(delta) < 1e-6:
        return 0.0
    return float(np.arctan2(delta[0], delta[1]))


def draw_occupancy_grid(ax, grid, extent=(-30, 30, -30, 30)):
    """Draw binary occupancy as vector row runs instead of a PDF bitmap."""
    occupancy = np.asarray(grid.T, dtype=bool)
    x_min, x_max, y_min, y_max = extent
    cell_width = (x_max - x_min) / occupancy.shape[1]
    cell_height = (y_max - y_min) / occupancy.shape[0]
    patches = []
    for row_index, row in enumerate(occupancy):
        padded = np.pad(row.astype(np.int8), (1, 1))
        transitions = np.flatnonzero(np.diff(padded))
        for start, end in transitions.reshape(-1, 2):
            patches.append(Rectangle(
                (x_min + start * cell_width,
                 y_min + row_index * cell_height),
                (end - start) * cell_width,
                cell_height,
            ))
    if patches:
        ax.add_collection(PatchCollection(
            patches, facecolor="black", edgecolor="none", zorder=1,
        ))
    ax.set_facecolor("white")


def validate_snapshot(data):
    metadata = json.loads(str(data["metadata_json"]))
    schema_version = metadata.get("schema_version")
    if schema_version not in (2, 3):
        raise ValueError("Unsupported qualitative snapshot schema")
    if not metadata.get("development_only"):
        raise ValueError("Snapshot is not marked development-only")
    if metadata.get("outcome_eligible") is not False:
        raise ValueError("Snapshot must be excluded from outcome estimates")
    if data["current_grids"].shape != (6, 120, 120):
        raise ValueError("Invalid current-grid shape")
    if data["temporal_grids"].shape != (6, 120, 120):
        raise ValueError("Invalid temporal-grid shape")
    if not np.array_equal(
        data["current_grids"],
        np.repeat(data["current_grids"][0:1], 6, axis=0),
    ):
        raise ValueError("Current grids are not repeated identically")
    np.testing.assert_allclose(data["current_horizons"], np.zeros(6))
    np.testing.assert_allclose(
        data["temporal_horizons"], np.arange(0.5, 3.01, 0.5)
    )
    if data["camera_images"].ndim != 4:
        raise ValueError("Camera images must be a four-dimensional array")
    if data["camera_images"].shape[0] != 6:
        raise ValueError("Qualitative snapshot must contain six cameras")
    if data["camera_images"].shape[-1] != 3:
        raise ValueError("Camera images must contain RGB pixels")
    expected_cameras = {
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    }
    if set(data["camera_ids"].tolist()) != expected_cameras:
        raise ValueError("Unexpected camera identifiers")
    if schema_version == 3:
        if not metadata.get("front_camera_sequence_complete"):
            raise ValueError("Future front-camera sequence is incomplete")
        if data["front_camera_images"].shape != (
            6,
            data["camera_images"].shape[1],
            data["camera_images"].shape[2],
            3,
        ):
            raise ValueError("Invalid future front-camera sequence shape")
        np.testing.assert_allclose(
            data["front_camera_horizons"], np.arange(0.5, 3.01, 0.5)
        )
        np.testing.assert_array_equal(
            data["front_camera_frames"], data["front_camera_target_frames"]
        )
    return metadata


def draw_camera_ring(fig, rect, camera_by_id):
    """Draw the SparseDrive-style rear/ego/front six-camera ring."""
    top_ids = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
    bottom_ids = ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
    gap_x = 0.004 * rect[2]
    gap_y = 0.035 * rect[3]
    panel_w = (rect[2] - 2 * gap_x) / 3
    panel_h = (rect[3] - gap_y) / 2

    for row, camera_ids in enumerate((top_ids, bottom_ids)):
        for col, camera_id in enumerate(camera_ids):
            x = rect[0] + col * (panel_w + gap_x)
            y = rect[1] + (1 - row) * (panel_h + gap_y)
            ax = fig.add_axes([x, y, panel_w, panel_h])
            image = camera_by_id[camera_id]
            artist = ax.imshow(image, aspect="auto", interpolation="antialiased")
            if row == 0:
                vertices = (
                    [(0.02, 0.08), (0.98, 0.00), (0.98, 1.00), (0.02, 0.86)]
                    if col == 0 else
                    [(0.02, 0.00), (0.98, 0.08), (0.98, 0.86), (0.02, 1.00)]
                    if col == 2 else
                    [(0.01, 0.00), (0.99, 0.00), (0.99, 1.00), (0.01, 1.00)]
                )
            else:
                vertices = (
                    [(0.02, 0.14), (0.98, 0.00), (0.98, 1.00), (0.02, 0.92)]
                    if col == 0 else
                    [(0.02, 0.00), (0.98, 0.14), (0.98, 0.92), (0.02, 1.00)]
                    if col == 2 else
                    [(0.01, 0.00), (0.99, 0.00), (0.99, 1.00), (0.01, 1.00)]
                )
            clip = Polygon(
                vertices, closed=True, transform=ax.transAxes,
                facecolor="none", edgecolor="white", linewidth=0.75,
            )
            artist.set_clip_path(clip)
            ax.add_patch(clip)
            ax.set_axis_off()

def draw_occupancy_axes(
    axes, current, temporal, original, length, width,
    current_horizons, temporal_horizons, *, show_planner=True,
):
    grid_extent = [-30, 30, -30, 30]
    x_min = max(-14.0, float(np.min(original[:, 0]) - 8.0))
    x_max = min(14.0, float(np.max(original[:, 0]) + 8.0))
    y_min = max(-5.0, float(np.min(original[:, 1]) - 5.0))
    y_max = min(32.0, float(np.max(original[:, 1]) + 9.0))

    for row, (grids, horizons, row_label) in enumerate((
        (current, current_horizons, "Current frame"),
        (temporal, temporal_horizons, "Temporal"),
    )):
        for index, ax in enumerate(axes[row]):
            draw_occupancy_grid(ax, grids[index], grid_extent)
            if show_planner:
                ax.plot(
                    original[:, 0], original[:, 1],
                    color="#1f6eb3", linewidth=1.15, marker="o",
                    markersize=1.8, zorder=3,
                )
                polygon = footprint_polygon(
                    original[index], waypoint_yaw(original, index),
                    length, width,
                )
                ax.add_patch(Polygon(
                    polygon, closed=True, facecolor="#4c9bd6",
                    edgecolor="#0d4f86", alpha=0.35,
                    linewidth=0.55, zorder=4,
                ))
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if index == 0:
                ax.set_ylabel(row_label, fontsize=7, labelpad=5)
            if row == 1:
                ax.set_xlabel(
                    "$t+{:.1f}$ s".format(float(temporal_horizons[index])),
                    fontsize=7, labelpad=3,
                )


def representative_proposal_indices(
    proposals, planner_mask, original_mode, count=7,
):
    """Choose a few distinct forward proposals for the overview icon."""
    finite = np.isfinite(proposals).all(axis=(1, 2))
    original_forward = float(proposals[original_mode, -1, 1])
    minimum_forward = max(2.0, 0.55 * original_forward)
    eligible = np.flatnonzero(
        finite & ~planner_mask & (proposals[:, -1, 1] >= minimum_forward)
    )
    eligible = eligible[eligible != original_mode]
    if eligible.size == 0:
        return []

    lateral = proposals[eligible, -1, 0]
    targets = np.linspace(
        float(np.quantile(lateral, 0.05)),
        float(np.quantile(lateral, 0.95)),
        count,
    )
    chosen = []
    references = [proposals[original_mode]]
    for target in targets:
        for offset in np.argsort(np.abs(lateral - target)):
            index = int(eligible[offset])
            candidate = proposals[index]
            if all(
                float(np.mean(np.linalg.norm(candidate - ref, axis=1))) > 0.45
                for ref in references
            ):
                chosen.append(index)
                references.append(candidate)
                break
    return chosen


def render_planner_output_icon(
    proposals, planner_mask, original_mode, vehicle_length, vehicle_width,
    outdir,
):
    """Render a minimal, vector planner-output glyph for the overview."""
    alternative_indices = representative_proposal_indices(
        proposals, planner_mask, original_mode,
    )
    selected = proposals[original_mode]
    visible = [selected] + [proposals[index] for index in alternative_indices]
    visible_points = np.concatenate(visible, axis=0)

    x_pad = 1.6
    x_limit = max(
        5.0,
        abs(float(np.min(visible_points[:, 0]))) + x_pad,
        abs(float(np.max(visible_points[:, 0]))) + x_pad,
    )
    y_min = -vehicle_length / 2.0 - 0.8
    y_max = max(8.0, float(np.max(visible_points[:, 1])) + 1.4)

    fig, ax = plt.subplots(figsize=(1.55, 1.70))
    origin = np.zeros((1, 2), dtype=np.float32)
    for index in alternative_indices:
        trajectory = np.concatenate([origin, proposals[index]], axis=0)
        ax.plot(
            trajectory[:, 0], trajectory[:, 1],
            color="#4c9bd6", alpha=0.28, linewidth=1.15,
            solid_capstyle="round", zorder=2,
        )

    selected_with_origin = np.concatenate([origin, selected], axis=0)
    ax.plot(
        selected_with_origin[:, 0], selected_with_origin[:, 1],
        color="#1f6eb3", alpha=1.0, linewidth=2.0,
        solid_capstyle="round", zorder=4,
    )
    ax.add_patch(Rectangle(
        (-vehicle_width / 2.0, -vehicle_length / 2.0),
        vehicle_width, vehicle_length,
        facecolor="#4c9bd6", edgecolor="#0d4f86",
        alpha=0.35, linewidth=0.8, zorder=5,
    ))
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    for extension in ("pdf", "png", "svg"):
        fig.savefig(
            outdir / ("fig01_planner_output." + extension),
            dpi=300, bbox_inches="tight", pad_inches=0.01,
        )
    plt.close(fig)


def render(snapshot: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with np.load(snapshot, allow_pickle=False) as data:
        metadata = validate_snapshot(data)
        current = data["current_grids"]
        temporal = data["temporal_grids"]
        original = data["proposals"][int(data["original_mode"])]
        length = float(data["vehicle_length"])
        width = float(data["vehicle_width"])
        current_horizons = data["current_horizons"]
        temporal_horizons = data["temporal_horizons"]
        camera_ids = data["camera_ids"].tolist()
        camera_images = data["camera_images"]
        front_camera_images = (
            data["front_camera_images"]
            if "front_camera_images" in data.files
            else None
        )
        proposals = data["proposals"]
        planner_mask = data["planner_mask"]
        original_mode = int(data["original_mode"])

    camera_by_id = {
        camera_id: camera_images[index]
        for index, camera_id in enumerate(camera_ids)
    }
    render_planner_output_icon(
        proposals, planner_mask, original_mode, length, width, outdir,
    )

    fig, axes = plt.subplots(
        2, 6, figsize=(7.15, 2.75), sharex=True, sharey=True,
        constrained_layout=True,
    )
    draw_occupancy_axes(
        axes, current, temporal, original, length, width,
        current_horizons, temporal_horizons,
    )

    for extension in ("pdf", "png", "svg"):
        fig.savefig(
            outdir / ("fig02_occupancy_case." + extension),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)

    grids_only, grids_only_axes = plt.subplots(
        2, 6, figsize=(7.15, 2.75), sharex=True, sharey=True,
        constrained_layout=True,
    )
    draw_occupancy_axes(
        grids_only_axes, current, temporal, original, length, width,
        current_horizons, temporal_horizons, show_planner=False,
    )
    for extension in ("pdf", "png", "svg"):
        grids_only.savefig(
            outdir / ("fig01_occupancy_grids." + extension),
            dpi=300, bbox_inches="tight", pad_inches=0.02,
        )
    plt.close(grids_only)

    ring = plt.figure(figsize=(3.2, 1.55))
    draw_camera_ring(ring, [0.01, 0.01, 0.98, 0.98], camera_by_id)
    for extension in ("pdf", "png", "svg"):
        ring.savefig(
            outdir / ("fig02_camera_ring." + extension),
            dpi=300, bbox_inches="tight", pad_inches=0.01,
            transparent=True,
        )
    plt.close(ring)

    if front_camera_images is None:
        combined = plt.figure(figsize=(7.15, 4.15))
        combined_axes = combined.subplots(2, 6, sharex=True, sharey=True)
        combined.subplots_adjust(
            left=0.065, right=0.995, bottom=0.075, top=0.535,
            wspace=0.08, hspace=0.22,
        )
        draw_occupancy_axes(
            combined_axes, current, temporal, original, length, width,
            current_horizons, temporal_horizons,
        )
        draw_camera_ring(combined, [0.18, 0.62, 0.64, 0.33], camera_by_id)
    else:
        combined, combined_axes = plt.subplots(
            3, 6, figsize=(7.15, 4.05),
            gridspec_kw={"height_ratios": [0.62, 1.0, 1.0]},
            constrained_layout=True,
        )
        for index, ax in enumerate(combined_axes[0]):
            ax.imshow(
                front_camera_images[index],
                aspect="auto",
                interpolation="antialiased",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        combined_axes[0, 0].set_ylabel(
            "Front camera", fontsize=7, labelpad=5,
        )
        draw_occupancy_axes(
            combined_axes[1:], current, temporal, original, length, width,
            current_horizons, temporal_horizons,
        )
    for extension in ("pdf", "png", "svg"):
        combined.savefig(
            outdir / ("fig02_scene_occupancy_case." + extension),
            dpi=300, bbox_inches="tight", pad_inches=0.02,
        )
    plt.close(combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    render(args.snapshot, args.outdir)


if __name__ == "__main__":
    main()

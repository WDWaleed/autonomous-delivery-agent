"""Visualization and rendering helpers for the delivery grid city.

Produces publication-quality stamped figures for reports and UI demos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from delivery.instance import Instance, get_canonical_instance
from delivery.stamp import banner, save_figure
from delivery.world.grid import GridWorld

# High-contrast, easily distinguishable terrain color palette
TERRAIN_COLORS = {
    "road": "#f8f9fa",      # Clean light asphalt / off-white
    "grass": "#4caf50",     # Saturated vivid green
    "mud": "#795548",       # Deep earthy chocolate brown
    "ford": "#0288d1",      # Vivid azure / water blue
    "blocked": "#1a1a1a",   # Jet black / impassable obstacle
}

TERRAIN_CMAP = mcolors.ListedColormap([
    TERRAIN_COLORS["blocked"],  # -1 -> 0 in mapped index
    TERRAIN_COLORS["road"],     #  0 -> 1
    TERRAIN_COLORS["grass"],    #  1 -> 2
    TERRAIN_COLORS["mud"],      #  2 -> 3
    TERRAIN_COLORS["ford"],     #  3 -> 4
])

BOUNDS = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
NORM = mcolors.BoundaryNorm(BOUNDS, TERRAIN_CMAP.N)


def render_grid_world(
    instance: Instance,
    path: Optional[List[Tuple[int, int]]] = None,
    rival_pos: Optional[Tuple[int, int]] = None,
    name: str = "canonical_grid",
    title: Optional[str] = None,
    output_dir: Optional[Path] = None,
    report_copy: bool = True,
) -> Tuple[Path, Path]:
    """Renders the grid city with terrain types, obstacles, depot, stops, rival, and optional path."""
    n = instance.n
    grid = instance.grid

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw terrain grid
    im = ax.imshow(
        grid,
        cmap=TERRAIN_CMAP,
        norm=NORM,
        origin="upper",
        extent=(0, n, n, 0),
    )

    # Grid line overlays
    ax.set_xticks(np.arange(0, n + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, n + 1, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.tick_params(which="minor", size=0)

    # Major ticks
    tick_step = 5 if n >= 20 else 2
    ax.set_xticks(np.arange(0, n, tick_step))
    ax.set_yticks(np.arange(0, n, tick_step))

    # Mark Start / Depot (Gold Amber Square with dark border)
    sr, sc = instance.start
    depot_patch = patches.Rectangle(
        (sc + 0.15, sr + 0.15),
        0.7,
        0.7,
        linewidth=2.5,
        edgecolor="#b26a00",
        facecolor="#ffb300",
        zorder=4,
        label="Start Depot",
    )
    ax.add_patch(depot_patch)
    ax.text(
        sc + 0.5,
        sr + 0.5,
        "D",
        color="#212121",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
        zorder=5,
    )

    # Mark Delivery Stops (Vivid Crimson Circles with crisp white borders)
    for idx, (stop_r, stop_c) in enumerate(instance.delivery_stops):
        stop_patch = patches.Circle(
            (stop_c + 0.5, stop_r + 0.5),
            0.38,
            linewidth=2,
            edgecolor="#ffffff",
            facecolor="#e53935",
            zorder=4,
            label="Delivery Stop" if idx == 0 else "",
        )
        ax.add_patch(stop_patch)
        ax.text(
            stop_c + 0.5,
            stop_r + 0.5,
            str(idx + 1),
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
            zorder=5,
        )

    # Mark Rival Agent (Purple Diamond)
    if rival_pos is not None:
        rr, rc = rival_pos
        rival_poly = patches.Polygon(
            [(rc + 0.5, rr + 0.1), (rc + 0.9, rr + 0.5), (rc + 0.5, rr + 0.9), (rc + 0.1, rr + 0.5)],
            linewidth=2,
            edgecolor="#4a148c",
            facecolor="#ab47bc",
            zorder=6,
            label=f"Rival ({instance.adversary.capitalize()})",
        )
        ax.add_patch(rival_poly)
        ax.text(
            rc + 0.5,
            rr + 0.5,
            "R",
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
            zorder=7,
        )

    # Optional path trace overlay
    if path and len(path) > 1:
        path_c = [c + 0.5 for _, c in path]
        path_r = [r + 0.5 for r, _ in path]
        ax.plot(
            path_c,
            path_r,
            color="#e65100",
            linewidth=3,
            linestyle="--",
            alpha=0.9,
            zorder=3,
            label="Planned Path",
        )

    # Legend for terrain types and entities
    legend_elements = [
        patches.Patch(facecolor=TERRAIN_COLORS["road"], edgecolor="#888", label=f"Road (${instance.terrain_cost['road']}$)"),
        patches.Patch(facecolor=TERRAIN_COLORS["grass"], edgecolor="#888", label=f"Grass (${instance.terrain_cost['grass']}$)"),
        patches.Patch(facecolor=TERRAIN_COLORS["mud"], edgecolor="#888", label=f"Mud (${instance.terrain_cost['mud']}$)"),
        patches.Patch(facecolor=TERRAIN_COLORS["ford"], edgecolor="#888", label=f"Ford (${instance.terrain_cost['ford']}$)"),
        patches.Patch(facecolor=TERRAIN_COLORS["blocked"], edgecolor="#888", label="Obstacle (-1)"),
        depot_patch,
        patches.Patch(facecolor="#e53935", edgecolor="#ffffff", label="Delivery Stop (1..k)"),
    ]
    if rival_pos is not None:
        legend_elements.append(rival_poly)

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        fontsize=9,
        framealpha=0.95,
    )

    plot_title = title or f"Autonomous Delivery Agent City Grid ({instance.id}) — {n}x{n}"
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Column", fontsize=10)
    ax.set_ylabel("Grid Row", fontsize=10)

    png_path, pdf_path = save_figure(fig, name, output_dir=output_dir, report_copy=report_copy)
    plt.close(fig)
    return png_path, pdf_path


def render_twist_overlay_view(
    instance: Instance,
    twist_name: str,
    name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    report_copy: bool = True,
) -> Tuple[Path, Path]:
    """Renders the city grid with a specific active twist overlay applied."""
    from delivery.twists.registry import get_twist_class
    from delivery.world.grid import GridWorld

    world = GridWorld(instance)
    twist_cls = get_twist_class(twist_name)
    twist = twist_cls()
    twist.initialize(instance, world)

    n = instance.n
    grid = instance.grid

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # Base grid
    ax.imshow(
        grid,
        cmap=TERRAIN_CMAP,
        norm=NORM,
        origin="upper",
        extent=(-0.5, n - 0.5, n - 0.5, -0.5),
        zorder=1,
    )

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5, alpha=0.3, zorder=2)
    ax.set_xticks(np.arange(0, n, 5))
    ax.set_yticks(np.arange(0, n, 5))

    # Start depot
    sr, sc = instance.start
    ax.plot(
        sc,
        sr,
        marker="s",
        markersize=13,
        color="#ffb300",
        markeredgecolor="#5d4037",
        markeredgewidth=1.8,
        zorder=6,
        label=f"Depot Start {instance.start}",
    )

    # Delivery stops
    for idx, (dr, dc) in enumerate(instance.delivery_stops, start=1):
        ax.plot(
            dc,
            dr,
            marker="o",
            markersize=11,
            color="#e53935",
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            zorder=6,
        )
        ax.text(
            dc,
            dr,
            str(idx),
            color="#ffffff",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
        )

    # Apply twist overlay
    twist.render_overlay(ax, world, state=instance.start)

    title_str = f"Twist Visualization: {twist.title} ({instance.id})"
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Column", fontsize=10)
    ax.set_ylabel("Grid Row", fontsize=10)

    fig_name = name or f"twist_{twist_name}"
    png_path, pdf_path = save_figure(fig, fig_name, output_dir=output_dir, report_copy=report_copy)
    plt.close(fig)
    return png_path, pdf_path


def render_world(
    world_or_inst: Any,
    route: Optional[List[Tuple[int, int]]] = None,
    agent: Optional[Tuple[int, int]] = None,
    rival: Optional[Tuple[int, int]] = None,
    layers: Tuple[str, ...] = (),
    name: str = "world_overview",
    output_dir: Optional[Path] = None,
    report_copy: bool = True,
) -> Tuple[Path, Path]:
    """Top-level renderer interface specified in milestone M1.3."""
    inst = world_or_inst.instance if hasattr(world_or_inst, "instance") else world_or_inst
    return render_grid_world(
        inst,
        path=route,
        rival_pos=rival,
        name=name,
        output_dir=output_dir,
        report_copy=report_copy,
    )


def main():
    print(banner("GRID WORLD RENDERER"))
    inst = get_canonical_instance()
    png_path, pdf_path = render_grid_world(inst, name="canonical_grid")
    render_world(inst, name="world_overview")
    render_twist_overlay_view(inst, "fog_partial_observability", name="twist_fog")
    render_twist_overlay_view(inst, "one_way_streets", name="twist_one_way")
    print(f"Rendered canonical city grid to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"  Overview: results/figures/world_overview.png")
    print(f"  Twist Fog: results/figures/twist_fog.png")
    print(f"  Twist One-Way: results/figures/twist_one_way.png")


if __name__ == "__main__":
    main()

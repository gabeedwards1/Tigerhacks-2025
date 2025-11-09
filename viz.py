# viz.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def show_search_heatmap(H, W, expanded_order, path, start, goal, layers, title="A* exploration", extent=None):
    """
    Terrain + A* expansion heatmap + legend.
    If extent is provided (from GeoTIFF window transform), axes are in map coords.
    """
    # Base image: prefer real elevation if available
    base = getattr(layers, "elevation_m", None)
    cmap = "terrain" if base is not None else "gray"
    if base is None:
        base = layers.slope * 0.7 + layers.rough * 0.3
        base = base.copy()
        base[layers.blocked] = base.max() + 0.3

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(base, origin='upper', cmap=cmap, alpha=0.9, extent=extent)

    # Expansion heat overlay
    if expanded_order:
        order_map = np.zeros((H, W), dtype=np.float32)
        for i, cell in enumerate(expanded_order):
            r, c = cell
            if 0 <= r < H and 0 <= c < W:
                order_map[r, c] = i + 1
        order_map /= max(1.0, order_map.max())
        heat = ax.imshow(order_map, origin='upper', cmap='viridis', alpha=0.6, extent=extent)
        cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("A* expansion (early → late)")

    # Path
    if path:
        ys, xs = zip(*path)
        if extent is None:
            ax.plot(xs, ys, color='cyan', linewidth=2.5, label="A* path")
            sx, sy = start[1], start[0]
            gx, gy = goal[1], goal[0]
        else:
            # convert (r,c) to map coords via provided extent linear mapping
            # extent = [xmin, xmax, ymax, ymin]; assume uniform scaling
            xmin, xmax, ymax, ymin = extent
            def rc_to_xy(r, c):
                # r in [0..H], c in [0..W]
                x = xmin + (xmax - xmin) * (c / (W - 1))
                y = ymin + (ymax - ymin) * (r / (H - 1))
                return x, y
            xs_m = []; ys_m = []
            for r, c in path:
                x, y = rc_to_xy(r, c)
                xs_m.append(x); ys_m.append(y)
            ax.plot(xs_m, ys_m, color='cyan', linewidth=2.5, label="A* path")
            sx, sy = rc_to_xy(*start)
            gx, gy = rc_to_xy(*goal)

        ax.scatter(sx, sy, s=100, edgecolors='black', facecolors='white', label="Start", zorder=3)
        ax.scatter(gx, gy, s=100, edgecolors='black', facecolors='yellow', label="Goal", zorder=3)

    legend_elements = [
        Line2D([0], [0], color='cyan', lw=2, label='A* path'),
        Line2D([0], [0], marker='o', color='w', label='Start',
               markerfacecolor='white', markeredgecolor='black', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Goal',
               markerfacecolor='yellow', markeredgecolor='black', markersize=9),
        Patch(facecolor='white', edgecolor='black', label='Blocked / Impassable'),
        Patch(facecolor='gray', label='Traversable (synthetic base)'),
        Patch(facecolor='purple', label='Early expansion (A*)'),
        Patch(facecolor='yellow', label='Late expansion (A*)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.85)
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

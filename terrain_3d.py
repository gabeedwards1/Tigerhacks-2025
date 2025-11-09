# terrain_3d.py
import numpy as np
import pyvista as pv

def plot_dem_3d(layers, path=None, meters_per_cell=None, title="Mars DEM 3D"):
    """
    layers.elevation_m: HxW elevation in meters (required)
    layers.blocked: boolean mask
    path: list[(r,c)]
    """
    elev = layers.elevation_m
    if elev is None:
        raise ValueError("layers.elevation_m is None; load from GeoTIFF to use 3D terrain.")
    H, W = elev.shape
    mpc = meters_per_cell or getattr(layers, "meters_per_cell", 1.0)

    # Build regular grid in real units
    xs = np.arange(W, dtype=np.float32) * mpc
    ys = np.arange(H, dtype=np.float32) * mpc
    xx, yy = np.meshgrid(xs, ys)

    # Create PyVista surface (Y forward so it matches images)
    surf = pv.StructuredGrid(xx, yy, elev.astype(np.float32))
    # Optional: color by slope (if available) or elevation
    if getattr(layers, "slope", None) is not None:
        surf["slope_deg"] = layers.slope.ravel(order="C")
        scalars = "slope_deg"
        cmap = "viridis"
    else:
        surf["elev_m"] = elev.ravel(order="C")
        scalars = "elev_m"
        cmap = "terrain"

    p = pv.Plotter()
    p.add_mesh(surf, scalars=scalars, cmap=cmap, show_edges=False)
    p.add_scalar_bar(title=scalars)

    # Overlay blocked cells as vertical “fences” (optional and sparse)
    if getattr(layers, "blocked", None) is not None:
        blk_idx = np.argwhere(layers.blocked)
        if len(blk_idx) > 0:
            yy_blk = blk_idx[:,0] * mpc
            xx_blk = blk_idx[:,1] * mpc
            zz = elev[blk_idx[:,0], blk_idx[:,1]]
            # draw small pillars to mark impassable
            for xb, yb, zb in zip(xx_blk[::200], yy_blk[::200], zz[::200]):  # decimate for speed
                p.add_lines(np.array([[xb, yb, zb], [xb, yb, zb + 50.0]]), color="white", width=2)

    # Overlay A* path as a tube
    if path:
        rc = np.array(path, dtype=np.int32)
        pr = rc[:,0].clip(0, H-1)
        pc = rc[:,1].clip(0, W-1)
        px = pc * mpc
        py = pr * mpc
        pz = elev[pr, pc]
        pts = np.c_[px, py, pz]
        p.add_mesh(pv.Spline(pts, 200).tube(radius=mpc*0.5), color="cyan")

    p.add_axes()
    p.show_grid()
    p.set_background("black")
    p.add_text(title, color="white")
    p.show()

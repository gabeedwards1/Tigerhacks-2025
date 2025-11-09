"""
run_simulation.py — Mars Rover Path Planning (fast ROI version, FIXED geo export)

Requires:
  pip install numpy matplotlib rasterio
Optional (for 3D):
  pip install pyvista pyvistaqt
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton

# Project modules
from rover_astar_sim import astar
from cost_layers import make_mars_from_geotiff_window, WeightedCost
from viz import show_search_heatmap
try:
    from terrain_3d import plot_dem_3d
    HAVE_3D = True
except Exception:
    HAVE_3D = False


# ==========================
#   Heuristics & Neighbors
# ==========================

def heuristic_fn(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def weighted_heuristic_fn(eps=1.2):
    def h(a, b):
        return eps * np.hypot(a[0] - b[0], a[1] - b[1])
    return h

def neighbors_fn_roi(H, W, bounds, four_connected=False):
    rmin, rmax, cmin, cmax = bounds
    steps4 = [(-1,0),(1,0),(0,-1),(0,1)]
    steps8 = steps4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    steps = steps4 if four_connected else steps8
    def fn(u):
        r, c = u
        nbrs = []
        for dr, dc in steps:
            rr, cc = r + dr, c + dc
            if rmin <= rr <= rmax and cmin <= cc <= cmax:
                nbrs.append((rr, cc))
        return nbrs
    return fn

def roi_bounds(points, H, W, pad=200):
    rs = [r for r,_ in points]; cs = [c for _,c in points]
    rmin = max(0, min(rs) - pad); rmax = min(H-1, max(rs) + pad)
    cmin = max(0, min(cs) - pad); cmax = min(W-1, max(cs) + pad)
    return rmin, rmax, cmin, cmax


# ==========================
#   Geo export helpers
# ==========================

def save_path_geojson_like(path_ll, outfile="route_lonlat.json"):
    with open(outfile, "w") as f:
        json.dump({"positions": path_ll}, f, indent=2)
    return outfile

def path_rc_to_lonlat_geotiff_resampled(
    path,
    *,
    transform,        # rasterio Affine for the selected window (native pixels)
    crs,              # dataset CRS
    win_h_native,     # height of the window in NATIVE pixels
    win_w_native,     # width  of the window in NATIVE pixels
    H_resampled,      # height of the resampled array used by A*
    W_resampled       # width  of the resampled array used by A*
):
    """
    Convert (row,col) from the RESAMPLED crop (H_resampled×W_resampled)
    to lon/lat using the window's native transform. We first map resampled
    indices -> native window indices, then call rasterio.transform.xy.
    """
    from rasterio.transform import xy
    try:
        from rasterio.crs import CRS
        from pyproj import Transformer
    except Exception:
        Transformer = None

    # resampled index -> native index (center-of-cell)
    sy = win_h_native / float(H_resampled)
    sx = win_w_native / float(W_resampled)

    to_wgs84 = None
    if crs and not getattr(crs, "is_geographic", False) and Transformer is not None:
        to_wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

    out = []
    for r_res, c_res in path:
        r_nat = (r_res + 0.5) * sy - 0.5
        c_nat = (c_res + 0.5) * sx - 0.5
        x_native, y_native = xy(transform, r_nat, c_nat, offset='center')
        if to_wgs84:
            lon, lat = to_wgs84.transform(x_native, y_native)
        else:
            lon, lat = x_native, y_native
        # keep lon in [-180,180)
        if lon >= 180: lon -= 360
        if lon < -180: lon += 360
        out.append({"lon": float(lon), "lat": float(lat)})
    return out


# ==========================
#   Main Tk App
# ==========================

class RoverApp:
    def __init__(self, root):
        self.root = root
        root.title("Mars Rover Path Planner")

        tk.Label(root, text="Mars Rover Path Planning", font=("Helvetica", 14, "bold")).pack(pady=8)

        # UI state
        self.tif_path = tk.StringVar()
        self.window = None
        self.layers = None

        # Geo metadata for export
        self.extent = None               # [xmin, xmax, ymax, ymin] (info only)
        self.window_transform = None     # Affine for the selected window (native)
        self.crs = None                  # dataset CRS
        self.win_size_native = None      # (H_native, W_native) of the window

        self.roundtrip = tk.BooleanVar(value=False)
        self.fast_mode = tk.BooleanVar(value=True)
        self.skip_3d   = tk.BooleanVar(value=not HAVE_3D)
        self.maxdim    = tk.IntVar(value=1024)

        # Controls
        frm = tk.Frame(root); frm.pack(pady=5)
        tk.Button(frm, text="Load GeoTIFF", command=self.load_tif).grid(row=0, column=0, padx=5)
        tk.Button(frm, text="Pick Region", command=self.pick_region).grid(row=0, column=1, padx=5)
        tk.Button(frm, text="Pick Points & Plan", command=self.plan).grid(row=0, column=2, padx=5)

        tk.Checkbutton(frm, text="Round Trip", variable=self.roundtrip).grid(row=1, column=0)
        tk.Checkbutton(frm, text="Fast Mode",  variable=self.fast_mode).grid(row=1, column=1)
        tk.Checkbutton(frm, text="Skip 3D",    variable=self.skip_3d).grid(row=1, column=2)

        tk.Label(frm, text="Max Dim:").grid(row=1, column=3, padx=(10,0))
        tk.Entry(frm, textvariable=self.maxdim, width=6).grid(row=1, column=4)

        # Log box
        self.logbox = tk.Text(root, height=11, width=88, state=tk.DISABLED,
                              bg="#111", fg="#0f0", font=("Courier", 9))
        self.logbox.pack(padx=10, pady=10)

    # ---- logging ----
    def log(self, msg):
        self.logbox.config(state=tk.NORMAL)
        self.logbox.insert(tk.END, msg + "\n")
        self.logbox.see(tk.END)
        self.logbox.config(state=tk.DISABLED)

    # ---- Step 1: load file ----
    def load_tif(self):
        path = filedialog.askopenfilename(title="Select GeoTIFF",
                                          filetypes=[("GeoTIFF", "*.tif *.tiff")])
        if not path:
            return
        self.tif_path.set(path)
        self.log(f"Selected GeoTIFF: {path}")

    # ---- Step 2: pick region ----
    def pick_region(self):
        import rasterio
        from rasterio.windows import Window

        if not self.tif_path.get():
            messagebox.showerror("Error", "Select a GeoTIFF first.")
            return

        tif_path = self.tif_path.get()
        self.log(f"Opening {tif_path} for region selection...")

        # Thumbnail for selection
        with rasterio.open(tif_path) as ds:
            scale = min(1.0, 800 / max(ds.width, ds.height))
            thumb = ds.read(
                1,
                out_shape=(int(ds.height * scale), int(ds.width * scale)),
                resampling=rasterio.enums.Resampling.bilinear
            ).astype(np.float32)
            thumb -= thumb.min()
            if thumb.max() > 0: thumb /= thumb.max()

        sel = {"extent": None}
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(thumb, cmap='gray', origin='upper')
        ax.set_title("Drag a box to select region. Close window when done.")

        def onselect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or x1 is None: return
            col_off = int(round(min(x0, x1) / scale))
            row_off = int(round(min(y0, y1) / scale))
            width   = int(round(abs(x1 - x0) / scale))
            height  = int(round(abs(y1 - y0) / scale))
            sel["extent"] = (col_off, row_off, width, height)
            ax.add_patch(plt.Rectangle((min(x0, x1), min(y0, y1)),
                                       abs(x1 - x0), abs(y1 - y0),
                                       edgecolor='red', facecolor='none', lw=2))
            fig.canvas.draw()

        _rs = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[MouseButton.LEFT],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            drag_from_anywhere=False
        )
        plt.show(block=True)

        if not sel["extent"]:
            self.log("No region selected.")
            return

        col_off, row_off, width, height = sel["extent"]
        if width < 10 or height < 10:
            self.log(f"Region too small ({width}×{height}). Pick a larger box.")
            messagebox.showwarning("Tiny selection", "Please pick at least 10×10 pixels.")
            return

        from rasterio.windows import Window
        win = Window(col_off, row_off, width, height)
        self.window = win
        self.win_size_native = (int(win.height), int(win.width))  # <-- FIX: store native window size
        self.log(f"Selected region: {width}×{height} native px")

        # Sanitize Max Dim and tighten if Fast Mode
        try:
            maxdim_val = int(self.maxdim.get())
        except Exception:
            maxdim_val = 1024
        if self.fast_mode.get():
            maxdim_val = min(maxdim_val, 768)
        if maxdim_val < 16:
            maxdim_val = 16

        # Load layers (resampled to target_max_dim)
        self.log("Loading GeoTIFF crop...")
        layers = make_mars_from_geotiff_window(
            tif_path,
            window=win,
            target_max_dim=maxdim_val,
            steep_block_thresh_deg=35.0,
            block_by_slope=True
        )
        self.layers = layers

        blk_frac = float(layers.blocked.mean())
        H, W = layers.slope.shape
        self.log(f"Loaded crop (resampled): {H}×{W} | meters/px ≈ {layers.meters_per_cell:.2f}")
        self.log(f"Blocked fraction: {blk_frac:.1%}")

        # Geographic extent + transform for the selected window
        with rasterio.open(tif_path) as ds:
            tf = rasterio.windows.transform(win, ds.transform)
            self.window_transform = tf
            self.crs = ds.crs
            # extent (info only) computed with NATIVE window size
            xmin, ymax = tf * (0, 0)
            xmax, ymin = tf * (int(win.width), int(win.height))
            self.extent = [xmin, xmax, ymax, ymin]
            if ds.crs and getattr(ds.crs, "is_geographic", False):
                self.log(f"Geo extent: lon {xmin:.3f}→{xmax:.3f}, lat {ymin:.3f}→{ymax:.3f}")
            else:
                self.log("Note: CRS not geographic; coordinates will be transformed to WGS84 if pyproj is available.")

    # ---- Step 3: pick points & plan ----
    def plan(self):
        if self.layers is None:
            messagebox.showerror("Error", "Load and crop a GeoTIFF first.")
            return

        # Pick points (start + 1+ goals)
        self.log("Pick start and goals (left-click). Close window when done.")
        img = self.layers.height if getattr(self.layers, "height", None) is not None else self.layers.slope
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img, cmap='gray', origin='upper')
        pts = []
        def onclick(event):
            if event.button == 1 and event.xdata is not None and event.ydata is not None:
                r, c = int(event.ydata), int(event.xdata)
                pts.append((r, c))
                ax.plot(c, r, 'ro'); fig.canvas.draw()
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)

        if len(pts) < 2:
            self.log("Need at least two points.")
            return

        start = pts[0]; goals = pts[1:]
        if self.roundtrip.get():
            goals.append(start)
        self.log(f"Start: {start} | Goals: {goals}")

        # Fast mode tuning
        H, W = self.layers.slope.shape
        four_conn = bool(self.fast_mode.get())
        eps       = 1.2 if self.fast_mode.get() else 1.0
        pad       = 150 if self.fast_mode.get() else int(0.35 * max(H, W))

        bounds = roi_bounds([start] + goals, H, W, pad=pad)
        rmin, rmax, cmin, cmax = bounds
        nfn = neighbors_fn_roi(H, W, bounds, four_connected=four_conn)
        hfn = weighted_heuristic_fn(eps=eps)
        self.log(f"ROI: rows {rmin}-{rmax}, cols {cmin}-{cmax} "
                 f"(~{(rmax-rmin+1)*(cmax-cmin+1):,} cells) | "
                 f"{'4' if four_conn else '8'}-connected | eps={eps}")

        cost_fn = WeightedCost().edge_cost_fn(self.layers)

        # Plan legs sequentially
        order = [start] + goals
        full_path = []
        total_cost = 0.0

        for i in range(len(order)-1):
            a, b = order[i], order[i+1]
            path, leg_cost, *_ = astar(a, b, nfn, cost_fn, hfn)
            if not path:
                self.log(f"Leg {a}->{b} not feasible. Try larger ROI (increase pad) or loosen blocking.")
                messagebox.showerror("Planning failed", f"No feasible path for leg {a}->{b}")
                return
            # avoid duplicating vertices when concatenating legs
            if full_path and path and full_path[-1] == path[0]:
                full_path.extend(path[1:])
            else:
                full_path.extend(path)
            total_cost += leg_cost
            self.log(f"Leg {a}->{b}: {len(path)} steps, cost={leg_cost:.1f}")

        self.log(f"Total: {len(full_path)} steps | cost={total_cost:.1f}")

        # 2D visualization
        show_search_heatmap(H, W, [], full_path, start, order[-1], self.layers,
                            title="A* (composite path)")

        # Optional 3D
        if HAVE_3D and not self.skip_3d.get():
            try:
                plot_dem_3d(self.layers, path=full_path,
                            meters_per_cell=self.layers.meters_per_cell,
                            title="Mars 3D Terrain (Path)")
            except Exception as e:
                self.log(f"3D view failed (skipped): {e}")

        # ---------- Export for globe (LOCAL-PATCH FIX) ----------
        if (self.window_transform is not None) and (self.win_size_native is not None):
            try:
                H_nat, W_nat = self.win_size_native
                ll = path_rc_to_lonlat_geotiff_resampled(
                    full_path,
                    transform=self.window_transform,
                    crs=self.crs,
                    win_h_native=H_nat,
                    win_w_native=W_nat,
                    H_resampled=H,
                    W_resampled=W
                )

                # --- detect whether coords look geographic ---
                lons = [p["lon"] for p in ll]
                lats = [p["lat"] for p in ll]
                lon_span = max(lons) - min(lons)
                lat_span = max(lats) - min(lats)

                # If values are in meters (large numbers or <10° spread), normalize them.
                if abs(lon_span) < 1.0 or abs(lat_span) < 1.0 or max(map(abs,lons)) > 360:
                    print("[INFO] Coordinates look local (projected); applying normalization.")
                    lon_min, lon_max = min(lons), max(lons)
                    lat_min, lat_max = min(lats), max(lats)
                    for p in ll:
                        p["lon"] = ((p["lon"] - lon_min) / (lon_max - lon_min)) * 10 - 5
                        p["lat"] = ((p["lat"] - lat_min) / (lat_max - lat_min)) * 10 - 5

                save_path_geojson_like(ll, "globe/route_lonlat.json")
                self.log("Exported route_lonlat.json for globe viewer (localized normalization).")
            except Exception as e:
                self.log(f"Geo export failed: {e}")



# ==========================
#   Main
# ==========================

if __name__ == "__main__":
    root = tk.Tk()
    app = RoverApp(root)
    app.log("Welcome! 1) Load GeoTIFF  2) Pick Region  3) Pick Points & Plan")
    app.log("Tip: Use Fast Mode first. Increase Max Dim later for detail.")
    root.mainloop()

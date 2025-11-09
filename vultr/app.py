# app.py — Local Flask API with A* over a Mars DEM streamed from Vultr
# deps: pip install flask numpy rasterio pillow pyproj requests

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Any
import io, math, heapq, requests, time
import numpy as np
from flask import Flask, request, jsonify, make_response
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

app = Flask(__name__)

# ======= CONFIG =======
COG_URL = "http://45.76.227.0:8081/mars_6p25_wgs84_cog.tif"   # remote GeoTIFF hosted on Vultr
MARS_R = 3_390_000.0  # Mars mean radius (m)

# Auto-rescale target range if raster is gray/hillshade-like
AUTO_MIN_M = -8200.0
AUTO_MAX_M = 21200.0

# Allow a tiny tolerance on single edges beyond the hard slope cap
EDGE_TOL = 1.10  # e.g., with max_slope=0.30, a single edge up to 0.33 may pass

# ======= CORS =======
@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Accept-Ranges"] = "bytes"
    return resp

# ======= utilities =======
def horiz_dist_m(lon1, lat1, lon2, lat2) -> float:
    """Flat-earth small-step metric using local scale (good for grid neighbors)."""
    to_rad = math.pi / 180.0
    x = (lon2 - lon1) * to_rad * math.cos((lat1 + lat2) * 0.5 * to_rad) * MARS_R
    y = (lat2 - lat1) * to_rad * MARS_R
    return math.hypot(x, y)

@dataclass
class GridSpec:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    W: int
    H: int

def lonlat_grid(spec: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(spec.min_lon, spec.max_lon, spec.W, dtype=np.float64)
    ys = np.linspace(spec.min_lat, spec.max_lat, spec.H, dtype=np.float64)
    lons = np.tile(xs, spec.H)           # size H*W
    lats = np.repeat(ys, spec.W)         # size H*W
    return lons, lats

def nearest_idx(lon, lat, spec: GridSpec) -> int:
    x = int(round((lon - spec.min_lon) / (spec.max_lon - spec.min_lon) * (spec.W - 1)))
    y = int(round((lat - spec.min_lat) / (spec.max_lat - spec.min_lat) * (spec.H - 1)))
    x = max(0, min(spec.W - 1, x))
    y = max(0, min(spec.H - 1, y))
    return y * spec.W + x

def idx_to_rc(i: int, W: int) -> Tuple[int, int]:
    return (i // W, i % W)

# ---- DEM sampling with automatic meter interpretation ----
def _looks_like_grayscale(arr: np.ndarray) -> bool:
    """Heuristic: values within uint8/uint16-like spans or very small dynamic range."""
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return True
    vmin, vmax = float(np.min(v)), float(np.max(v))
    dtype = arr.dtype
    if dtype == np.uint8 and 0.0 <= vmin <= 255.0 and 0.0 <= vmax <= 255.0:
        return True
    if dtype == np.uint16 and 0.0 <= vmin and vmax <= 65535.0:
        return True
    if (vmax - vmin) < 5.0:  # almost flat → likely shaded relief
        return True
    if 0.0 <= vmin and vmax < 1000.0:
        return True
    return False

def read_dem_window(spec: GridSpec) -> np.ndarray:
    """
    Reads (H,W) elevation in **meters**.
    - If dataset exposes band scales/offsets -> uses them.
    - Else, if the sample looks like grayscale/hillshade -> auto map [p2,p98] to [AUTO_MIN_M, AUTO_MAX_M].
    - Else -> raw values treated as meters.
    """
    with rasterio.open(COG_URL) as ds:
        # window in dataset CRS
        if ds.crs and ds.crs != CRS.from_epsg(4326):
            xs, ys = warp_transform(CRS.from_epsg(4326), ds.crs,
                                    [spec.min_lon, spec.max_lon],
                                    [spec.min_lat, spec.max_lat])
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
        else:
            minx, miny, maxx, maxy = spec.min_lon, spec.min_lat, spec.max_lon, spec.max_lat

        win = from_bounds(minx, miny, maxx, maxy, ds.transform)
        arr = ds.read(
            1, window=win, out_shape=(spec.H, spec.W),
            resampling=Resampling.bilinear, boundless=True, fill_value=np.nan
        )

        band_scale = None
        band_off   = None
        try:
            if getattr(ds, "scales", None):
                band_scale = (ds.scales or [None])[0]
            if getattr(ds, "offsets", None):
                band_off   = (ds.offsets or [None])[0]
        except Exception:
            band_scale, band_off = None, None

    # Fill NaNs
    if np.isnan(arr).any():
        valid = arr[np.isfinite(arr)]
        fill = float(np.mean(valid)) if valid.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)

    v = arr.astype(np.float64)

    # Case 1: dataset defines scale/offset → use them
    if band_scale not in (None, 1.0) or band_off not in (None, 0.0):
        s = 1.0 if band_scale is None else float(band_scale)
        o = 0.0 if band_off   is None else float(band_off)
        return v * s + o

    # Case 2: looks like gray → auto-rescale window to plausible Mars meters
    if _looks_like_grayscale(v):
        valid = v[np.isfinite(v)]
        if valid.size:
            p2, p98 = np.percentile(valid, [2, 98])
            span = max(p98 - p2, 1e-9)
            return ((v - p2) / span) * (AUTO_MAX_M - AUTO_MIN_M) + AUTO_MIN_M
        else:
            return np.zeros_like(v)

    # Case 3: treat as already-in-meters
    return v

# ---- Roughness + slope helpers ----
def compute_cell_metrics(elev_m: np.ndarray, spec: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      rough (H,W) in [0,1] via gradient magnitude percentile scaling,
      slope_grade (H,W) = |∇z| converted to grade per cell using metric spacing.
    """
    H, W = elev_m.shape
    mid_lat = 0.5 * (spec.min_lat + spec.max_lat)
    dlon = (spec.max_lon - spec.min_lon) / max(W - 1, 1)
    dlat = (spec.max_lat - spec.min_lat) / max(H - 1, 1)
    dx_m = (dlon * math.pi / 180.0) * math.cos(math.radians(mid_lat)) * MARS_R
    dy_m = (dlat * math.pi / 180.0) * MARS_R
    dx_m = max(dx_m, 1e-6)
    dy_m = max(dy_m, 1e-6)

    gy, gx = np.gradient(elev_m, dy_m, dx_m)  # dZ/dy, dZ/dx in m/m
    grad_mag = np.hypot(gx, gy)               # grade magnitude

    valid = grad_mag[np.isfinite(grad_mag)]
    if valid.size == 0:
        rough = np.zeros_like(grad_mag, dtype=np.float32)
    else:
        lo, hi = np.percentile(valid, [5, 95])
        span = max(hi - lo, 1e-6)
        rough = np.clip((grad_mag - lo) / span, 0, 1).astype(np.float32)

    return rough.astype(np.float32), grad_mag.astype(np.float32)

# ======= A* core (your implementation) =======
def neighbors_8(u, H, W):
    r, c = u
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield (rr, cc)

def euclid(a, b):
    ar, ac = a; br, bc = b
    return math.hypot(ar - br, ac - bc)

def reconstruct(parent, goal):
    path = []
    v = goal
    while v is not None:
        path.append(v)
        v = parent.get(v)
    path.reverse()
    return path

def astar(
    start: Tuple[int,int],
    goal: Tuple[int,int],
    neighbors_fn: Callable[[Tuple[int,int]], Any],
    edge_cost_fn: Callable[[Tuple[int,int], Tuple[int,int]], Optional[float]],
    heuristic_fn: Callable[[Tuple[int,int], Tuple[int,int]], float],
    *,
    weight: float = 1.0,
    epsilon: Optional[float] = None,
    max_expansions: Optional[int] = None,
    max_time_sec: Optional[float] = None,
    beam_width: Optional[int] = None
):
    if start == goal:
        return [start], 0.0, 0, [start], None

    t0 = time.time()
    counter = 0
    openh: List[Tuple[float, float, int, Tuple[int,int]]] = []
    h0 = heuristic_fn(start, goal)
    heapq.heappush(openh, (h0 * weight, h0, counter, start))
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    expansions = 0
    expanded_order = []
    best_goal_cost = None
    best_goal_node = None

    while openh:
        if max_time_sec is not None and (time.time() - t0) > max_time_sec:
            if best_goal_node is not None:
                return reconstruct(parent, best_goal_node), best_goal_cost, expansions, expanded_order, None
            return None, float("inf"), expansions, expanded_order, None

        f, h, _, u = heapq.heappop(openh)

        if epsilon is not None and best_goal_cost is not None:
            min_f = f
            if best_goal_cost <= (1.0 + epsilon) * min_f:
                return reconstruct(parent, best_goal_node), best_goal_cost, expansions, expanded_order, None

        if u in closed:
            continue
        closed.add(u)
        expanded_order.append(u)
        expansions += 1
        if max_expansions is not None and expansions >= max_expansions:
            if best_goal_node is not None:
                return reconstruct(parent, best_goal_node), best_goal_cost, expansions, expanded_order, None
            return None, float("inf"), expansions, expanded_order, None

        if u == goal:
            return reconstruct(parent, u), g[u], expansions, expanded_order, None

        gu = g[u]
        for v in neighbors_fn(u):
            c = edge_cost_fn(u, v)
            if c is None:
                continue
            alt = gu + c
            if v in closed:
                continue
            old = g.get(v)
            if old is None or alt < old - 1e-12:
                g[v] = alt
                parent[v] = u
                hv = heuristic_fn(v, goal)
                counter += 1
                heapq.heappush(openh, (alt + weight * hv, hv, counter, v))
                if v == goal:
                    if (best_goal_cost is None) or (alt < best_goal_cost - 1e-12):
                        best_goal_cost = alt
                        best_goal_node = v

        if beam_width is not None and len(openh) > beam_width:
            openh = heapq.nsmallest(beam_width, openh)
            heapq.heapify(openh)

    if best_goal_node is not None:
        return reconstruct(parent, best_goal_node), best_goal_cost, expansions, expanded_order, None

    return None, float("inf"), expansions, expanded_order, None

# ======= Energy model (optional) =======
class EnergyParams:
    def __init__(self,
                 mass_kg=185.0, g=3.71, Crr=0.03, eta=0.70,
                 meters_per_cell=1.0,
                 k_rough_J_per_m=40.0, downhill_regen_eff=0.10,
                 min_grade=-1.0, max_grade=1.0):
        self.mass_kg = mass_kg
        self.g = g
        self.Crr = Crr
        self.eta = max(1e-6, min(1.0, eta))
        self.meters_per_cell = meters_per_cell
        self.k_rough_J_per_m = k_rough_J_per_m
        self.downhill_regen_eff = max(0.0, min(0.5, downhill_regen_eff))
        self.min_grade = min_grade
        self.max_grade = max_grade

def _grid_step_m(dr, dc, meters_per_cell):
    base = math.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0
    return base * meters_per_cell

@dataclass
class Layers:
    elevation_m: np.ndarray   # (H,W)
    rough: np.ndarray         # (H,W) ~ [0,1]
    blocked: np.ndarray       # (H,W) bool

def move_energy_J(u, v, layers: Layers, P: EnergyParams):
    (r0, c0), (r1, c1) = u, v
    if layers.blocked[r1, c1]:
        return None
    dr, dc = r1 - r0, c1 - c0
    d_m = _grid_step_m(dr, dc, P.meters_per_cell)
    dh_m = float(layers.elevation_m[r1, c1] - layers.elevation_m[r0, c0])
    grade = (dh_m / d_m) if d_m > 0 else 0.0
    grade = max(P.min_grade, min(P.max_grade, grade))
    theta = math.atan(grade)

    m, g = P.mass_kg, P.g
    F_grav_up  = m * g * max(0.0, math.sin(theta))
    F_roll     = P.Crr * m * g * abs(math.cos(theta))
    E_mech_no_regen = (F_grav_up + F_roll) * d_m
    E_grav_downhill = m * g * max(0.0, -math.sin(theta)) * d_m
    E_regen = P.downhill_regen_eff * E_grav_downhill
    rough = float(layers.rough[r1, c1])
    E_rough = P.k_rough_J_per_m * rough * d_m
    E_drive_in = E_mech_no_regen / P.eta
    E_batt = E_drive_in + E_rough - E_regen
    return max(0.0, E_batt)

def physical_energy_cost_fn(layers: Layers, params: EnergyParams, scale_cost=1.0):
    def cost(u, v):
        E = move_energy_J(u, v, layers, params)
        if E is None:
            return None
        return float(E * scale_cost)
    return cost

# ======= helpers for connectivity =======
def nearest_unblocked(rc: Tuple[int,int], blocked: np.ndarray, max_radius: int = 25) -> Tuple[int,int]:
    """If rc is blocked, walk outward until we find the closest free cell."""
    r, c = rc
    H, W = blocked.shape
    if not blocked[r, c]:
        return rc
    best = None
    best_d2 = 1e18
    for rad in range(1, max_radius + 1):
        r0, r1 = max(0, r - rad), min(H - 1, r + rad)
        c0, c1 = max(0, c - rad), min(W - 1, c + rad)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if not blocked[rr, cc]:
                    d2 = (rr - r) * (rr - r) + (cc - c) * (cc - c)
                    if d2 < best_d2:
                        best = (rr, cc)
                        best_d2 = d2
        if best is not None:
            return best
    return rc  # give up if fully enclosed

# ======= public preview endpoints =======
@app.route("/", methods=["GET"])
def root():
    return {"ok": True, "source_tif": COG_URL, "astar": "/astar/solve (POST JSON)"}

@app.route("/cog/part", methods=["GET"])
def cog_part():
    bbox = request.args.get("bbox", "")
    try:
        minx, miny, maxx, maxy = [float(x) for x in bbox.split(",")]
    except Exception:
        return jsonify({"error": "bbox=minx,miny,maxx,maxy required"}), 400

    W = int(request.args.get("width", "256"))
    H = int(request.args.get("height", "256"))
    resampling = getattr(Resampling, request.args.get("resampling", "nearest").lower(), Resampling.nearest)

    try:
        r = requests.head(COG_URL, timeout=5)
        if r.status_code != 200:
            return jsonify({"error": "Remote COG not reachable"}), 502
    except Exception as e:
        return jsonify({"error": f"COG check failed: {e}"}), 502

    with rasterio.open(COG_URL) as ds:
        if ds.crs and ds.crs != CRS.from_epsg(4326):
            xs, ys = warp_transform(CRS.from_epsg(4326), ds.crs, [minx, maxx], [miny, maxy])
            minx_ds, maxx_ds = min(xs), max(xs)
            miny_ds, maxy_ds = min(ys), max(ys)
        else:
            minx_ds, miny_ds, maxx_ds, maxy_ds = minx, miny, maxx, maxy

        win = from_bounds(minx_ds, miny_ds, maxx_ds, maxy_ds, ds.transform)
        arr = ds.read(1, window=win, out_shape=(H, W), resampling=resampling, boundless=True, fill_value=np.nan).astype("float64")

    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.percentile(valid, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(valid)), float(np.nanmax(valid))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0

    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    png = (scaled * 255).astype("uint8")
    buf = io.BytesIO(); Image.fromarray(png, "L").save(buf, "PNG"); buf.seek(0)
    resp = make_response(buf.read()); resp.headers["Content-Type"] = "image/png"; return resp

@app.route("/cog/point", methods=["GET"])
def cog_point():
    try:
        lon = float(request.args["lon"]); lat = float(request.args["lat"])
    except Exception:
        return jsonify({"error":"lon and lat required"}), 400

    try:
        r = requests.head(COG_URL, timeout=5)
        if r.status_code != 200:
            return jsonify({"error": "Remote COG not reachable"}), 502
    except Exception as e:
        return jsonify({"error": f"COG check failed: {e}"}), 502

    with rasterio.open(COG_URL) as ds:
        xy = (lon, lat)
        if ds.crs and ds.crs != CRS.from_epsg(4326):
            xs, ys = warp_transform(CRS.from_epsg(4326), ds.crs, [lon], [lat])
            xy = (xs[0], ys[0])
        try:
            v = float(list(ds.sample([xy]))[0][0])
        except Exception:
            v = float("nan")
    return jsonify({"coordinate":[lon,lat], "values":[None if np.isnan(v) else v]})

# ======= A* API =======
@app.route("/astar/solve", methods=["POST"])
def astar_solve():
    """
    JSON body (minimal):
    {
      "positions":[{"lon":..,"lat":..}, ...],  // >= 2
      "grid": 128,
      "margin_km": 40,
      "max_slope": 0.6,
      "slope_weight": 2.0,
      "cost": "slope" | "energy",             // default "slope"
      "weight": 1.0,                           // Weighted A*
      "epsilon": null,                         // (1+ε)-optimal early stop
      "beam_width": null,
      "max_time_sec": null,
      "max_expansions": null
    }
    """
    # Ensure remote COG reachable
    try:
        r = requests.head(COG_URL, timeout=5)
        if r.status_code != 200:
            return jsonify({"error": "Remote COG not reachable"}), 502
    except Exception as e:
        return jsonify({"error": f"COG check failed: {e}"}), 502

    data = request.get_json(force=True, silent=True) or {}
    pts = data.get("positions") or []
    if len(pts) < 2:
        return jsonify({"error":"positions must have at least 2 points"}), 400

    # Hyperparameters
    N           = int(data.get("grid", 128))
    mk          = float(data.get("margin_km", 40.0))
    max_grade   = float(data.get("max_slope", 0.6))
    slope_w     = float(data.get("slope_weight", 2.0))
    cost_mode   = (data.get("cost") or "slope").lower()

    weight      = float(data.get("weight", 1.0))
    epsilon     = data.get("epsilon", None)
    epsilon     = (None if epsilon in (None, "", "null") else float(epsilon))
    beam_width  = data.get("beam_width", None)
    beam_width  = (None if beam_width in (None, "", "null") else int(beam_width))
    max_time    = data.get("max_time_sec", None)
    max_time    = (None if max_time in (None, "", "null") else float(max_time))
    max_exp     = data.get("max_expansions", None)
    max_exp     = (None if max_exp in (None, "", "null") else int(max_exp))

    lons = [float(p["lon"]) for p in pts]
    lats = [float(p["lat"]) for p in pts]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    mid_lat = 0.5 * (min_lat + max_lat)

    def km2deg_lat(km): return (km * 1000.0) / (MARS_R * (math.pi / 180.0))
    def km2deg_lon(km, lat): return (km * 1000.0) / (MARS_R * math.cos(math.radians(lat)) * (math.pi / 180.0))

    # Expand bbox by margin
    min_lon -= km2deg_lon(mk, mid_lat); max_lon += km2deg_lon(mk, mid_lat)
    min_lat -= km2deg_lat(mk);          max_lat += km2deg_lat(mk)

    # Build grid + DEM (meters)
    spec = GridSpec(min_lon, min_lat, max_lon, max_lat, N, N)
    elev = read_dem_window(spec)             # (H,W) meters
    rough, slope_grade = compute_cell_metrics(elev, spec)

    # Metric spacing for energy model & rough edge distance
    dlon = (spec.max_lon - spec.min_lon) / max(N - 1, 1)
    dlat = (spec.max_lat - spec.min_lat) / max(N - 1, 1)
    dx_m = (dlon * math.pi / 180.0) * math.cos(math.radians(mid_lat)) * MARS_R
    dy_m = (dlat * math.pi / 180.0) * MARS_R
    meters_per_cell = 0.5 * (dx_m + dy_m)

    # Block too-steep cells (hard constraint)
    blocked = slope_grade > max_grade

    layers = Layers(elevation_m=elev.astype(np.float32),
                    rough=rough,
                    blocked=blocked)

    # Neighbors and lon/lat mapping
    H, W = N, N
    xs = np.linspace(spec.min_lon, spec.max_lon, W)
    ys = np.linspace(spec.min_lat, spec.max_lat, H)

    def rc_to_lonlat(r: int, c: int) -> Tuple[float, float]:
        return float(xs[c]), float(ys[r])

    def neigh(u: Tuple[int,int]):
        return neighbors_8(u, H, W)

    # Heuristic in meters using lon/lat conversion
    def h_m(u: Tuple[int,int], g: Tuple[int,int]) -> float:
        ulon, ulat = rc_to_lonlat(u[0], u[1])
        glon, glat = rc_to_lonlat(g[0], g[1])
        return horiz_dist_m(ulon, ulat, glon, glat)

    # Edge costs (slope or energy)
    if cost_mode == "energy":
        params = EnergyParams(meters_per_cell=meters_per_cell)
        edge_cost = physical_energy_cost_fn(layers, params, scale_cost=1.0)
    else:
        # slope-based distance cost with hard max_slope (+ small EDGE_TOL on a single step)
        def edge_cost(u: Tuple[int,int], v: Tuple[int,int]) -> Optional[float]:
            r0, c0 = u; r1, c1 = v
            if layers.blocked[r1, c1]:
                return None
            lon0, lat0 = rc_to_lonlat(r0, c0)
            lon1, lat1 = rc_to_lonlat(r1, c1)
            hd = horiz_dist_m(lon0, lat0, lon1, lat1)
            if hd <= 1e-6:
                return None
            dh = float(layers.elevation_m[r1, c1] - layers.elevation_m[r0, c0])
            grade = abs(dh) / hd
            if grade > (max_grade * EDGE_TOL):
                return None
            return hd * (1.0 + slope_w * grade)

    # Map waypoints -> nearest grid rc and **snap to nearest unblocked cell**
    way_rc: List[Tuple[int,int]] = []
    start_blocked_flags: List[bool] = []
    for lon, lat in zip(lons, lats):
        i = nearest_idx(lon, lat, spec)
        rc = idx_to_rc(i, W)
        was_blocked = bool(layers.blocked[rc[0], rc[1]])
        rc = nearest_unblocked(rc, layers.blocked, max_radius=25)
        start_blocked_flags.append(was_blocked)
        way_rc.append(rc)

    # Run A* leg-by-leg (Start->WP1->...->End)
    path_rc: List[Tuple[int,int]] = []
    legs_cost: List[float] = []
    totals = 0.0

    for i in range(len(way_rc) - 1):
        s_rc = way_rc[i]
        t_rc = way_rc[i + 1]
        path, cost, *_ = astar(
            start=s_rc,
            goal=t_rc,
            neighbors_fn=neigh,
            edge_cost_fn=edge_cost,
            heuristic_fn=h_m,
            weight=weight,
            epsilon=epsilon,
            max_expansions=max_exp,
            max_time_sec=max_time,
            beam_width=beam_width
        )
        if path is None:
            diag = {
                "leg": i + 1,
                "grid": N,
                "margin_km": mk,
                "max_slope": max_grade,
                "edge_tol": EDGE_TOL,
                "blocked_ratio": float(layers.blocked.mean()),
                "start_blocked_clicked": start_blocked_flags[i],
                "end_blocked_clicked": start_blocked_flags[i+1],
                "max_grade_in_window": float(np.max(slope_grade)),
            }
            return jsonify({"error": f"No path for leg {i+1}. Try bigger grid/margin or relax constraints.",
                            "diag": diag}), 200
        if i > 0:
            path = path[1:]  # stitch
        path_rc.extend(path)
        legs_cost.append(float(cost))
        totals += float(cost)

    # Convert to lon/lat polyline
    positions = []
    for (r, c) in path_rc:
        lon, lat = rc_to_lonlat(r, c)
        positions.append({"lon": float(lon), "lat": float(lat)})

    return jsonify({"positions": positions, "total_cost_m": float(totals), "legs_m": legs_cost})

# ======= main =======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, threaded=True)

# cost_layers.py
# ----------------
# Terrain layers and edge-cost functions for Mars rover path planning.
#
# Exposes:
#   - CostLayers              (data container)
#   - make_mars_from_geotiff_window(path, window, target_max_dim, ...)
#   - make_synthetic_mars(H=256, W=256, seed=0, ...)
#   - WeightedCost            (builds edge_cost_fn(layers))
#
# Dependencies: numpy, rasterio (for GeoTIFF path)
# Optional: nothing else

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

# RasterIO is imported lazily inside functions that need it.


# -----------------------------
# Data container for terrain
# -----------------------------

@dataclass
class CostLayers:
    """
    height: DEM in meters (or None if unknown)
    slope:  per-cell slope in degrees, shape (H,W)
    rough:  per-cell roughness [0..1], shape (H,W)
    blocked: boolean mask, True = impassable, shape (H,W)
    meters_per_cell: approximate ground resolution for one grid step
    """
    height: Optional[np.ndarray]
    slope: np.ndarray
    rough: np.ndarray
    blocked: np.ndarray
    meters_per_cell: float


# -----------------------------
# Utilities
# -----------------------------

def _safe_float(x) -> float:
    """Return Python float for a 0-dim numpy scalar."""
    return float(np.asarray(x))

def _compute_slope_deg(height: np.ndarray, meters_per_cell: float) -> np.ndarray:
    """
    Slope from DEM using central differences.
      slope = arctan( sqrt( (dz/dx)^2 + (dz/dy)^2 ) )  [degrees]
    """
    if height is None:
        raise ValueError("Height/DEM required to compute slope.")
    # spacing in meters for both axes
    gy, gx = np.gradient(height, meters_per_cell, meters_per_cell)
    grad_mag = np.hypot(gx, gy)
    slope_rad = np.arctan(grad_mag)
    slope_deg = np.degrees(slope_rad)
    # sanitize
    slope_deg = np.nan_to_num(slope_deg, nan=0.0, posinf=90.0, neginf=0.0)
    return slope_deg.astype(np.float32)

def _compute_roughness(height: np.ndarray) -> np.ndarray:
    """
    Cheap roughness proxy from gradient magnitude; normalized [0..1].
    No SciPy required.
    """
    if height is None:
        # If no DEM, provide a mild constant roughness.
        return np.full((1, 1), 0.0, dtype=np.float32)
    gy, gx = np.gradient(height.astype(np.float32))
    g = np.hypot(gx, gy)
    # Local normalize to [0..1]
    g = g - g.min()
    vmax = g.max()
    if vmax > 1e-12:
        g = g / vmax
    return g.astype(np.float32)


# -----------------------------
# GeoTIFF window loader
# -----------------------------

def make_mars_from_geotiff_window(
    tif_path: str,
    window,
    target_max_dim: Optional[int] = 1024,
    steep_block_thresh_deg: float = 35.0,
    block_by_slope: bool = True,
) -> CostLayers:
    """
    Load a sub-window of a (huge) GeoTIFF, downsample to target_max_dim on the
    longer side, and compute slope/rough/blocked masks.

    Args:
      tif_path: path to a GeoTIFF DEM or shaded relief.
      window:   rasterio.windows.Window specifying (col_off,row_off,width,height)
      target_max_dim: largest output dimension in pixels (None/<=0 keeps native)
      steep_block_thresh_deg: slope above which cells become blocked if enabled
      block_by_slope: if True, slopes above threshold are impassable

    Returns:
      CostLayers
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif_path) as ds:
        H_src, W_src = int(window.height), int(window.width)

        # --- robust guards ---
        if H_src < 1 or W_src < 1:
            raise ValueError(
                f"Selected window is empty (H={H_src}, W={W_src}). "
                "Please reselect a non-zero region."
            )

        # work out resolution (approx meters per pixel along y)
        # If transform is geographic, meters/px is not exact; use crude estimate
        # from degree length near equator (not perfect but good for cost scaling).
        tf = rasterio.windows.transform(window, ds.transform)
        # Pixel size (xres, yres) in *native units* (deg if EPSG:4326, meters if projected)
        xres = abs(tf.a)
        yres = abs(tf.e)

        # Try to derive meters-per-cell:
        meters_per_cell = 1.0
        try:
            if ds.crs and getattr(ds.crs, "is_geographic", False):
                # degrees -> meters (approx): 1° lat ≈ 111.32 km; use yres
                meters_per_cell = float(yres * 111_320.0)
            else:
                # projected (meters per pixel)
                meters_per_cell = float(yres)
        except Exception:
            meters_per_cell = 1.0  # fallback

        # --- downsampling scale ---
        if target_max_dim is None or int(target_max_dim) < 1:
            target_max_dim = max(H_src, W_src)

        denom = max(1, max(H_src, W_src))
        scale = min(1.0, float(target_max_dim) / float(denom))

        H_out = max(1, int(round(H_src * scale)))
        W_out = max(1, int(round(W_src * scale)))

        # --- read DEM as float32, resampled to (H_out,W_out) ---
        # If dataset has multiple bands, try band 1 as height proxy.
        arr = ds.read(
            1,
            window=window,
            out_shape=(H_out, W_out),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        # Handle nodata
        nodata = ds.nodata
        if nodata is not None:
            mask_nodata = np.isclose(arr, nodata)
        else:
            mask_nodata = np.isnan(arr)
        arr = np.where(mask_nodata, np.nan, arr)

    height = arr  # DEM (can be NaN at nodata)
    # Slope & roughness
    slope_deg = _compute_slope_deg(np.nan_to_num(height, nan=np.nanmean(height)), meters_per_cell)
    rough = _compute_roughness(np.nan_to_num(height, nan=np.nanmean(height)))

    # Blocked:
    blocked = np.zeros_like(slope_deg, dtype=bool)

    # Block nodata/NaNs
    blocked |= ~np.isfinite(height)

    # Optionally block by slope threshold
    if block_by_slope and steep_block_thresh_deg is not None:
        blocked |= (slope_deg >= float(steep_block_thresh_deg))

    # Final sanitize
    slope_deg = np.nan_to_num(slope_deg, nan=0.0, posinf=90.0, neginf=0.0).astype(np.float32)
    rough = np.nan_to_num(rough, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    return CostLayers(
        height=height.astype(np.float32),
        slope=slope_deg,
        rough=rough,
        blocked=blocked,
        meters_per_cell=float(meters_per_cell),
    )


# -----------------------------
# Synthetic terrain (for tests)
# -----------------------------

def make_synthetic_mars(
    H: int = 256,
    W: int = 256,
    seed: int = 0,
    steep_block_thresh_deg: float = 40.0,
    block_by_slope: bool = True,
    meters_per_cell: float = 5.0,
) -> CostLayers:
    """
    Generates a synthetic "Mars-like" height field with gentle undulations,
    craters, and a few steep ridges. Good for quick tests without a GeoTIFF.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 6*np.pi, H), np.linspace(0, 6*np.pi, W), indexing='ij')

    base = 200 * np.sin(0.2*xx) * np.cos(0.15*yy)
    long_waves = 120 * np.sin(0.05*xx + 0.3) * np.cos(0.04*yy - 0.8)
    noise = rng.normal(0, 10.0, (H, W))
    height = (base + long_waves + noise).astype(np.float32)

    # a few "craters"
    for _ in range(6):
        r0 = rng.integers(0, H)
        c0 = rng.integers(0, W)
        rr, cc = np.ogrid[:H, :W]
        dist = np.hypot(rr - r0, cc - c0)
        height -= 150 * np.exp(-(dist**2) / (2*(rng.uniform(8, 18)**2)))

    slope_deg = _compute_slope_deg(height, meters_per_cell)
    rough = _compute_roughness(height)

    blocked = np.zeros((H, W), dtype=bool)
    if block_by_slope and steep_block_thresh_deg is not None:
        blocked |= (slope_deg >= float(steep_block_thresh_deg))

    return CostLayers(
        height=height,
        slope=np.nan_to_num(slope_deg, nan=0.0).astype(np.float32),
        rough=np.nan_to_num(rough, nan=0.0).astype(np.float32),
        blocked=blocked,
        meters_per_cell=float(meters_per_cell),
    )


# -----------------------------
# Edge cost builder
# -----------------------------

class WeightedCost:
    """
    Builds an edge cost function that blends:
      - distance (meters)
      - slope penalty (degrees)
      - surface roughness penalty [0..1]
    and treats blocked cells as impassable (infinite cost).

    Parameters:
      w_dist:  weight for pure travel distance
      w_slope: weight for slope penalty (scaled by slope/45)
      w_rough: weight for roughness penalty (scaled by rough)
      diag_cost: multiplier for diagonal steps (sqrt(2) by default)
      block_penalty: cost to return for blocked cells (np.inf => impassable)
    """
    def __init__(
        self,
        w_dist: float = 1.0,
        w_slope: float = 3.0,
        w_rough: float = 1.0,
        diag_cost: float = np.sqrt(2.0),
        block_penalty: float = np.inf,
    ):
        self.w_dist = float(w_dist)
        self.w_slope = float(w_slope)
        self.w_rough = float(w_rough)
        self.diag_cost = float(diag_cost)
        self.block_penalty = float(block_penalty)

    def edge_cost_fn(self, layers: CostLayers) -> Callable[[Tuple[int,int], Tuple[int,int]], float]:
        """
        Returns a function(u, v) that computes edge cost from cell u -> v.
        Assumes 4- or 8-connected neighbors; diagonal steps use diag_cost.

        Cost model:
            step_dist_meters
          * ( w_dist
            + w_slope * (slope_deg(v) / 45)
            + w_rough * rough(v) )

        If destination v is blocked -> returns block_penalty (np.inf).
        """
        slope = layers.slope
        rough = layers.rough
        blocked = layers.blocked
        mpc = float(layers.meters_per_cell)

        H, W = slope.shape

        def cost(u: Tuple[int,int], v: Tuple[int,int]) -> float:
            ur, uc = u
            vr, vc = v

            # Bound check (should be unnecessary if neighbor generator is correct)
            if not (0 <= vr < H and 0 <= vc < W):
                return self.block_penalty

            # If destination is blocked => impassable
            if bool(blocked[vr, vc]):
                return self.block_penalty

            dr = vr - ur
            dc = vc - uc
            step = self.diag_cost if (dr != 0 and dc != 0) else 1.0
            step_dist = mpc * step

            sdeg = _safe_float(slope[vr, vc])  # degrees
            rgh = _safe_float(rough[vr, vc])   # 0..1

            penalty = (
                self.w_dist
                + self.w_slope * (sdeg / 45.0)
                + self.w_rough * rgh
            )

            return float(step_dist * penalty)

        return cost

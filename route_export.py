# route_export.py
# Utilities to export an A* path to route_lonlat.json for the Three.js globe.

from __future__ import annotations
import json
from typing import Iterable, Tuple, Sequence, Optional

# ----------------------------
# Case A: Full-planet equirectangular grid (H×W)
# r = 0 at north edge, r = H-1 at south edge
# c = 0 at 180°W, c = W-1 near 180°E
# ----------------------------

def rc_to_lonlat_full_mars(r: int, c: int, H: int, W: int) -> Tuple[float, float]:
    """Center-of-cell mapping from grid (row,col) to lon/lat degrees."""
    lon = ((c + 0.5) / W) * 360.0 - 180.0
    lat = 90.0 - ((r + 0.5) / H) * 180.0
    return lon, lat

def write_route_lonlat_grid(path_rc: Sequence[Tuple[int,int]], H: int, W: int,
                            out_path: str = "route_lonlat.json") -> None:
    """Export path of (r,c) indices for a full-planet equirectangular grid."""
    positions = []
    for r, c in path_rc:
        lon, lat = rc_to_lonlat_full_mars(int(r), int(c), H, W)
        positions.append({"lon": lon, "lat": lat})
    with open(out_path, "w") as f:
        json.dump({"positions": positions}, f, indent=2)
    print(f"Wrote {len(positions)} points to {out_path}")

# ----------------------------
# Case B: Image pixels (x,y) on an equirectangular map of size W×H
# ----------------------------

def xy_to_lonlat(x: float, y: float, W: int, H: int) -> Tuple[float, float]:
    lon = (x / W) * 360.0 - 180.0
    lat = 90.0 - (y / H) * 180.0
    return lon, lat

def write_route_lonlat_pixels(path_xy: Sequence[Tuple[float,float]], W: int, H: int,
                              out_path: str = "route_lonlat.json") -> None:
    positions = []
    for x, y in path_xy:
        lon, lat = xy_to_lonlat(float(x), float(y), W, H)
        positions.append({"lon": lon, "lat": lat})
    with open(out_path, "w") as f:
        json.dump({"positions": positions}, f, indent=2)
    print(f"Wrote {len(positions)} points to {out_path}")

# ----------------------------
# Case C: GeoTIFF raster indices (row,col) -> lon/lat
# Requires rasterio; if CRS != EPSG:4326, uses pyproj to transform
# ----------------------------

def write_route_lonlat_geotiff(path_rc: Sequence[Tuple[int,int]],
                               tif_path: str,
                               out_path: str = "route_lonlat.json") -> None:
    try:
        import rasterio
        from rasterio.transform import rowcol, xy
        from rasterio.crs import CRS
        from pyproj import Transformer
    except Exception as e:
        raise RuntimeError(
            "GeoTIFF export requires rasterio and pyproj. "
            "Install with: pip install rasterio pyproj"
        ) from e

    with rasterio.open(tif_path) as ds:
        crs = ds.crs
        transform = ds.transform
        # Build transformer to WGS84 if needed
        if crs and crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            to_wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
            def to_lonlat(col: float, row: float):
                X, Y = xy(transform, row, col, offset="center")  # native coords
                lon, lat = to_wgs84.transform(X, Y)
                return lon, lat
        else:
            # Already lon/lat; xy() returns (lon, lat)
            def to_lonlat(col: float, row: float):
                lon, lat = xy(transform, row, col, offset="center")
                return lon, lat

        positions = []
        for r, c in path_rc:
            lon, lat = to_lonlat(float(c), float(r))
            positions.append({"lon": float(lon), "lat": float(lat)})

    with open(out_path, "w") as f:
        json.dump({"positions": positions}, f, indent=2)
    print(f"Wrote {len(positions)} points to {out_path}")

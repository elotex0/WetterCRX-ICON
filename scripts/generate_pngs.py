import sys
import json
import os
import gc
import warnings
from zoneinfo import ZoneInfo

import cfgrib
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

from encoding_config import NODATA_U8, LEVELS, ENCODING

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]          # z.B. "data/t2m"
output_dir = sys.argv[2]        # z.B. "icon/t2m"
var_type = sys.argv[3].lower()  # 't2m', 'tp', 'tp_acc', 'ww', 'cape_ml', 'dbz_cmax', 'wind', 'snow'
grid_dir = sys.argv[4] if len(sys.argv) > 4 else "data/grid"

os.makedirs(output_dir, exist_ok=True)

if var_type not in ENCODING:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)
enc = ENCODING[var_type]

# ICON (global) läuft auf dem nativen Dreiecksgitter - die Gitterkoordinaten
# müssen separat aus den CLAT-/CLON-GRIB2-Dateien geladen werden.
clat_path = os.path.join(grid_dir, "clat.grib2")
clon_path = os.path.join(grid_dir, "clon.grib2")

for gp in (clat_path, clon_path):
    if not os.path.exists(gp):
        raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gp}")

# ------------------------------
# Ausgaberaster (identisch zum ICON-D2-Raster)
# ------------------------------
# regular-lat-lon, EPSG:4326, row0 = Norden, Werte = Pixelzentren.
# MUSS zu GRID in generate_metadata.py passen (dort wird die Dateigröße geprüft).
GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX = -3.94, 43.18, 20.34, 58.08
GRID_WIDTH, GRID_HEIGHT = 1215, 746

# Wettercodes (ww): Kantenlänge der "Blöcke" in Ausgabepixeln. Das native
# ICON-Gitter ist ~13 km grob - mit BLOCK > 1 werden die Codes auf quadratische
# Kacheln gerastert (wie im bisherigen WebP-Look). BLOCK = 1 -> reines
# Nearest-Neighbor ohne Rasterung.
WW_BLOCK = 8

lon_t = np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_WIDTH)
lat_t = np.linspace(GRID_LAT_MAX, GRID_LAT_MIN, GRID_HEIGHT)   # Nord -> Süd
lon_tt, lat_tt = np.meshgrid(lon_t, lat_t)                     # Shape (HEIGHT, WIDTH)

# Für Triangulation/Nearest-Neighbor wird die Länge mit cos(Breite) skaliert,
# damit die Dreiecke in der (lon, lat)-Ebene annähernd unverzerrt sind.
COS_LAT0 = np.cos(np.radians(0.5 * (GRID_LAT_MIN + GRID_LAT_MAX)))
target_points = np.column_stack((lon_tt.ravel() * COS_LAT0, lat_tt.ravel()))
n_target = target_points.shape[0]


def encode_uint8(data, enc):
    """float-Array (NaN erlaubt) -> uint8-Array (255 = NoData)."""
    nan_mask = ~np.isfinite(data)
    vmin, vmax, tr = enc["vmin"], enc["vmax"], enc["transform"]
    d = np.where(nan_mask, vmin, data).astype(np.float64)

    if tr == "raw":
        q = np.clip(np.rint(d), 0, LEVELS)
    else:
        d = np.clip(d, vmin, vmax)
        if tr == "linear":
            u = (d - vmin) / (vmax - vmin)
        elif tr == "sqrt":
            u = np.sqrt((d - vmin) / (vmax - vmin))
        elif tr == "ssqrt":
            u = (np.sign(d) * np.sqrt(np.abs(d) / vmax) + 1.0) / 2.0
        else:
            raise ValueError(tr)
        q = np.rint(u * LEVELS)

    q = q.astype(np.uint8)
    q[nan_mask] = NODATA_U8
    return q


def _load_grid_coords(path):
    """Lädt Gitterkoordinaten aus CLAT-/CLON-GRIB2-Datei (in Grad)."""
    ds_grid = cfgrib.open_dataset(path)
    varname = list(ds_grid.data_vars)[0]
    values = np.asarray(ds_grid[varname].values).ravel()
    ds_grid.close()

    if np.nanmax(np.abs(values)) <= (np.pi + 0.1):
        values = np.rad2deg(values)
    return values


# ------------------------------
# Gitter laden + auf Ausschnitt zuschneiden
# ------------------------------
lats_all = _load_grid_coords(clat_path)
lons_all = _load_grid_coords(clon_path)
lons_all = np.where(lons_all > 180.0, lons_all - 360.0, lons_all)
n_grid = lats_all.shape[0]

# Puffer um den Ausschnitt, damit am Rand keine Löcher durch die konvexe
# Hülle entstehen (ICON-Gitter ist irregulär)
margin = 2.0  # Grad
sel = (
    np.isfinite(lons_all) & np.isfinite(lats_all) &
    (lons_all >= GRID_LON_MIN - margin) & (lons_all <= GRID_LON_MAX + margin) &
    (lats_all >= GRID_LAT_MIN - margin) & (lats_all <= GRID_LAT_MAX + margin)
)
sel_idx = np.nonzero(sel)[0]          # Indizes ins volle Gitter (für die Daten)
points = np.column_stack((lons_all[sel_idx] * COS_LAT0, lats_all[sel_idx]))

print(f"Gitterpunkte nach Zuschnitt: {len(sel_idx)} von {n_grid}")
if len(sel_idx) < 3:
    raise RuntimeError("Zu wenige Gitterpunkte im Ausschnitt - stimmen clat/clon?")

print("Baue Triangulation für Interpolation & Hüllen-Maske auf ...")
tri = Delaunay(points, qhull_options="Qbb Qc Qz Qt")


def locate(query_points):
    """Umgebendes Dreieck + baryzentrische Gewichte für beliebige Punkte.
    Rückgabe: inside (bool), verts (n_inside, 3), weights (n_inside, 3)."""
    simplex = tri.find_simplex(query_points)
    inside = simplex >= 0
    s = simplex[inside]
    T = tri.transform[s]                                   # (n, 3, 2)
    delta = query_points[inside] - T[:, 2]
    b = np.einsum("nij,nj->ni", T[:, :2], delta)           # (n, 2)
    w = np.column_stack((b, 1.0 - b.sum(axis=1)))          # (n, 3)
    verts = tri.simplices[s]                               # (n, 3)

    # entartete Dreiecke (NaN-Gewichte) wie "außerhalb" behandeln
    ok = np.all(np.isfinite(w), axis=1)
    if not np.all(ok):
        idx_inside = np.nonzero(inside)[0]
        inside[idx_inside[~ok]] = False
        verts, w = verts[ok], w[ok]
    return inside, verts, w


if var_type == "ww":
    # Wettercodes dürfen nicht gemittelt werden -> nächster Gitterpunkt.
    # Gerechnet wird auf einem groben Blockraster (WW_BLOCK x WW_BLOCK Pixel),
    # das danach per Replikation auf die volle Auflösung hochskaliert wird.
    # Die Blöcke sind in Web-Mercator quadratisch (Zeilenhöhe in Pixeln
    # wächst nach Norden), damit sie auf der Karte quadratisch aussehen.
    tree = cKDTree(points)

    col_block = np.arange(GRID_WIDTH) // WW_BLOCK
    n_cb = int(col_block.max()) + 1
    lon_c = np.array([lon_t[col_block == i].mean() for i in range(n_cb)])

    merc_y = np.log(np.tan(np.pi / 4 + np.radians(lat_t) / 2))        # absteigend (Nord -> Süd)
    dy_block = np.radians(WW_BLOCK * (lon_t[1] - lon_t[0]))           # Blockhöhe = Blockbreite
    row_block = np.floor((merc_y[0] - merc_y) / dy_block).astype(int)
    n_rb = int(row_block.max()) + 1
    y_c = np.array([merc_y[row_block == i].mean() for i in range(n_rb)])
    lat_c = np.degrees(2 * np.arctan(np.exp(y_c)) - np.pi / 2)

    lon_cc, lat_cc = np.meshgrid(lon_c, lat_c)                        # (n_rb, n_cb)
    coarse_points = np.column_stack((lon_cc.ravel() * COS_LAT0, lat_cc.ravel()))
    _, ww_nn_idx = tree.query(coarse_points)
    ww_inside = tri.find_simplex(coarse_points) >= 0
else:
    inside, verts, weights = locate(target_points)


def to_target_grid(values_native):
    """Werte auf dem (zugeschnittenen) Dreiecksgitter (1D) -> Zielraster
    (HEIGHT, WIDTH), row0 = Norden. Außerhalb der Gitter-Hülle = NaN."""
    values_native = np.asarray(values_native, dtype=np.float64)

    if var_type == "ww":
        coarse = np.where(ww_inside, values_native[ww_nn_idx], np.nan)
        coarse = coarse.reshape(n_rb, n_cb)
        return coarse[np.ix_(row_block, col_block)]

    out = np.full(n_target, np.nan, dtype=np.float64)
    out[inside] = np.einsum("ni,ni->n", values_native[verts], weights)
    return out.reshape(GRID_HEIGHT, GRID_WIDTH)


# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted([f for f in os.listdir(data_dir) if f.endswith(".grib2")])

meta = {
    "var": var_type,
    "unit": enc["unit"],
    "dtype": "uint8",
    "nodata": NODATA_U8,
    "levels": LEVELS,
    "transform": enc["transform"],
    "vmin": enc["vmin"],
    "vmax": enc["vmax"],
    "decode": {
        "linear": "v = vmin + (q/254) * (vmax - vmin)",
        "sqrt": "v = vmin + (q/254)^2 * (vmax - vmin)",
        "ssqrt": "u = q/254*2 - 1; v = sign(u) * u^2 * vmax",
        "raw": "v = q",
    }[enc["transform"]],
    "orientation": "row0 = Norden, row-major, uint8",
    "projection": "EPSG:4326",
    "extent_4326": [GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX],
    "width": GRID_WIDTH,
    "height": GRID_HEIGHT,
    "files": [],
}

for filename in all_files_global:
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # --------------------------
    # Daten je Typ (auf dem Dreiecksgitter)
    # --------------------------
    if var_type == "t2m":
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            ds.close()
            continue
        data = ds["t2m"].values - 273.15
    elif var_type == "tp":
        if "tprate" not in ds:
            print(f"Keine tprate in {filename}")
            ds.close()
            continue
        data = ds["tprate"].values
    elif var_type == "tp_acc":
        if "tp" not in ds:
            print(f"Keine tp in {filename}")
            ds.close()
            continue
        data = ds["tp"].values
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn.lower() in ["ww", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            ds.close()
            continue
        data = ds[varname].values
    elif var_type == "cape_ml":
        if "CAPE_ML" not in ds:
            print(f"Keine CAPE_ML in {filename}")
            ds.close()
            continue
        data = ds["CAPE_ML"].values
        data[data < 0] = np.nan
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds:
            print(f"Keine DBZ_CMAX in {filename}")
            ds.close()
            continue
        data = ds["DBZ_CMAX"].values
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine fg10 in {filename}")
            ds.close()
            continue
        data = ds["fg10"].values * 3.6  # m/s -> km/h
        data[data < 0] = np.nan
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde in {filename}")
            ds.close()
            continue
        data = ds["sde"].values * 100  # -> cm
        data[data < 0] = np.nan
    else:
        print(f"Unbekannter var_type: {var_type}")
        ds.close()
        continue

    data = np.asarray(data)
    if data.ndim == 3:
        data = data[0]
    data = data.ravel()

    # --------------------------
    # Zeiten
    # --------------------------
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0]) if "step" in ds else pd.to_timedelta(0)
        valid_time_utc = run_time_utc + step if run_time_utc is not None else None

    if valid_time_utc is None:
        print(f"{filename}: keine Gültigkeitszeit ermittelbar, überspringe")
        ds.close()
        continue

    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    ds.close()

    # --------------------------
    # Dreiecksgitter -> Zielraster -> uint8
    # --------------------------
    if data.shape[0] != n_grid:
        print(f"{filename}: Anzahl Datenpunkte ({data.shape[0]}) passt nicht zum Grid ({n_grid}), überspringe")
        continue

    grid_data = to_target_grid(data[sel_idx])     # row0 = Norden, NaN außerhalb der Hülle
    q = encode_uint8(grid_data, enc)

    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}"
    out_path = os.path.join(output_dir, outname)
    np.ascontiguousarray(q, dtype=np.uint8).tofile(out_path)

    meta["files"].append({
        "file": outname,
        "valid_time_utc": valid_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_local": valid_time_local.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    print(f"{filename} -> {outname} ({GRID_WIDTH}x{GRID_HEIGHT}, {q.nbytes} Bytes)")

    del data, grid_data, q
    gc.collect()

print("META: " + json.dumps(meta, ensure_ascii=False))
print("Fertig!")
import sys
import cfgrib
import pandas as pd
import os
import struct
import zlib
from zoneinfo import ZoneInfo
from scipy.ndimage import gaussian_filter
import numpy as np
import gc
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay
from PIL import Image
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "output"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3].lower()# 't2m', 'tp', 'ww', 'cape_ml', 'dbz_cmax', 'wind', etc.
grid_dir = sys.argv[4] if len(sys.argv) > 4 else "data/grid"

os.makedirs(output_dir, exist_ok=True)

# ICON-RUC läuft auf dem nativen Dreiecksgitter - die Gitterkoordinaten
# müssen separat aus den von DWD bereitgestellten CLAT-/CLON-GRIB2-Dateien
# geladen werden.
clat_path = os.path.join(grid_dir, "clat.grib2")
clon_path = os.path.join(grid_dir, "clon.grib2")

for gp in (clat_path, clon_path):
    if not os.path.exists(gp):
        raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gp}")

# ------------------------------
# WW-Farben
# ------------------------------
ww_colors_base = {
    0: "#FFFFFF", 1: "#D3D3D3", 2: "#A9A9A9", 3: "#696969",
    45: "#FFFF00", 48: "#FFD700",
    56: "#FFA500", 57: "#C06A00",
    51: "#00FF00", 53: "#00C300", 55: "#009700",
    61: "#00FF00", 63: "#00C300", 65: "#009700",
    80: "#00FF00", 81: "#00C300", 82: "#009700",
    66: "#FF6347", 67: "#8B0000",
    71: "#ADD8E6", 73: "#6495ED", 75: "#00008B",
    77: "#ADD8E6", 85: "#6495ED", 86: "#00008B",
    95: "#FF77FF", 96: "#C71585", 99: "#C71585"
}

ignore_codes = {4}

# ------------------------------
# Temperatur-Farben
# ------------------------------
t2m_bounds = list(range(-36, 50, 2))
t2m_colors = LinearSegmentedColormap.from_list(
    "t2m_smoooth",
    [
        "#F675F4", "#F428E9", "#B117B5", "#950CA2", "#640180",
        "#3E007F", "#00337E", "#005295", "#1292FF", "#49ACFF",
        "#8FCDFF", "#B4DBFF", "#B9ECDD", "#88D4AD", "#07A125",
        "#3FC107", "#9DE004", "#E7F700", "#F3CD0A", "#EE5505",
        "#C81904", "#AF0E14", "#620001", "#C87879", "#FACACA",
        "#E1E1E1", "#6D6D6D"
    ],
    N=len(t2m_bounds)
)
t2m_norm = BoundaryNorm(t2m_bounds, ncolors=len(t2m_bounds))

# ------------------------------
# Niederschlags-Farben (tp)
# ------------------------------
prec_bounds = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4"
])
prec_colors.set_under(alpha=0)
prec_norm = mcolors.BoundaryNorm(prec_bounds, prec_colors.N)

# ------------------------------
# Aufsummierter Niederschlag (tp_acc)
# ------------------------------
tp_acc_bounds = [0.0, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                  125, 150, 175, 200, 250, 300, 400, 500]
tp_acc_colors = ListedColormap([
    "#FFFFFF", "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4", "#969696"
])
tp_acc_norm = mcolors.BoundaryNorm(tp_acc_bounds, tp_acc_colors.N)

# ------------------------------
# CAPE-Farben
# ------------------------------
cape_bounds = [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
cape_colors = ListedColormap([
    "#676767", "#006400", "#008000", "#00CC00", "#66FF00", "#FFFF00",
    "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000", "#FF0095",
    "#FC439F", "#FF88D3", "#FF99FF"
])
cape_norm = mcolors.BoundaryNorm(cape_bounds, cape_colors.N)

# ------------------------------
# DBZ-CMAX Farben
# ------------------------------
dbz_bounds = [0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 67, 70]
dbz_colors = ListedColormap([
    "#676767", "#FFFFFF", "#B3EFED", "#8CE7E2", "#00F5ED",
    "#00CEF0", "#01AFF4", "#028DF6", "#014FF7", "#0000F6",
    "#00FF01", "#01DF00", "#00D000", "#00BF00", "#00A701",
    "#019700", "#FFFF00", "#F9F000", "#EDD200", "#E7B500",
    "#FF5000", "#FF2801", "#F40000", "#EA0001", "#CC0000",
    "#FFC8FF", "#E9A1EA", "#D379D3", "#BE55BE", "#960E96"
])
dbz_colors.set_under(alpha=0)
dbz_norm = mcolors.BoundaryNorm(dbz_bounds, dbz_colors.N)

# ------------------------------
# Windböen-Farben
# ------------------------------
wind_bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300]
wind_colors = ListedColormap([
    "#68AD05", "#8DC00B", "#B1D415", "#D5E81C", "#FBFC22",
    "#FAD024", "#F9A427", "#FC7929", "#FB4D2B", "#EA2B57",
    "#FB22A5", "#FC22CE", "#FC22F5", "#FC62F8", "#FD80F8",
    "#FFBFFC", "#FEDFFE", "#FEFFFF", "#E1E0FF", "#C3C3FF",
    "#A5A5FF", "#A5A5FF", "#6868FE"
])
wind_norm = mcolors.BoundaryNorm(wind_bounds, wind_colors.N)

# ------------------------------
# Schneehöhen-Farben
# ------------------------------
snow_bounds = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 400]
snow_colors = ListedColormap([
    "#F8F8F8", "#DCDBFA", "#AAA9C8", "#75BAFF", "#349AFF", "#0582FF",
    "#0069D2", "#004F9C", "#01327F", "#4B007F", "#64007F", "#9101BB",
    "#C300FC", "#D235FF", "#EBA6FF", "#F4CEFF", "#FAB2CA", "#FF9798",
    "#FE6E6E", "#DF093F", "#BE0000", "#A40000", "#880000", "#460000"
])
snow_norm = mcolors.BoundaryNorm(snow_bounds, snow_colors.N)

# Bounding Box ICON
extent = [-4.1616, 20.5444, 43.0440, 58.1647]  # lon_min, lon_max, lat_min, lat_max

FOOTER_TEXTS = {
    "ww": "Signifikantes Wetter",
    "t2m": "Temperatur 2m (°C)",
    "tp": "Niederschlag, 1Std (mm)",
    "tp_acc": "Akkumulierter Niederschlag (mm)",
    "cape_ml": "CAPE-Index (J/kg)",
    "dbz_cmax": "Sim. max. Radarreflektivität (dBZ)",
    "wind": "Windböen (km/h)",
    "snow": "Schneehöhe (cm)",
}

VALUE_UNITS = {
    "ww": "",
    "t2m": "°C",
    "tp": "mm",
    "tp_acc": "mm",
    "cape_ml": "J/kg",
    "dbz_cmax": "dBZ",
    "wind": "km/h",
    "snow": "cm",
}

VALUE_DECIMALS = {
    "ww": 0,
    "t2m": 1,
    "tp": 1,
    "tp_acc": 1,
    "cape_ml": 0,
    "dbz_cmax": 0,
    "wind": 0,
    "snow": 1,
}

VALUE_NODATA = -9999.0

# ------------------------------
# EPSG:4326 -> EPSG:3857 (Web Mercator)
# ------------------------------
EARTH_RADIUS = 6378137.0
WEBMERCATOR_WIDTH = 1024

def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y

def webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH):
    lon_min, lon_max, lat_min, lat_max = extent
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_height = max(int(round(out_width * aspect)), 1)
    x_new = np.linspace(x_min, x_max, out_width)
    y_new = np.linspace(y_min, y_max, out_height)
    return x_new, y_new

# Domain Extent in Web Mercator berechnen (für Manifest)
_dom_x_min, _dom_y_min = lonlat_to_webmercator(extent[0], extent[2])
_dom_x_max, _dom_y_max = lonlat_to_webmercator(extent[1], extent[3])
DOMAIN_EXTENT_3857 = [float(_dom_x_min), float(_dom_y_min), float(_dom_x_max), float(_dom_y_max)]

def data_to_rgba(data, cmap, norm):
    """Wandelt 2D-Datenarray in RGBA-uint8-Array um."""
    rgba = cmap(norm(data))
    rgba = (rgba * 255).astype(np.uint8)
    nan_mask = ~np.isfinite(data)
    rgba[nan_mask, 3] = 0
    return rgba

def save_transparent_webp(data, cmap, norm, out_path):
    rgba = data_to_rgba(data, cmap, norm)
    img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")
    img.save(out_path, format="WEBP", lossless=True, method=4)

def _load_grid_coords(path):
    """Lädt Gitterkoordinaten aus CLAT-/CLON-GRIB2-Datei."""
    ds_grid = cfgrib.open_dataset(path)
    varname = list(ds_grid.data_vars)[0]
    values = np.asarray(ds_grid[varname].values).ravel()
    ds_grid.close()

    if np.nanmax(np.abs(values)) <= (np.pi + 0.1):
        values = np.rad2deg(values)
    return values

# ------------------------------
# Gitterkoordinaten laden + auf Extent zuschneiden
# ------------------------------
lats = _load_grid_coords(clat_path)
lons = _load_grid_coords(clon_path)

# Puffer um die Extent, damit am Rand keine Löcher durch die
# konvexe Hülle entstehen (ICON-Gitter ist irregulär)
margin = 2.0  # Grad, ggf. anpassen (kleiner = schneller, aber Randrisiko)
lon_min, lon_max, lat_min, lat_max = extent
grid_mask = (
    (lons >= lon_min - margin) & (lons <= lon_max + margin) &
    (lats >= lat_min - margin) & (lats <= lat_max + margin)
)

lats = lats[grid_mask]
lons = lons[grid_mask]

print(f"Gitterpunkte nach Zuschnitt: {grid_mask.sum()} von {grid_mask.size}")

# ------------------------------
# Interpolation: LinearNDInterpolator statt Nearest+Distanz-Cutoff
# ------------------------------
# LinearNDInterpolator gibt außerhalb der konvexen Huelle der Punktwolke
# automatisch NaN zurueck -> kein Extrapolieren, kein Verwischen am Rand,
# kein manueller Distanz-Cutoff noetig.

# Web Mercator Koordinaten der Dreiecksgitterpunkte vorberechnen
lons_merc = EARTH_RADIUS * np.radians(lons)
lats_merc = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lats) / 2))
points_merc_base = np.column_stack((lons_merc, lats_merc))

# --- Sanity check der zugeschnittenen Gitterpunkte ---
finite_mask = np.all(np.isfinite(points_merc_base), axis=1)
n_bad = (~finite_mask).sum()
if n_bad:
    print(f"Warnung: {n_bad} nicht-endliche Gitterpunkte gefunden, werden entfernt.")

if not np.all(finite_mask):
    valid_idx = np.nonzero(finite_mask)[0]
    points_merc_base = points_merc_base[valid_idx]
    grid_mask_idx = np.nonzero(grid_mask)[0][valid_idx]
else:
    valid_idx = None
    grid_mask_idx = np.nonzero(grid_mask)[0]

# Triangulation einmalig aufbauen (teuerster Schritt) und fuer alle Dateien
# wiederverwenden - nur die Werte werden pro Zeitschritt ausgetauscht.
print("Baue Basis-Triangulation für Interpolation & Hüllen-Maske auf ...")
base_tri = Delaunay(points_merc_base, qhull_options="Qbb Qc Qz Qt")
interp_linear_base = LinearNDInterpolator(base_tri, np.zeros(len(points_merc_base), dtype=np.float64))

# Zielgitter (Web Mercator) ist für alle Dateien identisch -> einmal berechnen
x_new, y_new = webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH)
xx, yy = np.meshgrid(x_new, y_new)
target_points = np.column_stack((xx.ravel(), yy.ravel()))

# Huellen-Maske einmalig berechnen: Punkte ausserhalb der Dreiecksgitter-Huelle -> NaN
outside_hull = base_tri.find_simplex(target_points) < 0
outside_hull_2d = outside_hull.reshape(xx.shape)

# ------------------------------
# Eingebettete Rohdaten (DVAL-Chunk) im WebP
# ------------------------------
# WebP ist ein RIFF-Container, der beliebige zusätzliche Chunks mit eigenem
# FourCC-Tag erlaubt - konforme Reader ignorieren unbekannte Chunks einfach.
# Für t2m/wind hängen wir einen "DVAL"-Chunk mit den echten physikalischen
# Werten (nicht den Farben!) an, komprimiert mit zlib, plus die exakte
# Web-Mercator-Domäne in Metern für die pixelgenaue Rücktransformation im
# Frontend. `x_new`/`y_new` sind hier bereits das volle Zielraster
# (row0 = Süden, wie bei render_data_merc), daher genügt ein einfacher
# Index-Crop darauf.
EMBED_DATA_VARS = {"t2m", "wind"}
GERMANY_BBOX_LONLAT = [5.5, 15.3, 47.0, 55.3]  # lon_min, lon_max, lat_min, lat_max

_gbx_min, _gby_min = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[0], GERMANY_BBOX_LONLAT[2])
_gbx_max, _gby_max = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[1], GERMANY_BBOX_LONLAT[3])

_col_i0 = max(0, np.searchsorted(x_new, _gbx_min, side="left") - 1)
_col_i1 = min(len(x_new) - 1, np.searchsorted(x_new, _gbx_max, side="right"))
_row_i0 = max(0, np.searchsorted(y_new, _gby_min, side="left") - 1)
_row_i1 = min(len(y_new) - 1, np.searchsorted(y_new, _gby_max, side="right"))

GERMANY_CROP_EXTENT_3857 = [
    float(x_new[_col_i0]), float(y_new[_row_i0]),
    float(x_new[_col_i1]), float(y_new[_row_i1]),
]


def crop_to_germany(data_south_first):
    """data_south_first: 2D-Array wie render_data_merc (row0 = Süden,
    aufsteigend in Mercator-Y wie y_new). Schneidet auf die
    Deutschland-Bbox zu."""
    return data_south_first[_row_i0:_row_i1 + 1, _col_i0:_col_i1 + 1]


DVAL_FOURCC = b"DVAL"

# Quantisierungsschritt je Variable (feiner als die Anzeige-Nachkommastellen
# in VALUE_DECIMALS, damit keinerlei sichtbarer Genauigkeitsverlust entsteht).
QUANTUM_STEP = {
    "t2m": 0.05,   # °C, Anzeige mit 1 Dezimalstelle -> 0.05 ist mehr als genug
    "wind": 0.2,   # km/h, Anzeige mit 0 Dezimalstellen -> 0.2 ist mehr als genug
}
NAN_SENTINEL_I16 = -32768


def embed_data_chunk(webp_path, data, extent_3857, quantum, fourcc=DVAL_FOURCC):
    """Hängt ein rohes Datenfeld als privaten, int16-quantisierten RIFF-Chunk
    an ein WebP an.

    data: 2D-Array (float), row0 = Norden (also bereits wie fürs Bild
          gespiegelt).
    extent_3857: [x_min, y_min, x_max, y_max] in Web-Mercator-Metern -
                 exakt das Raster, auf dem `data` liegt.
    quantum: Rasterschritt in den Originaleinheiten (z.B. 0.05 für °C).
    """
    height, width = data.shape

    nan_mask = ~np.isfinite(data)
    data_filled = np.where(nan_mask, 0.0, data)  # verhindert NaN->int Warnung beim Runden/Casten
    quant = np.round(data_filled / quantum)
    # Sicherheitsclip: verhindert einen int16-Überlauf bei extremen
    # Ausreißern, ohne das eigentlich zulässige Wertespektrum
    # (t2m/wind liegen weit darunter) einzuschränken.
    quant = np.clip(quant, -32767, 32767).astype(np.int16)
    quant[nan_mask] = NAN_SENTINEL_I16

    header = struct.pack("<BBII", 2, 1, width, height)
    header += struct.pack("<4d", *extent_3857)
    header += struct.pack("<d", quantum)
    compressed = zlib.compress(np.ascontiguousarray(quant, dtype="<i2").tobytes(), level=9)
    payload = header + compressed

    size = len(payload)
    chunk = fourcc + struct.pack("<I", size) + payload
    if size % 2 == 1:
        chunk += b"\x00"  # RIFF-Padding auf gerade Länge, zählt nicht zu size

    with open(webp_path, "rb") as f:
        content = f.read()

    if content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
        raise ValueError(f"{webp_path} ist keine gültige WebP-Datei (RIFF/WEBP-Header fehlt)")

    riff_size = struct.unpack("<I", content[4:8])[0]
    new_riff_size = riff_size + len(chunk)

    with open(webp_path, "wb") as f:
        f.write(content[:4])
        f.write(struct.pack("<I", new_riff_size))
        f.write(content[8:])
        f.write(chunk)


# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted([f for f in os.listdir(data_dir) if f.endswith(".grib2")])

for filename in all_files_global:
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # --------------------------
    # Daten je Typ
    # --------------------------
    if var_type == "t2m":
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            ds.close()
            continue
        data = ds["t2m"].values - 273.15
        cmap, norm = t2m_colors, t2m_norm
    elif var_type == "tp":
        if "tprate" not in ds:
            print(f"Keine tprate in {filename}")
            ds.close()
            continue
        data = ds["tprate"].values
        data[data < 0.1] = np.nan
        cmap, norm = prec_colors, prec_norm
    elif var_type == "tp_acc":
        if "tp" not in ds:
            print(f"Keine tp in {filename}")
            ds.close()
            continue
        data = ds["tp"].values
        data[data < 0.1] = np.nan
        cmap, norm = tp_acc_colors, tp_acc_norm
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn in ["WW", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            ds.close()
            continue
        data = ds[varname].values
        cmap, norm = None, None
    elif var_type == "cape_ml":
        if "CAPE_ML" not in ds:
            print(f"Keine CAPE_ML in {filename}")
            ds.close()
            continue
        data = ds["CAPE_ML"].values
        data[data < 0] = np.nan
        cmap, norm = cape_colors, cape_norm
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds:
            print(f"Keine DBZ_CMAX in {filename}")
            ds.close()
            continue
        data = ds["DBZ_CMAX"].values
        cmap, norm = dbz_colors, dbz_norm
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine fg10 in {filename}")
            ds.close()
            continue
        data = ds["fg10"].values * 3.6  # m/s -> km/h
        data[data < 0] = np.nan
        cmap, norm = wind_colors, wind_norm
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde in {filename}")
            ds.close()
            continue
        data = ds["sde"].values * 100  # -> cm
        data[data < 0] = np.nan
        cmap, norm = snow_colors, snow_norm
    else:
        print(f"Unbekannter var_type: {var_type}")
        ds.close()
        continue

    if data.ndim == 3:
        data = data[0]

    data = data.ravel()[grid_mask_idx]

    # --------------------------
    # Zeiten
    # --------------------------
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0]) if "step" in ds else pd.to_timedelta(0)
        valid_time_utc = run_time_utc + step if run_time_utc else None

    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin")) if valid_time_utc else None

    ds.close()

    # --------------------------
    # Farb-Mapping je Typ (auf Dreiecksgitter-Daten)
    # --------------------------
    if var_type == "ww":
        valid_mask = np.isfinite(data)
        codes = np.unique(data[valid_mask]).astype(int)
        codes = [c for c in codes if c in ww_colors_base and c not in ignore_codes]
        codes.sort()
        cmap = ListedColormap([ww_colors_base[c] for c in codes]) if codes else ListedColormap(["#FFFFFF00"])
        norm = mcolors.Normalize(vmin=-0.5, vmax=max(len(codes) - 0.5, 0.5))
        code2idx = {c: i for i, c in enumerate(codes)}
        idx_data = np.full_like(data, fill_value=np.nan, dtype=float)
        for c, i in code2idx.items():
            idx_data[data == c] = i
        render_data = idx_data
    else:
        render_data = data.copy()

    # --------------------------
    # Interpolation auf Web-Mercator-Zielgitter
    # --------------------------
    if render_data.shape[0] != points_merc_base.shape[0]:
        print(f"{filename}: Anzahl Datenpunkte passt nicht zum Grid, überspringe")
        continue

    if var_type == "ww":
        # Kategoriale Codes: erst auf ein grobes reguläres Gitter interpolieren,
        # danach per Nearest-Neighbor hochskalieren -> echte quadratische Blöcke
        # statt der Dreiecksform des nativen ICON-Gitters.
        valid_mask = np.isfinite(render_data)
        if not np.any(valid_mask):
            print(f"{filename}: Keine gültigen Daten")
            continue

        coarse_factor = 8  # größer = gröbere/deutlichere Vierecke, kleiner = feiner
        coarse_width = max(WEBMERCATOR_WIDTH // coarse_factor, 1)
        x_coarse, y_coarse = webmercator_target_grid(extent, out_width=coarse_width)
        xx_c, yy_c = np.meshgrid(x_coarse, y_coarse)
        coarse_points = np.column_stack((xx_c.ravel(), yy_c.ravel()))

        interpolator_nn = NearestNDInterpolator(
            points_merc_base[valid_mask], render_data[valid_mask]
        )
        coarse_result = interpolator_nn(coarse_points).reshape(xx_c.shape)

        outside_hull_coarse = base_tri.find_simplex(coarse_points) < 0
        coarse_result[outside_hull_coarse.reshape(xx_c.shape)] = np.nan

        scale_y = xx.shape[0] / coarse_result.shape[0]
        scale_x = xx.shape[1] / coarse_result.shape[1]
        row_idx = np.clip((np.arange(xx.shape[0]) / scale_y).astype(int), 0, coarse_result.shape[0] - 1)
        col_idx = np.clip((np.arange(xx.shape[1]) / scale_x).astype(int), 0, coarse_result.shape[1] - 1)
        render_data_merc = coarse_result[np.ix_(row_idx, col_idx)]
    else:
        # Kontinuierliche Felder: wiederverwendete Triangulation, nur Werte tauschen.
        # Ausserhalb der Huelle liefert LinearNDInterpolator ohnehin automatisch NaN.
        interp_linear_base.values[:, 0] = render_data.astype(np.float64)
        render_data_merc = interp_linear_base(target_points).reshape(xx.shape)

    # Huellen-Maske anwenden (für ww notwendig, für linear redundant aber unschädlich)
    render_data_merc[outside_hull_2d] = np.nan

    # --------------------------
    # WebP speichern
    # --------------------------
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.webp" if valid_time_local else f"{var_type}_unknown.webp"
    out_path = os.path.join(output_dir, outname)
    save_transparent_webp(render_data_merc, cmap, norm, out_path)

    # Für t2m/wind zusätzlich die echten physikalischen Werte (°C bzw.
    # km/h, nicht die Farben) als privaten RIFF-Chunk direkt ins WebP
    # einbetten - row0 = Norden, damit der Chunk 1:1 zur Bildorientierung
    # passt (das Bild wird in save_transparent_webp beim Speichern
    # gespiegelt, render_data_merc selbst hat row0 = Süden).
    if var_type in EMBED_DATA_VARS:
        germany_data = crop_to_germany(render_data_merc)          # row0 = Süden
        quantum = QUANTUM_STEP.get(var_type, 0.1)
        embed_data_chunk(out_path, germany_data[::-1], GERMANY_CROP_EXTENT_3857, quantum)  # row0 = Norden

    print(f"{filename} -> {outname}")

    # Aufräumen
    del data, render_data, render_data_merc
    gc.collect()

print("Fertig!")

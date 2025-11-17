import sys
import cfgrib
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
from adjustText import adjust_text
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from zoneinfo import ZoneInfo
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy.interpolate import NearestNDInterpolator
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]
output_dir = sys.argv[2]
var_type = sys.argv[3].lower()
gridfile = sys.argv[4] if len(sys.argv) > 4 else "data/grid/grid.nc"

if not os.path.exists(gridfile):
    raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gridfile}")
    
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Geo-Daten
# ------------------------------
cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden',
             'Stuttgart', 'Düsseldorf', 'Nürnberg', 'Erfurt', 'Leipzig',
             'Bremen', 'Saarbrücken', 'Hannover'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05, 48.78, 51.23,
            49.45, 50.98, 51.34, 53.08, 49.24, 52.37],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73, 9.18, 6.78,
            11.08, 11.03, 12.37, 8.80, 6.99, 9.73]
})

eu_cities = pd.DataFrame({
    'name': [
        'Berlin', 'Oslo', 'Warschau',
        'Lissabon', 'Madrid', 'Rom',
        'Ankara', 'Helsinki', 'Reykjavik',
        'London', 'Paris'
    ],
    'lat': [
        52.52, 59.91, 52.23,
        38.72, 40.42, 41.90,
        39.93, 60.17, 64.13,
        51.51, 48.85
    ],
    'lon': [
        13.40, 10.75, 21.01,
        -9.14, -3.70, 12.48,
        32.86, 24.94, -21.82,
        -0.13, 2.35
    ]
})

# ------------------------------
# Farben und Normen
# ------------------------------
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
ww_categories = {
    "Bewölkung": [0, 1, 2, 3],
    "Nebel": [45],
    "Schneeregen": [56, 57],
    "Regen": [61, 63, 65],
    "gefr. Regen": [66, 67],
    "Schnee": [71, 73, 75],
    "Gewitter": [95, 96],
}

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

prec_bounds = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4", "#969696"
])
prec_norm = BoundaryNorm(prec_bounds, prec_colors.N)

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
dbz_norm = mcolors.BoundaryNorm(dbz_bounds, dbz_colors.N)

# ------------------------------
# Aufsummierter Niederschlag (tp_acc)
# ------------------------------
tp_acc_bounds = [0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                 125, 150, 175, 200, 250, 300, 400, 500]
tp_acc_colors = ListedColormap([
    "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
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
# Luftdruck
# ------------------------------

# Luftdruck-Farben (kontinuierlicher Farbverlauf für 45 Bins)
pmsl_bounds_colors = list(range(912, 1070, 4))  # Alle 4 hPa (45 Bins)
pmsl_colors = LinearSegmentedColormap.from_list(
    "pmsl_smooth",
    [
       "#FF6DFF", "#C418C4", "#950CA2", "#5A007D", "#3D007F",
       "#00337E", "#0472CB", "#4FABF8", "#A3D4FF", "#79DAAD",
       "#07A220", "#3EC008", "#9EE002", "#F3FC01", "#F19806",
       "#F74F11", "#B81212", "#8C3234", "#C87879", "#F9CBCD",
       "#E2E2E2"

    ],
    N=len(pmsl_bounds_colors)  # Genau 45 Farben für 45 Bins
)
pmsl_norm = BoundaryNorm(pmsl_bounds_colors, ncolors=len(pmsl_bounds_colors))

#-------------------------------
# Schneehöhen-Farben
#------------------------------
snow_bounds = [0, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 400]  # in cm
snow_colors = ListedColormap([
        "#F8F8F8", "#DCDBFA", "#AAA9C8", "#75BAFF", "#349AFF", "#0682FF",
        "#0069D2", "#004F9C", "#01327F", "#4B007F", "#64007F", "#9101BB",
        "#C300FC", "#D235FF", "#EBA6FF", "#F4CEFF", "#FAB2CA", "#FF9798",
        "#FE6E6E", "#DF093F", "#BE0000", "#A40000", "#880000"
    ])
snow_norm = mcolors.BoundaryNorm(snow_bounds, snow_colors.N)

# ------------------------------
# Kartenparameter
# ------------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX

# Bounding Box Deutschland (fix, keine GeoJSON nötig)
extent = [5, 16, 47, 56]

extent_eu = [-23.5, 45.0, 29.5, 68.4]

# ------------------------------
# WW-Legende Funktion
# ------------------------------
def add_ww_legend_bottom(fig, ww_categories, ww_colors_base):
    legend_height = 0.12
    legend_ax = fig.add_axes([0.05, 0.01, 0.9, legend_height])
    legend_ax.axis("off")
    for i, (label, codes) in enumerate(ww_categories.items()):
        n_colors = len(codes)
        block_width = 1.0 / len(ww_categories)
        gap = 0.05 * block_width
        x0 = i * block_width
        x1 = (i + 1) * block_width
        inner_width = x1 - x0 - gap
        color_width = inner_width / n_colors
        for j, c in enumerate(codes):
            color = ww_colors_base.get(c, "#FFFFFF")
            legend_ax.add_patch(mpatches.Rectangle((x0 + j * color_width, 0.3),
                                                  color_width, 0.6,
                                                  facecolor=color, edgecolor='black'))
        legend_ax.text((x0 + x1)/2, 0.05, label, ha='center', va='bottom', fontsize=10)

# ------------------------------
# ICON Grid laden (einmal!)
# ------------------------------
nc = netCDF4.Dataset(gridfile)  # Datei öffnen
lats = np.rad2deg(nc.variables["clat"][:])
lons = np.rad2deg(nc.variables["clon"][:])
nc.close()

# ------------------------------
# Dateien durchgehen
# ------------------------------
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".grib2"):
        continue
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # --------------------------
    # Daten je Typ
    # --------------------------
    if var_type == "t2m":
        if "t2m" not in ds: continue
        data = ds["t2m"].values - 273.15
        cmap, norm = t2m_colors, t2m_norm
    elif var_type == "t2m_eu":
        if "t2m" not in ds: continue
        data = ds["t2m"].values - 273.15
        cmap, norm = t2m_colors, t2m_norm
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn in ["WW", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            continue
        data = ds[varname].values
        cmap = None
    elif var_type == "ww_eu":
        varname = next((vn for vn in ds.data_vars if vn in ["WW", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            continue
        data = ds[varname].values
        cmap = None
    elif var_type == "pmsl":
        if "prmsl" not in ds:
            print(f"Keine prmsl-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["prmsl"].values / 100
        data[data < 0] = np.nan
        cmap, norm = pmsl_colors, pmsl_norm
    elif var_type == "pmsl_eu":
        if "prmsl" not in ds:
            print(f"Keine prmsl-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["prmsl"].values / 100
        data[data < 0] = np.nan
        cmap, norm = pmsl_colors, pmsl_norm
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde-Variable in {filename}")
            continue
        data = ds["sde"].values
        data[data < 0] = np.nan
        data = data * 100  # in cm umrechnen
        cmap, norm = snow_colors, snow_norm
    elif var_type == "snow_eu":
        if "sde" not in ds:
            print(f"Keine sde-Variable in {filename}")
            continue
        data = ds["sde"].values
        data[data < 0] = np.nan
        data = data * 100  # in cm umrechnen
        cmap, norm = snow_colors, snow_norm
    else:
        print(f"Var_type {var_type} nicht implementiert")
        continue

    if data.ndim == 3: data = data[0]

    # --------------------------
    # Zeiten
    # --------------------------
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0])
        valid_time_utc = run_time_utc + step
    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    # --------------------------
    # Figure
    # --------------------------
    if var_type in ["pmsl_eu", "ww_eu", "t2m_eu", "snow_eu"]:
        scale = 0.9
        fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
        shift_up = 0.02
        ax = fig.add_axes([0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
                            projection=ccrs.PlateCarree())
        ax.set_extent(extent_eu)
        ax.set_axis_off()
        ax.set_aspect('auto')
    else:
        scale = 0.9
        fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
        shift_up = 0.02
        ax = fig.add_axes([0.0, BOTTOM_AREA_PX/FIG_H_PX + shift_up, 1.0, TOP_AREA_PX/FIG_H_PX],
                        projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        ax.set_axis_off()
        ax.set_aspect('auto')

    # ------------------------------
    # Regelmäßiges Gitter definieren
    # ------------------------------
    if var_type in ["pmsl_eu", "t2m_eu", "ww_eu", "snow_eu"]:
        res = 0.1   # gröber für Europa (~11 km)
        lon_min, lon_max, lat_min, lat_max = extent_eu
        buffer = res * 20
        nx = int(round(lon_max - lon_min) / res) + 1
        ny = int(round(lat_max - lat_min) / res) + 1
        lon_grid = np.linspace(lon_min - buffer, lon_max + buffer, nx + 15)
        lat_grid = np.linspace(lat_min - buffer, lat_max + buffer, ny + 15)
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    else:
        lon_min, lon_max, lat_min, lat_max = extent
        res = 0.1  # Auflösung in Grad (anpassbar, z. B. 0.05 für höhere Auflösung)
        if var_type == "ww":
            res = 0.15
        elif var_type == "pmsl":
            res = 0.025
        else:
            res = 0.03
        lon_grid = np.arange(lon_min, lon_max + res, res)
        lat_grid = np.arange(lat_min, lat_max + res, res)
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # ------------------------------
    # Interpolation auf regelmäßiges Gitter
    # ------------------------------
    points = np.column_stack((lons, lats))
    valid_mask = np.isfinite(data)
    points_valid = points[valid_mask]
    data_valid = data[valid_mask]

    # Nearest Neighbor Interpolation (schnell und ausreichend für viele Fälle)
    interpolator = NearestNDInterpolator(points_valid, data_valid)
    data_grid = interpolator(lon_grid, lat_grid)

    # ------------------------------
    # pcolormesh Plot
    # ------------------------------
    if cmap is not None:
        # Für Variablen mit vorgegebener Farbkarte (t2m, tp, dbz_cmax, tp_acc, cape_ml)
        im = ax.pcolormesh(lon_grid, lat_grid, data_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        if var_type == "t2m":
            data_smooth = gaussian_filter (data_grid, sigma = 2.0)
            im = ax.pcolormesh(lon_grid, lat_grid, data_smooth, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        elif var_type == "t2m_eu":
            data_smooth = gaussian_filter (data_grid, sigma = 2.0)
            im = ax.pcolormesh(lon_grid, lat_grid, data_smooth, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        elif var_type == "snow":
            data_smooth = gaussian_filter (data_grid, sigma = 2.0)
            im = ax.pcolormesh(lon_grid, lat_grid, data_smooth, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        elif var_type == "snow_eu":
            data_smooth = gaussian_filter (data_grid, sigma = 2.0)
            im = ax.pcolormesh(lon_grid, lat_grid, data_smooth, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        
        elif var_type == "pmsl":
            # --- Luftdruck auf Meereshöhe (Deutschland) ---
            data_smooth = gaussian_filter (data_grid, sigma = 2.0)
            im = ax.pcolormesh(lon_grid, lat_grid, data_smooth, cmap=pmsl_colors, norm=pmsl_norm, shading="auto")
            data_hpa = data_smooth  # Daten liegen bereits in hPa vor

            # Haupt-Isobaren (alle 4 hPa)
            main_levels = list(range(912, 1070, 4))
            # Feine Isobaren (alle 1 hPa)
            fine_levels = list(range(912, 1070, 1))

            # Nur Levels zeichnen, die im Datenbereich liegen
            main_levels = [lev for lev in main_levels if data_hpa.min() <= lev <= data_hpa.max()]
            fine_levels = [lev for lev in fine_levels if data_hpa.min() <= lev <= data_hpa.max()]

            # Feine Isobaren (weiß, dünn, leicht transparent)
            ax.contour(
                lon_grid, lat_grid, data_hpa,
                levels=fine_levels,
                colors='gray', linewidths=0.5, alpha=0.4
            )

            # Haupt-Isobaren (weiß, etwas dicker)
            cs_main = ax.contour(
                lon_grid, lat_grid, data_hpa,
                levels=main_levels,
                colors='white', linewidths=0.8, alpha=0.9
            )

            # Isobaren-Beschriftung (Zahlen direkt auf Linien)
            ax.clabel(cs_main, inline=True, fmt='%d', fontsize=9, colors='black')

            # --- Extremwerte (Tief & Hoch) markieren, aber nur wenn im Extent ---
            min_idx = np.unravel_index(np.nanargmin(data_hpa), data_hpa.shape)
            max_idx = np.unravel_index(np.nanargmax(data_hpa), data_hpa.shape)
            min_val = data_hpa[min_idx]
            max_val = data_hpa[max_idx]

            lon_min, lon_max, lat_min, lat_max = extent

            # Tiefdruckzentrum
            lat_i, lon_i = min_idx
            lon_minpt = lon_grid[lat_i, lon_i]
            lat_minpt = lat_grid[lat_i, lon_i]
            if lon_min <= lon_minpt <= lon_max and lat_min <= lat_minpt <= lat_max:
                ax.text(
                    lon_minpt, lat_minpt,
                    f"{min_val:.0f}",
                    color='white', fontsize=11, fontweight='bold',
                    ha='center', va='center',
                    transform=ccrs.PlateCarree(),
                    clip_on=True,
                    path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')]
                )

            # Hochdruckzentrum
            lat_j, lon_j = max_idx
            lon_maxpt = lon_grid[lat_j, lon_j]
            lat_maxpt = lat_grid[lat_j, lon_j]
            if lon_min <= lon_maxpt <= lon_max and lat_min <= lat_maxpt <= lat_max:
                ax.text(
                    lon_maxpt, lat_maxpt,
                    f"{max_val:.0f}",
                    color='white', fontsize=11, fontweight='bold',
                    ha='center', va='center',
                    transform=ccrs.PlateCarree(),
                    clip_on=True,
                    path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')]
                )
        elif var_type == "pmsl_eu":
            # --- Luftdruck auf Meereshöhe (Deutschland) ---
            data_smooth = gaussian_filter (data_grid, sigma = 2.0)
            im = ax.pcolormesh(lon_grid, lat_grid, data_smooth, cmap=pmsl_colors, norm=pmsl_norm, shading="auto")
            data_hpa = data_smooth  # Daten liegen bereits in hPa vor

            # Haupt-Isobaren (alle 4 hPa)
            main_levels = list(range(912, 1070, 4))
            # Feine Isobaren (alle 1 hPa)
            fine_levels = list(range(912, 1070, 1))

            # Nur Levels zeichnen, die im Datenbereich liegen
            main_levels = [lev for lev in main_levels if data_hpa.min() <= lev <= data_hpa.max()]
            fine_levels = [lev for lev in fine_levels if data_hpa.min() <= lev <= data_hpa.max()]

            # Feine Isobaren (weiß, dünn, leicht transparent)
            ax.contour(
                lon_grid, lat_grid, data_hpa,
                levels=fine_levels,
                colors='gray', linewidths=0.5, alpha=0.4
            )

            # Haupt-Isobaren (weiß, etwas dicker)
            cs_main = ax.contour(
                lon_grid, lat_grid, data_hpa,
                levels=main_levels,
                colors='white', linewidths=0.8, alpha=0.9
            )

            # Isobaren-Beschriftung (Zahlen direkt auf Linien)
            ax.clabel(cs_main, inline=True, fmt='%d', fontsize=9, colors='black')

            # --- Extremwerte (Tief & Hoch) markieren, aber nur wenn im Extent ---
            min_idx = np.unravel_index(np.nanargmin(data_hpa), data_hpa.shape)
            max_idx = np.unravel_index(np.nanargmax(data_hpa), data_hpa.shape)
            min_val = data_hpa[min_idx]
            max_val = data_hpa[max_idx]

            lon_min, lon_max, lat_min, lat_max = extent

            # Tiefdruckzentrum
            lat_i, lon_i = min_idx
            lon_minpt = lon_grid[lat_i, lon_i]
            lat_minpt = lat_grid[lat_i, lon_i]
            if lon_min <= lon_minpt <= lon_max and lat_min <= lat_minpt <= lat_max:
                ax.text(
                    lon_minpt, lat_minpt,
                    f"{min_val:.0f}",
                    color='white', fontsize=11, fontweight='bold',
                    ha='center', va='center',
                    transform=ccrs.PlateCarree(),
                    clip_on=True,
                    path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')]
                )

            # Hochdruckzentrum
            lat_j, lon_j = max_idx
            lon_maxpt = lon_grid[lat_j, lon_j]
            lat_maxpt = lat_grid[lat_j, lon_j]
            if lon_min <= lon_maxpt <= lon_max and lat_min <= lat_maxpt <= lat_max:
                ax.text(
                    lon_maxpt, lat_maxpt,
                    f"{max_val:.0f}",
                    color='white', fontsize=11, fontweight='bold',
                    ha='center', va='center',
                    transform=ccrs.PlateCarree(),
                    clip_on=True,
                    path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')]
                )
    else:
        # WW-Farben
        valid_mask = np.isfinite(data)
        codes = np.unique(data[valid_mask]).astype(int)
        codes = [c for c in codes if c in ww_colors_base]
        codes.sort()
        cmap = ListedColormap([ww_colors_base[c] for c in codes])
        code2idx = {c: i for i, c in enumerate(codes)}
        idx_data = np.full_like(data_grid, fill_value=np.nan, dtype=float)
        for c, i in code2idx.items():
            idx_data[data_grid == c] = i
        im = ax.pcolormesh(lon_grid, lat_grid, idx_data, cmap=cmap, vmin=-0.5, vmax=len(codes)-0.5, transform=ccrs.PlateCarree())

    # Bundesländer-Grenzen aus Cartopy (statt GeoJSON)
    if var_type in ["pmsl_eu", "t2m_eu", "ww_eu", "snow_eu"]:
        # 🌍 Europa: nur Ländergrenzen + europäische Städte
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="black", linewidth=0.7)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black", linewidth=0.7)

        for _, city in eu_cities.iterrows():
            ax.plot(city["lon"], city["lat"], "o", markersize=6,
                    markerfacecolor="black", markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5)
            txt = ax.text(city["lon"] + 0.3, city["lat"] + 0.3, city["name"],
                          fontsize=9, color="black", weight="bold", zorder=6)
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

    else:
        # 🇩🇪 Deutschland: Bundesländer, Grenzen und Städte
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="#2C2C2C", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")

        for _, city in cities.iterrows():
            ax.plot(city["lon"], city["lat"], "o", markersize=6,
                    markerfacecolor="black", markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5)
            txt = ax.text(city["lon"] + 0.1, city["lat"] + 0.1, city["name"],
                          fontsize=9, color="black", weight="bold", zorder=6)
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

    ax.add_patch(mpatches.Rectangle((0,0),1,1, transform=ax.transAxes, fill=False, color="black", linewidth=2))

    # --------------------------
    # Colorbar (falls relevant)
    # --------------------------
    legend_h_px = 50
    legend_bottom_px = 45
    if var_type in ["t2m", "t2m_eu", "pmsl", "pmsl_eu", "snow", "snow_eu"]:
        bounds = t2m_bounds if var_type == "t2m" else t2m_bounds if var_type == "t2m_eu" else pmsl_bounds_colors if var_type == "pmsl" else pmsl_bounds_colors if var_type == "pmsl_eu" else snow_bounds if var_type == "snow" else snow_bounds
        cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
        cbar.ax.tick_params(colors="black", labelsize=7)
        cbar.outline.set_edgecolor("black")
        cbar.ax.set_facecolor("white")

        if var_type == "t2m":
            tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "t2m_eu":
            tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type=="pmsl":
            tick_labels = [str(tick) if tick % 8 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type=="pmsl_eu":
            tick_labels = [str(tick) if tick % 8 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type=="snow":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in snow_bounds])
        if var_type=="snow_eu":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in snow_bounds])

    else:
        add_ww_legend_bottom(fig, ww_categories, ww_colors_base)

    # Footer
    footer_ax = fig.add_axes([0.0, (legend_bottom_px + legend_h_px)/FIG_H_PX, 1.0,
                              (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px)/FIG_H_PX])
    footer_ax.axis("off")
    footer_texts = {
        "ww": "Signifikantes Wetter",
        "ww_eu": "Signifikantes Wetter, Europa",
        "t2m": "Temperatur 2m (°C)",
        "t2m_eu": "Temperatur 2m (°C), Europa",
        "pmsl": "Luftdruck auf Meereshöhe (hPa)",
        "pmsl_eu": "Luftdruck auf Meereshöhe (hPa), Europa",
        "snow": "Schneehöhe (cm)",
        "snow_eu": "Schneehöhe (cm), Europa"
    }

    left_text = footer_texts.get(var_type, var_type) + \
                f"\nICON ({pd.to_datetime(run_time_utc).hour:02d}z), Deutscher Wetterdienst" \
                if run_time_utc is not None else \
                footer_texts.get(var_type, var_type) + "\nICON (??z), Deutscher Wetterdienst"

    footer_ax.text(0.01, 0.85, left_text, fontsize=12, fontweight="bold", va="top", ha="left")
    footer_ax.text(0.734, 0.92, "Prognose für:", fontsize=12, va="top", ha="left", fontweight="bold")
    footer_ax.text(0.99, 0.68, f"{valid_time_local:%d.%m.%Y %H:%M} Uhr",
                   fontsize=12, va="top", ha="right", fontweight="bold")

    # Speichern
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.png"
    plt.savefig(os.path.join(output_dir, outname), dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()

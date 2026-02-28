from Classes import moc, sst, innerocean
import matplotlib.pyplot as plt
import os
from pathlib import Path
import statsmodels.api as sm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from matplotlib import colors
import subprocess
from typing import List, Optional
import re
from matplotlib.colors import LinearSegmentedColormap
from cartopy.util import add_cyclic_point
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from shapely.geometry import MultiPoint, mapping
import json
import pickle
import datetime as dt
import matplotlib.ticker as ticker

try:
    import alphashape
except Exception:
    alphashape = None


def set_font_sizes():
    plt.rcParams['axes.titlesize'] = 24   
    plt.rcParams['figure.titlesize'] = 24  
    plt.rcParams['axes.labelsize'] = 18        
    plt.rcParams['legend.fontsize'] = 14         
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['figure.titleweight'] = 'bold' 
    plt.rcParams['scatter.edgecolors'] = 'w' 

set_font_sizes()

## paths

# MOC
home_moc = '/work/uo1075/u241372/ProcessedFiles/msftmz'
hist_ens = f'{home_moc}/msftmz_historical_Oyr_MPI-ESM1-2-LR_historical_e50i1p1f1_1958-2014.nc'
assimilation_atm = f'{home_moc}/msftmz_asfreeERAf_Oyr_MPI-ESM1-2-LR_asfreeERAf_r1i6p2_1958-2014.nc'
assimilation_oce_ens = f'{home_moc}/msftmz_asSEIKERAf_Oyr_MPI-ESM1-2-LR_asSEIKERAf_e16i8p4_1958-2014.nc'

# upwelling
upwelling_dict_hist_path = '/work/uo1075/u241372/upwellings/upwelling_dict_hist'
upwelling_ass_atm_path = '/work/uo1075/u241372/ProcessedFiles/msftmz/upwelling_ass_atm.nc'
upwelling_dict_ass_oce_path = '/work/uo1075/u241372/upwellings/upwelling_dict_ass_oce'


# inner ocean

home_inner = '/work/uo1075/u241372/ProcessedFiles'
reference_path = '/work/uo1075/u241372/ProcessedFiles/innerocean_obsv_EN.4.2.1.f.analysis.g10.1958-2014.nc'

# ensemble mean
hist_thetao_ens_path = f'{home_inner}/thetao/thetao_historical_Oyr_MPI-ESM1-2-LR_historical_e50i1p1f1_1958-2014_r360x180.nc'
hist_so_ens_path = f'{home_inner}/so/so_historical_Oyr_MPI-ESM1-2-LR_historical_e50i1p1f1_1958-2014_r360x180.nc'
hist_rhopoto_ens_path = f'{home_inner}/rhopoto/rhopoto_historical_Oyr_MPI-ESM1-2-LR_historical_e50i1p1f1_1958-2014_r360x180.nc'

ass_atm_thetao_path = f'{home_inner}/thetao/thetao_asfreeERAf_Oyr_MPI-ESM1-2-LR_asfreeERAf_r1i6p2_1958-2014_r360x180.nc'
ass_atm_so_path = f'{home_inner}/so/so_asfreeERAf_Oyr_MPI-ESM1-2-LR_asfreeERAf_r1i6p2_1958-2014_r360x180.nc'
ass_atm_rhopoto_path = f'{home_inner}/rhopoto/rhopoto_asfreeERAf_Oyr_MPI-ESM1-2-LR_asfreeERAf_r1i6p2_1958-2014_r360x180.nc'

ass_oce_thetao_ens_path = f'{home_inner}/thetao/thetao_asSEIKERAf_Oyr_MPI-ESM1-2-LR_asSEIKERAf_e16i8p4_1958-2014_r360x180.nc'
ass_oce_so_ens_path = f'{home_inner}/so/so_asSEIKERAf_Oyr_MPI-ESM1-2-LR_asSEIKERAf_e16i8p4_1958-2014_r360x180.nc'
ass_oce_rhopoto_ens_path = f'{home_inner}/rhopoto/rhopoto_asSEIKERAf_Oyr_MPI-ESM1-2-LR_asSEIKERAf_e16i8p4_1958-2014_r360x180.nc'

# SST
home_sst = "/work/uo1075/u241372/ProcessedFiles/tos"

files_ens = [
    f"{home_sst}/tos_historical_Oyr_MPI-ESM1-2-LR_historical_e50i1p1f1_1958-2014_r360x180.nc",
    f"{home_sst}/tos_asfreeERAf_Oyr_MPI-ESM1-2-LR_asfreeERAf_r1i6p2_1958-2014_r360x180.nc",
    f"{home_sst}/tos_asSEIKERAf_Oyr_MPI-ESM1-2-LR_asSEIKERAf_e16i8p4_1958-2014_r360x180.nc",
    f"{home_sst}/tos_obsv_reanalysis_Oyr_HadISST_r1i1p1_1958-2014_r360x180_1,1.nc"
]

# regression

regress_paths = ['/work/uo1075/u241372/Regression/scipy/ens_p/regress_hist',
                 '/work/uo1075/u241372/Regression/scipy/ens_p/regress_ass_atm',
                 '/work/uo1075/u241372/Regression/scipy/ens_p/regress_ass_oce']


## MOC

m_hist_ens = moc(hist_ens, typ='hist', suptitle = "HIST")
m_ass_atm = moc(assimilation_atm, typ='ass', suptitle = "ATM")
m_ass_oce_ens = moc(assimilation_oce_ens, typ='ass', suptitle="OCE")

## methods 

f, a = m_ass_atm.plot_panel_v3(split=True, arrows=True)

## mean MOC plots

fig, ax = m_hist_ens.plot_panel_v3(split=True, scatter=True)
fig, ax = m_ass_atm.plot_panel_v3(split=True, scatter=True)
fig, ax = m_ass_oce_ens.plot_panel_v3(split=True, scatter=True)


## upwelling pathways


def plot_baker_envelope(fig, ax, upwelling_dict, suptitle, suptitle_c, add=False):
    """
    Plot ensemble mean lines and envelope (min/max across members) for upwelling pathways.

    Parameters
    - fig, ax: matplotlib figure and axis to draw on
    - upwelling_dict: dict containing at least 'ensmean' and zero-or-more member entries.
      Each entry is expected to have attributes/columns: time, amoc_max, atlantic_Up, SO_Up, indopac_resUp
    - suptitle: figure suptitle text
    - suptitle_c: suptitle color
    """

    def build_envelope(varname):
        """Return (min_array, max_array) across all non-ensmean members for given varname.
           Returns (None, None) if no members provide the variable.
        """
        arrays = []
        for key, val in upwelling_dict.items():
            if hasattr(val, varname):
                arr = np.asarray(getattr(val, varname))
                arrays.append(arr)
        if not arrays:
            return None, None
        stacked = np.vstack(arrays)
        return np.nanmean(stacked, axis=0), np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0)

    # Colors for each pathway

    if add:
        cmap = {
            'amoc_max': 'purple',
            'atlantic_Up': 'deepskyblue',
            'SO_Up+indopac_resUp': 'y',
        }
    else:
        cmap = {
            'amoc_max': 'purple',
            'atlantic_Up': 'deepskyblue',
            'SO_Up': 'darkorange',
            'indopac_resUp': 'yellowgreen'
        }
    

    ymax = -np.inf

    key1 = list(upwelling_dict.keys())[0]
    x = upwelling_dict[key1].time
    mean_vals = []
    ens = {}
    # Plot each pathway: envelope (fill_between) + ensemble mean line
    for varname, color in cmap.items():
        ensmean, vmin, vmax = build_envelope(varname)
        ens[varname]=ensmean
        
        # fill envelope if available
        if vmin is not None and vmax is not None:
            ax.fill_between(x, vmin, vmax, color=color, alpha=0.25, linewidth=0) #, label=f'{varname} range')
            current_max = np.nanmax(vmax)
        else:
            current_max = -10

        # plot ensemble mean
        if 'ensmean' in list(upwelling_dict.keys()):
            print('ensmean')
            ensmean = upwelling_dict['ensmean'][varname]
        ax.plot(x, ensmean, c=color, linewidth=1.8, label=f'{varname} ensemble mean')
        mean_vals.append(np.nanmean(ensmean))

        
    ax.set_ylim([0, 38])
    ax.set_xlim([x[0], x[-1]])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Year')
    ax.set_ylabel('Pathway Strength (Sv)')

    #if 'Historical' not in suptitle:
    #    ax.yaxis.label.set_color('none')

    # Reduce duplicate legend entries by using unique labels
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    kept_handles = []
    kept_labels = []
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = True
            kept_handles.append(h)
            if add:
                kept_labels = ['AMOC max', 'Atlantic Up', 'Southern Ocean + Indo-Pac res Up']
            else:
                kept_labels = [f'AMOC max, mean={mean_vals[0]:.2f}', 
                               f'Atlantic Up, mean={mean_vals[1]:.2f}', 
                               f'Southern Ocean Up, mean={mean_vals[2]:.2f}', 
                               f'Indopac residual Up, mean={mean_vals[3]:.2f}']
    ax.legend(kept_handles, kept_labels, loc='upper left')

    ax.grid(which='major', linestyle='-', alpha=0.7)
    ax.grid(which='minor', linestyle=':', alpha=0.3)

    ax.text(
        -.1, 0.5, suptitle,
        color=suptitle_c,
        transform=ax.transAxes,
        rotation=90,
        va='center', ha='center',
        fontsize=28, fontweight='bold'
        )
    # fig.suptitle(suptitle, fontsize=24, fontweight='bold', color=suptitle_c)

    plt.tight_layout()
    plt.show()

    return ens

def load_xr_dict(
    outdir: str or Path,
    *,
    manifest_name: str = "manifest.json",
    decode_on_open: bool = False,
):
    """
    Load a dictionary written by save_xr_dict with format netcdf or zarr.

    Returns a dict mapping original keys to xarray objects or Python objects for pickled entries.

    - decode_on_open: if True, will call .load() (compute) after opening; otherwise returns lazy xarray objects.
    """
    outdir = Path(outdir)
    manifest_path = outdir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found")

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    restored = {}
    for key, fname in manifest.items():
        path = outdir / fname
        if fname.endswith(".nc"):
            ds = xr.open_dataset(path)
            if decode_on_open:
                ds = ds.load()
            restored[key] = ds
        elif fname.endswith(".zarr"):
            ds = xr.open_zarr(path)
            if decode_on_open:
                ds = ds.load()
            restored[key] = ds
        elif fname.endswith(".pkl"):
            with open(path, "rb") as fh:
                restored[key] = pickle.load(fh)
        else:
            # try to infer
            if path.is_dir() and path.suffix == ".zarr":
                ds = xr.open_zarr(path)
                if decode_on_open:
                    ds = ds.load()
                restored[key] = ds
            elif path.suffix == ".nc":
                ds = xr.open_dataset(path)
                if decode_on_open:
                    ds = ds.load()
                restored[key] = ds
            else:
                # last resort: attempt pickle
                try:
                    with open(path, "rb") as fh:
                        restored[key] = pickle.load(fh)
                except Exception:
                    raise RuntimeError(f"Cannot infer loader for {path}")

    return restored


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Example dictionary where each value is an xarray.Dataset
    # upwelling_dict = {"r1i1p1f1": ds1, "r2i1p1f1": ds2, ...}

    # Save as NetCDF files (one file per member)
    # manifest = save_xr_dict(upwelling_dict, "saved_upwelling_nc", format="netcdf", overwrite=True)

    # Save as Zarr stores (recommended for large/dask-backed arrays)
    # manifest = save_xr_dict(upwelling_dict, "saved_upwelling_zarr", format="zarr", overwrite=True)

    # Or pickle everything into a single file (fast, but binary; avoid untrusted sources)
    # save_xr_dict(upwelling_dict, "upwelling_all.pkl", format="pickle", overwrite=True)

    # Reload:
    # restored = load_xr_dict("saved_upwelling_nc")
    pass


# HIST
upwelling_dict_hist = load_xr_dict(upwelling_dict_hist_path)
upwelling_dict_hist.pop('ensmean')

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
suptitle_hist = 'HIST'
suptitle_c_hist = 'crimson'

ens_hist = plot_baker_envelope(fig, ax, upwelling_dict_hist, suptitle_hist, suptitle_c_hist)


# ATM
up_ass_atm = xr.open_dataset(upwelling_ass_atm_path)
up_ass_atm['time'] = up_ass_atm.time.dt.year
upwelling_dict_ass_atm = {'r1': up_ass_atm}

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
suptitle_ass_atm = 'ATM'
suptitle_c_ass_atm = 'forestgreen'

ens_atm = plot_baker_envelope(fig, ax, upwelling_dict_ass_atm, suptitle_ass_atm, suptitle_c_ass_atm)


# OCE
upwelling_dict_ass_oce = load_xr_dict(upwelling_dict_ass_oce_path)
upwelling_dict_ass_oce.pop('ensmean')

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
suptitle_ass_oce = 'OCE'
suptitle_c_ass_oce = 'mediumblue'

ens_oce = plot_baker_envelope(fig, ax, upwelling_dict_ass_oce, suptitle_ass_oce, suptitle_c_ass_oce)



## zonal mean temperature and salinity

# reference
reference = xr.open_dataset(reference_path)
reference = reference.drop_vars(
    [v for v in reference.data_vars if v not in ['temperature', 'salinity']]
)
reference['temperature'] = reference.temperature-273.15

files_ens = [hist_thetao_ens_path,
             ass_atm_thetao_path,
             ass_oce_thetao_ens_path,
             reference
]
typs = ["hist", "ass", "ass", "ass"]  # adjust as needed
titles = ["Historical Ensemble", "Assimilation Atmosphere", "Assimilation Ocean Ensemble", "EN4 Reference"]

thetao_ens = innerocean(files_ens, varname="thetao", typs=typs, suptitles=titles)

f = thetao_ens.plot_basins()
f = thetao_ens.plot_zonalmean_v2(clevels=np.arange(0,15,0.5))


figs = thetao_ens.hovmoeller_v2(lats=[-34.5,40], basins=True)



files_ens = [hist_so_ens_path,
             ass_atm_so_path,
             ass_oce_so_ens_path,
             reference
]
typs = ["hist", "ass", "ass", "ass"]
titles = ["Historical Ensemble", "Assimilation Atmosphere", "Assimilation Ocean Ensemble", "EN4 Reference"]

so_ens = innerocean(files_ens, varname="so", typs=typs, suptitles=titles)

f = so_ens.plot_zonalmean_v2(clevels=np.arange(34.3, 36, 0.1))

figs = so_ens.hovmoeller_v2(lats=[-34.5,40], basins=True) 



## SST

sst_all_ens = sst(files_ens, names=['historical_ens', 'assimilation_atm', 'assimilation_oce_ens', 'observations'])

f = sst_all_ens.plot_mean_maps(diff=True, vert=True)
f = sst_all_ens.plot_mean_maps(corr=True, vert=True)


## regression 

def _normalize_lons_to_range(lons, target_lon):
    """
    Convert lons (array-like) to the same 0..360 or -180..180 range as target_lon.
    target_lon is an array of longitudes used for plotting (e.g. lon_cyc).
    """
    lons = np.asarray(lons)
    if np.nanmin(target_lon) >= 0:
        # target uses 0..360
        return np.mod(lons, 360)
    else:
        # target uses -180..180
        lons = np.mod(lons + 180, 360) - 180
        return lons


def _make_region_polygon(df_pts, method="convex", alpha=None):
    """
    Build a shapely polygon around points in df_pts (DataFrame with 'lon' and 'lat').
    method = 'convex' (default) or 'alpha' (concave, requires alphashape).
    Returns shapely geometry (Polygon or MultiPolygon).
    """
    if "lon" not in df_pts.columns or "lat" not in df_pts.columns:
        raise ValueError("df_pts must contain 'lon' and 'lat' columns")

    lons = df_pts["lon"].values
    lats = df_pts["lat"].values
    pts = list(zip(lons, lats))

    if method == "convex":
        poly = MultiPoint(pts).convex_hull
    elif method == "alpha":
        if alphashape is None:
            raise RuntimeError("alphashape not installed; install it or use method='convex'")
        if alpha is None:
            alpha = alphashape.optimizealpha(pts)
        poly = alphashape.alphashape(pts, alpha)
    else:
        raise ValueError("method must be 'convex' or 'alpha'")

    return poly


def _plot_shapely_on_ax(ax, poly, lon_ref, line_kwargs=None):
    """
    Plot shapely polygon `poly` on the cartopy axis `ax`.
    lon_ref: array-like longitudes used by the contour/plot (e.g. lon_cyc)
    line_kwargs: kwargs forwarded to ax.plot()
    """
    if line_kwargs is None:
        line_kwargs = dict(color="red", linewidth=2, zorder=12)

    geom = mapping(poly)

    def _plot_ring(coords):
        xs, ys = zip(*coords)
        xs = _normalize_lons_to_range(xs, lon_ref)
        # close polygon if necessary
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xs = tuple(xs) + (xs[0],)
            ys = tuple(ys) + (ys[0],)
        ax.plot(xs, ys, transform=ccrs.PlateCarree(), **line_kwargs)

    if geom["type"] == "Polygon":
        _plot_ring(geom["coordinates"][0])
    elif geom["type"] == "MultiPolygon":
        for poly_coords in geom["coordinates"]:
            _plot_ring(poly_coords[0])
    else:
        # fallback: try exterior coords attribute
        try:
            coords = list(poly.exterior.coords)
            _plot_ring(coords)
        except Exception:
            # nothing to plot
            pass

def plot_box(ax, lats_sel, lons_sel, lon_ref=None, **plot_kwargs):
    """
    Very small helper to draw a rectangular box on a Cartopy (PlateCarree) axis.

    Parameters
    - ax: GeoAxes (e.g. projection=ccrs.PlateCarree())
    - lats_sel: slice(lat0, lat1) or (lat0, lat1)
    - lons_sel: slice(lon0, lon1) or (lon0, lon1)
    - lon_ref: optional 1D array of longitudes used by the map (e.g. lon_cyc).
               If provided and min(lon_ref) >= 0 the box longitudes will be plotted in 0..360 space.
               Otherwise -180..180 is assumed.
    - plot_kwargs: forwarded to ax.plot (color, linewidth, linestyle, zorder, ...)

    Usage:
      plot_box(ax, slice(-5,5), slice(190,240), lon_ref=lon_cyc, color='k', linewidth=1.5, linestyle='--')
    """

    # unpack
    if isinstance(lats_sel, slice):
        lat0, lat1 = lats_sel.start, lats_sel.stop
    else:
        lat0, lat1 = lats_sel

    if isinstance(lons_sel, slice):
        lon0, lon1 = lons_sel.start, lons_sel.stop
    else:
        lon0, lon1 = lons_sel

    # decide lon domain: True -> 0..360, False -> -180..180
    if lon_ref is not None:
        to_360 = np.nanmin(lon_ref) >= 0
    else:
        # infer from lons: if either lon > 180 assume 0..360
        to_360 = (lon0 > 180) or (lon1 > 180)

    if plot_kwargs is None:
        plot_kwargs = dict(c="red", linewidth=2)
        
    def _norm(lon):
        return (np.mod(lon, 360)) if to_360 else ((np.mod(lon + 180, 360) - 180))

    lon0n, lon1n = _norm(lon0), _norm(lon1)

    def _draw(lon_a, lon_b):
        xs = [lon_a, lon_b, lon_b, lon_a, lon_a]
        ys = [lat0, lat0, lat1, lat1, lat0]
        try:
            ax.plot(xs, ys, transform=ccrs.PlateCarree(), **plot_kwargs)
        except Exception:
            ax.plot(xs, ys, **plot_kwargs)

    if lon1n >= lon0n:
        _draw(lon0n, lon1n)
    else:
        # crosses dateline -> split into two boxes
        if to_360:
            _draw(lon0n, 360.)
            _draw(0., lon1n)
        else:
            _draw(lon0n, 180.)
            _draw(-180., lon1n)



def plot_maps_sig_v2(
    path_list,
    alpha=False,
    alpha_val=0.22,
    detr=False,
    amoc_vars=None,
    spg_csv="spg_grid.csv",
    spg_method="convex",
    spg_alpha=None,
    spg_line_kwargs=None,
    regions=True,
    lim=None
):
    """
    Same as your function but with optional SPG boundary plotting and formatted lat/lon labels.

    Provide spg_csv="spg_grid.csv" to draw the boundary. The polygon is computed once
    and plotted on every map panel (after contouring) so it overlays the maps.
    """

    if amoc_vars is None:
        amoc_vars = ['amoc_max', 'atlantic_Up', 'SO_Up', 'indopac_resUp']
        col_titles = ['AMOC max', 'Atlantic Up', 'Southern Ocean Up', 'Indopac residual Up']
        col_colors = ['purple', 'deepskyblue', 'darkorange', 'yellowgreen']
    else:
        col_titles = amoc_vars
        col_colors = ['y']*len(amoc_vars)

    if detr:
        for i in range(len(amoc_vars)):
            amoc_vars[i] = amoc_vars[i] + '_detr'

    if lim is None:
        vmin, vmax = -0.3, 0.3
    else:
        vmin, vmax = -lim, lim

    nrows = len(path_list)
    ncols = len(amoc_vars)
    csteps = 21

    row_colors = ['crimson', 'forestgreen', 'mediumblue']
    titles = ['HIST', 'ATM', 'OCE']
    
    cmap = plt.get_cmap('PRGn')
    if alpha:
        vmin, vmax = -0.4, 0.4
        alpha_low = 0.2
        csteps = 256
        cmap_colors = cmap(np.linspace(0, 1, csteps))
        values = np.linspace(vmin, vmax, csteps)
        cmap_colors[:, -1] = np.where((values > -alpha_val) & (values < alpha_val), alpha_low, 1.0)
        cmap = colors.ListedColormap(cmap_colors)

    # Prepare SPG polygon once if requested
    spg_poly = None
    if spg_csv is not None:
        df_pts = pd.read_csv(spg_csv)
        spg_poly = _make_region_polygon(df_pts, method=spg_method, alpha=spg_alpha)

    fig = plt.figure(figsize=(5.25 * ncols, 3 * nrows + .75))
    gs = fig.add_gridspec(
        nrows + 2, ncols,
        height_ratios=[5.25] * nrows + [1, 0.75],
        wspace=0.03, hspace=0.0
    )

    axes = np.empty((nrows, ncols), dtype=object)
    for c in range(ncols):
        for r in range(nrows):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())

    im = None

    for r, (p, row_title, row_c) in enumerate(zip(path_list, titles, row_colors)):
        for c, (amoc_name, col_title, col_c) in enumerate(zip(amoc_vars, col_titles, col_colors)):
            ax = axes[r, c]
            path = f"{p}_{amoc_name}_coeffs_ensmean.nc"
            ds = xr.open_dataset(path)
            ens_slope = ds["slope_sig"]
            rest = ds["slope"].where(np.isnan(ds["slope_sig"]))
            
            ens_slope_cyc, lon_cyc = add_cyclic_point(ens_slope.values, coord=ens_slope.lon.values)
            rest_cyc, _ = add_cyclic_point(rest.values, coord=ens_slope.lon.values)

            ax.add_feature(cfeature.COASTLINE, edgecolor='lightgrey', linewidth=0.6)
            ax.add_feature(cfeature.LAND, color='whitesmoke')

            im = ax.contourf(
                lon_cyc, ens_slope.lat, ens_slope_cyc,
                levels=np.linspace(vmin, vmax, int(vmax * 100 + 1)),
                cmap=cmap,
                extend='both',
                transform=ccrs.PlateCarree()
            )

            ax.contourf(
                lon_cyc, ens_slope.lat, rest_cyc,
                levels=np.linspace(vmin, vmax, int(vmax * 100 + 1)),
                cmap=cmap,
                extend='both',
                transform=ccrs.PlateCarree(),
                alpha=0.15
            )

            # Plot SPG polygon on top of the map (if provided)
            if spg_poly is not None:
                if c==0 and r in [1,2]:
                    _plot_shapely_on_ax(ax, spg_poly, lon_ref=lon_cyc, line_kwargs=spg_line_kwargs)

            if regions:
                if c in [2,3] and r in [0,1]:
                    plot_box(ax, lats_sel = slice(-5, 5), lons_sel = slice(190, 240), color='cyan', linewidth=2)
    
                if c == 0 and r in [1,2]:
                    plot_box(ax, lats_sel = slice(10.5,22.5), lons_sel = slice(305, 338), color='magenta', linewidth=2)

            # Set ticks on all axes
            _set_map_ticks(ax, r, c, nrows, ncols)

            # vertical col titles (leftmost column only)
            if c == 0:
                ax.text(
                    -0.2, 0.5, row_title,
                    color=row_c,
                    transform=ax.transAxes,
                    rotation=90,
                    va='center', ha='center',
                    fontsize=28, fontweight='bold'
                )

            if r==0:
                # coloured column suptitle
                fig.text(
                    (axes[0, c].get_position().x0 + axes[0, c].get_position().x1) / 2,
                    axes[0, c].get_position().y1 + 0.015,
                    col_title,
                    ha='center', va='bottom',
                    fontsize=20, fontweight='bold',
                    color=col_c
                )

    # colorbar
    cax = fig.add_subplot(gs[-1, :])
    fig.colorbar(im, cax=cax, orientation='horizontal', label="Regression coefficient (K/Sv)")

    plt.show()
    return (fig)


def _set_map_ticks(ax, row_idx, col_idx, nrows, ncols):
    """
    Set lat/lon ticks with proper labels.
    
    Labels:
    - Latitude: 90°S, 60°S, 30°S, 0°, 30°N, 60°N, 90°N
    - Longitude: 180°, 120°W, 60°W, 0°, 60°E, 120°E, 180°
    
    Only show labels on:
    - Left column (col_idx == 0) for latitude labels
    - Bottom row (row_idx == nrows - 1) for longitude labels
    """
    
    # Latitude ticks and labels
    lat_ticks = [-90, -60, -30, 0, 30, 60, 90]
    lat_labels = ['90°S', '60°S', '30°S', '0°', '30°N', '60°N', '90°N']
    
    # Longitude ticks and labels (0-360 convention)
    # lon_ticks = [360, 300, 240, 0, 60, 120, 180]
    lon_ticks = [-120, -60, 0, 60, 120, 180]
    lon_labels = ['120°W', '60°W', '0°', '60°E', '120°E', '180°']
    
    # Set latitude ticks
    ax.set_yticks(lat_ticks)
    if col_idx == 0:
        ax.set_yticklabels(lat_labels, fontsize=12)
    else:
        ax.set_yticklabels([])
    
    # Set longitude ticks
    ax.set_xticks(lon_ticks)
    if row_idx == nrows - 1:
        ax.set_xticklabels(lon_labels, fontsize=12)#, rotation=0, ha='right')
    else:
        ax.set_xticklabels([])
    
    # Show gridlines for ticks
    # ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


f = plot_maps_sig_v2(regress_paths)



#!/usr/bin/env python3
"""
regression.py

Takes:
- a list of *detrended* SST files (time, lat, lon)
- a list of *detrended* AMOC files (same length; one per SST file)

For each chosen AMOC variable:
    - compute regression maps SST -> AMOC separately for each file pair
    - create one multi-panel figure with all members:
        - 5x10 if len(files) == 50
        - 4x4 if len(files) == 16
        - otherwise: quasi-quadratic Grid

Example:
  python regression_maps_per_member.py \
    --sst-files sst_m1.nc sst_m2.nc ... \
    --amoc-files amoc_m1.nc amoc_m2.nc ... \
    --sst-var sst_detr \
    --amoc-vars amoc_max atlantic_Up \
    --fig-prefix regress_hist_ \
    --suptitle "Historical AMOC–SST regression"
"""

import argparse
import math
from typing import List, Optional

import numpy as np
import xarray as xr

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors
except Exception:
    plt = None

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

try:
    from scipy import stats
except Exception:
    stats = None  # will check later and raise meaningful error


def set_font_sizes():
    if plt is None:
        return
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['scatter.edgecolors'] = 'w'


def _compute_regression_maps(sst_detr, amoc_series: np.ndarray):
    """
    Vectorised OLS regression of SST onto a single AMOC time series:
        y(t, lat, lon) ~ intercept(lat,lon) + slope(lat,lon) * x(t)

    sst_detr : xarray.DataArray with dims (time, lat, lon)
    amoc_series : 1D numpy array with length = time

    Returns: slope, intercept, rvalue, pvalue as 2D arrays (lat, lon)
    """
    if stats is None:
        raise RuntimeError("scipy is required for p-values. Install scipy and try again.")

    sst_arr = sst_detr.values.astype(float)
    x = np.asarray(amoc_series, dtype=float)

    if sst_arr.shape[0] != x.shape[0]:
        raise ValueError(
            f"Time dimension mismatch: SST has {sst_arr.shape[0]} steps, "
            f"AMOC series has {x.shape[0]}"
        )

    T = sst_arr.shape[0]
    spatial_shape = sst_arr.shape[1:]      # (lat, lon)
    N = int(np.prod(spatial_shape))        # number of grid points

    # (time, lat, lon) -> (time, N)
    y_flat = sst_arr.reshape(T, N)

    # masks: where both x and y are present
    x_mask = ~np.isnan(x)                  # (T,)
    y_mask = ~np.isnan(y_flat)             # (T, N)
    mask = x_mask[:, None] & y_mask        # (T, N)

    count = mask.sum(axis=0)               # valid time steps per point (N,)
    valid = count >= 3                     # need at least 3 points for t-test with df >= 1

    # Prepare outputs
    slope = np.full(N, np.nan, dtype=float)
    intercept = np.full(N, np.nan, dtype=float)
    rcoef = np.full(N, np.nan, dtype=float)
    pval = np.full(N, np.nan, dtype=float)

    if not np.any(valid):
        out_shape = spatial_shape
        return (
            slope.reshape(out_shape),
            intercept.reshape(out_shape),
            rcoef.reshape(out_shape),
            pval.reshape(out_shape),
        )

    # Masked x and y for vectorised sums (replace invalid by 0)
    x2d = np.broadcast_to(x[:, None], (T, N))
    x_masked = np.where(mask, x2d, 0.0)      # (T, N)
    y_masked = np.where(mask, y_flat, 0.0)   # (T, N)

    # sums and means
    sum_x = x_masked.sum(axis=0)             # (N,)
    sum_y = y_masked.sum(axis=0)             # (N,)
    # avoid division by zero; only compute means where valid
    mean_x = np.zeros(N, dtype=float)
    mean_y = np.zeros(N, dtype=float)
    mean_x[valid] = sum_x[valid] / count[valid]
    mean_y[valid] = sum_y[valid] / count[valid]

    # deviations
    xm = np.where(mask, x2d - mean_x[None, :], 0.0)    # (T, N)
    ym = np.where(mask, y_flat - mean_y[None, :], 0.0) # (T, N)

    # sums of squares and cross products
    Sxy = (xm * ym).sum(axis=0)    # (N,)
    Sxx = (xm * xm).sum(axis=0)
    Syy = (ym * ym).sum(axis=0)

    # good points: enough samples and non-zero variance
    good = valid & (Sxx > 0) & (Syy > 0)

    # slope and intercept
    slope[good] = Sxy[good] / Sxx[good]
    intercept[good] = mean_y[good] - slope[good] * mean_x[good]

    # Pearson r
    rcoef[good] = Sxy[good] / np.sqrt(Sxx[good] * Syy[good])

    # residual sum of squares: SSR = Syy - slope * Sxy
    SSR = np.empty(N, dtype=float)
    SSR.fill(np.nan)
    SSR[good] = Syy[good] - slope[good] * Sxy[good]
    # numerical safety
    SSR = np.where(SSR < 0, 0.0, SSR)

    # degrees of freedom per point
    df = count - 2  # array (N,)
    # compute standard error of slope: se = sqrt( SSR / df / Sxx )
    se_slope = np.full(N, np.nan, dtype=float)
    denom = Sxx
    denom_pos = denom > 0
    valid_se = good & (df > 0) & denom_pos
    se_slope[valid_se] = np.sqrt( (SSR[valid_se] / df[valid_se]) / denom[valid_se] )

    # t-stat for slope
    t_stat = np.full(N, np.nan, dtype=float)
    nonzero_se = valid_se & (se_slope > 0)
    t_stat[nonzero_se] = slope[nonzero_se] / se_slope[nonzero_se]

    # two-sided p-value from t distribution
    # stats.t.sf works elementwise
    pval_nonzero = np.full(N, np.nan, dtype=float)
    nz = nonzero_se
    if np.any(nz):
        # use survival function for better numeric stability: p = 2 * sf(|t|, df)
        pval_nonzero[nz] = 2.0 * stats.t.sf(np.abs(t_stat[nz]), df[nz])
    pval = pval_nonzero  # assign

    out_shape = spatial_shape
    return (
        slope.reshape(out_shape),
        intercept.reshape(out_shape),
        rcoef.reshape(out_shape),
        pval.reshape(out_shape),
    )


def _guess_spatial_coords(da):
    lat_names = ['lat', 'latitude', 'y']
    lon_names = ['lon', 'longitude', 'x']
    lat_name = None
    lon_name = None
    for n in lat_names:
        if n in da.coords:
            lat_name = n
            break
    for n in lon_names:
        if n in da.coords:
            lon_name = n
            break
    lat_vals = da.coords[lat_name].values if lat_name is not None else None
    lon_vals = da.coords[lon_name].values if lon_name is not None else None
    return lat_vals, lon_vals, (lat_name, lon_name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Compute regression maps from detrended SST and AMOC for multiple SST/AMOC file pairs'
    )
    p.add_argument(
        '--sst-files',
        nargs='+',
        required=True,
        help='List of NetCDF files containing detrended SST (one file per member/realisation)'
    )
    p.add_argument(
        '--amoc-files',
        nargs='+',
        required=True,
        help='List of NetCDF files containing detrended AMOC (same length as --sst-files)'
    )
    p.add_argument(
        '--sst-var',
        required=True,
        help='Name of detrended SST variable in each SST file (e.g. sst_detr)'
    )
    p.add_argument(
        '--amoc-vars',
        nargs='*',
        help='Names of AMOC variables to use; if omitted, all data vars of the first AMOC file are used'
    )
    p.add_argument(
        '--fig-prefix',
        required=True,
        help='Prefix for output PNGs (one per AMOC variable, e.g. "regress_hist_")'
    )
    p.add_argument(
        '--suptitle',
        help='Optional figure title prefix'
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if len(args.sst_files) != len(args.amoc_files):
        raise SystemExit(
            f"--sst-files and --amoc-files must have the same length, got "
            f"{len(args.sst_files)} vs {len(args.amoc_files)}"
        )

    if stats is None:
        raise SystemExit("scipy is required but not available. Install scipy and try again.")

    # open all AMOC datasets
    amoc_datasets = [xr.open_dataset(f) for f in args.amoc_files]

    # determine AMOC variable names
    if args.amoc_vars and len(args.amoc_vars) > 0:
        amoc_names = args.amoc_vars
    else:
        amoc_names = list(amoc_datasets[0].data_vars)

    # labels for figure panels (names from SST filenames)
    file_labels_all = [os.path.basename(f) for f in args.sst_files]

    # template for lat/lon coords
    ds0 = xr.open_dataset(args.sst_files[0])
    if args.sst_var not in ds0:
        raise SystemExit(
            f"SST variable '{args.sst_var}' not found in first SST file {args.sst_files[0]}"
        )
    sst_template = ds0[args.sst_var]
    lat_vals, lon_vals, (lat_name, lon_name) = _guess_spatial_coords(sst_template)

    # loop over each AMOC variable
    for amoc_name in amoc_names:
        slope_maps = []
        intercept_maps = []
        r_maps = []
        p_maps = []
        file_labels = []

        for sst_path, amoc_ds, label in zip(args.sst_files, amoc_datasets, file_labels_all):
            if amoc_name not in amoc_ds:
                print(f"Warning: AMOC variable '{amoc_name}' not found in {getattr(amoc_ds, 'encoding', {}).get('source', 'AMOC file')}; skipping this pair")
                continue

            ds_sst = xr.open_dataset(sst_path)
            if args.sst_var not in ds_sst:
                print(f"Warning: SST variable '{args.sst_var}' not found in {sst_path}; skipping this pair")
                continue

            sst_da = ds_sst[args.sst_var]
            amoc_da = amoc_ds[amoc_name]

            if 'time' not in sst_da.dims or 'time' not in amoc_da.dims:
                print(f"Warning: no 'time' dim for pair ({label}, {amoc_name}); skipping")
                continue

            # NOTE: original code had an alignment step commented out.
            # If time coordinates differ you'll want to align here (inner join).
            sst_aligned = sst_da
            amoc_aligned = amoc_da

            slope_map, intercept_map, r_map, p_map = _compute_regression_maps(
                sst_aligned, amoc_aligned.values.astype(float)
            )
            slope_maps.append(slope_map)
            intercept_maps.append(intercept_map)
            r_maps.append(r_map)
            p_maps.append(p_map)
            file_labels.append(label)

        if len(slope_maps) == 0:
            print(f"No valid SST/AMOC pairs for AMOC variable '{amoc_name}', skipping.")
            continue

        slope_arr = np.stack(slope_maps, axis=0)      # (member, lat, lon)
        intercept_arr = np.stack(intercept_maps, axis=0)
        r_arr = np.stack(r_maps, axis=0)
        p_arr = np.stack(p_maps, axis=0)

        # Ensemble mean (Nan-aware)
        slope_ens = np.nanmean(slope_arr, axis=0)
        intercept_ens = np.nanmean(intercept_arr, axis=0)
        r_ens = np.nanmean(r_arr, axis=0)
        p_ens = np.nanmean(p_arr, axis=0)


        # ---------- NetCDF: members ----------
        # One variable per file plus suffixes: _slope, _r, _p
        coords_members = {
            lat_name: lat_vals,
            lon_name: lon_vals,
        }

        data_vars = {}

        for idx in range(slope_arr.shape[0]):
            slope_map = slope_arr[idx]
            intercept_map = intercept_arr[idx]
            r_map = r_arr[idx]
            p_map = p_arr[idx]

            label = file_labels[idx]
            parts = label.split("_")
            if len(parts) > 5:
                base_name = parts[5]
            else:
                # Fallback, if the filename does not have enough "_"
                base_name = f"member{idx+1}"

            # ensure base_name is safe and unique per suffix
            var_base = base_name
            # Add slope, r, p variables with suffixes
            var_slope = var_base + "_slope"
            var_intercept = var_base + "_intercept"
            var_r = var_base + "_r"
            var_p = var_base + "_p"

            # ensure uniqueness in data_vars
            k = 1
            while var_slope in data_vars or var_intercept in data_vars or var_r in data_vars or var_p in data_vars:
                var_base = f"{base_name}_dup{k}"
                var_slope = var_base + "_slope"
                var_intercept = var_base + "_intecept"
                var_r = var_base + "_r"
                var_p = var_base + "_p"
                k += 1

            data_vars[var_slope] = ((lat_name, lon_name), slope_map)
            data_vars[var_intercept] = ((lat_name, lon_name), intercept_map)
            data_vars[var_r] = ((lat_name, lon_name), r_map)
            data_vars[var_p] = ((lat_name, lon_name), p_map)

        ds_members = xr.Dataset(data_vars, coords=coords_members)

        members_out = f"{args.fig_prefix}{amoc_name}_coeffs_members.nc"
        ds_members.to_netcdf(members_out)
        print(f"Wrote member regression coeffs to {members_out}")

        # ---------- NetCDF: ensemble mean ----------
        coords_ens = {
            lat_name: lat_vals,
            lon_name: lon_vals,
        }
        ds_ens = xr.Dataset(
            {
                "slope": ((lat_name, lon_name), slope_ens),
                "intercept": ((lat_name, lon_name), intercept_ens),
                "r": ((lat_name, lon_name), r_ens),
                "p": ((lat_name, lon_name), p_ens),
            },
            coords=coords_ens,
        )

        ens_out = f"{args.fig_prefix}{amoc_name}_coeffs_ensmean.nc"
        ds_ens.to_netcdf(ens_out)
        print(f"Wrote ensemble-mean regression coeffs to {ens_out}")


if __name__ == '__main__':
    main()

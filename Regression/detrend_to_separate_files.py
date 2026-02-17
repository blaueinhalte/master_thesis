#!/usr/bin/env python3
"""
detrend_to_separate_files.py

Detrend SST (time,lat,lon) and multiple AMOC time series and write
them to two NetCDF files:
  1) detrended SST
  2) detrended AMOC series

This simplified version uses a single detrending function for both SST and
AMOC. It prefers scipy.signal.detrend when available (and when a series
contains no NaNs). If a series contains NaNs, a linear least-squares fit
is computed on the valid points and subtracted (preserving NaNs).
"""

import argparse
from typing import Optional

import numpy as np
import xarray as xr

try:
    from scipy import signal
except Exception:
    signal = None


def _to_time_numeric(da: xr.DataArray) -> np.ndarray:
    """Return a 1D numeric time array (float) suitable for trend fitting."""
    if 'time' in da.coords:
        t = da['time'].to_index().to_numpy()
        if np.issubdtype(t.dtype, np.datetime64):
            x = (t - t[0]) / np.timedelta64(1, 'D')
            return x.astype(float)
        return t.astype(float)
    # fallback: index along first dimension
    length = da.sizes[da.dims[0]]
    return np.arange(length, dtype=float)


def detrend_da(da: xr.DataArray) -> xr.DataArray:
    """
    Detrend an xarray.DataArray along its first (time) dimension.

    - If scipy.signal.detrend is available and the series contains no NaNs,
      use it for speed.
    - If NaNs are present, perform a simple linear least-squares fit on valid
      points and subtract that fitted trend (preserving NaNs).
    """
    time_x = _to_time_numeric(da)
    arr = da.values
    tlen = arr.shape[0]

    # Handle 1D series quickly
    def _detrend_1d(y):
        y = y.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.full_like(y, np.nan)
        # If no NaNs and scipy is available, use it
        if signal is not None and mask.all():
            return signal.detrend(y, type='linear')
        # Otherwise fit linear trend on valid points and subtract
        print('fallback to numpy')
        xm = time_x[mask]
        ym = y[mask]
        A = np.vstack([xm, np.ones_like(xm)]).T
        slope, intercept = np.linalg.lstsq(A, ym, rcond=None)[0]
        trend = slope * time_x + intercept
        out = np.full_like(y, np.nan)
        out[mask] = y[mask] - trend[mask]
        return out

    if arr.ndim == 1:
        detr = _detrend_1d(arr)
    else:
        rest = int(np.prod(arr.shape[1:]))
        flat = arr.reshape((tlen, rest))
        detr_flat = np.full_like(flat, np.nan, dtype=float)
        for i in range(rest):
            detr_flat[:, i] = _detrend_1d(flat[:, i])
        detr = detr_flat.reshape(arr.shape)

    return xr.DataArray(detr, coords=da.coords, dims=da.dims, name=(da.name) if da.name else None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Detrend SST and multiple AMOC series and write detrended data to two NetCDF files'
    )
    p.add_argument('--sst-file', required=True,
                   help='NetCDF file containing SST (time,lat,lon)')
    p.add_argument('--sst-var', required=True,
                   help='SST variable name in the SST file')
    p.add_argument('--amoc-file', required=True,
                   help='NetCDF file containing one or more AMOC time series')
    p.add_argument('--amoc-vars', nargs='*',
                   help='Names of AMOC variables to use. If omitted, all data variables are used')
    p.add_argument('--out-sst', default='detrended_sst.nc',
                   help='Output NetCDF path for detrended SST')
    p.add_argument('--out-amoc', default='detrended_amoc.nc',
                   help='Output NetCDF path for detrended AMOC series')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.sst_file is not None:
        # Load SST
        sst_ds = xr.open_dataset(args.sst_file)
        if args.sst_var not in sst_ds:
            raise SystemExit(f"SST variable '{args.sst_var}' not found in {args.sst_file}")
        sst_da = sst_ds[args.sst_var]  
        # Detrend SST using the single detrend function
        sst_detr = detrend_da(sst_da)
        # keep coords (lat/lon/time) intact; rename variable to indicate detrended
        sst_name = f"{args.sst_var}_detr"
        ds_sst_out = xr.Dataset({sst_name: sst_detr})
        ds_sst_out.to_netcdf(args.out_sst)
        print(f"Wrote detrended SST to {args.out_sst}")
    else:
        sst_da = None

    # Load AMOC
    amoc_ds = xr.open_dataset(args.amoc_file)
    if args.amoc_vars and len(args.amoc_vars) > 0:
        amoc_names = args.amoc_vars
    else:
        amoc_names = list(amoc_ds.data_vars.keys())

    amoc_detr_dict = {}
    used_names = []

    for name in amoc_names:
        if name not in amoc_ds:
            print(f"Warning: AMOC variable '{name}' not found in {args.amoc_file}; skipping")
            continue
        amoc_da = amoc_ds[name]

        # make sure time lengths match (we require same time axis as SST)
        if 'time' in amoc_da.coords and sst_da is not None and 'time' in sst_da.coords:
            if len(amoc_da['time']) != len(sst_da['time']):
                print(f"Warning: time length mismatch for '{name}': "
                      f"SST {len(sst_da['time'])} vs AMOC {len(amoc_da['time'])}; skipping")
                continue

        # Detrend the 1D AMOC series (or any series where time is first dim)
        amoc_detr = detrend_da(amoc_da)

        # Force AMOC to use the SST time coordinate (keeps datetime dtype & alignment)
        if sst_da is not None and 'time' in sst_da.coords:
            amoc_detr = xr.DataArray(amoc_detr.values, coords={'time': sst_da['time']}, dims=('time',), name=name + '_detr')
        else:
            amoc_detr.name = name + '_detr'

        amoc_detr_dict[name + '_detr'] = amoc_detr
        used_names.append(name)

    if len(used_names) == 0:
        raise SystemExit("No valid AMOC variables found to detrend")

    ds_amoc_out = xr.Dataset(amoc_detr_dict)
    # ensure time coord from SST if present
    if sst_da is not None and 'time' in sst_da.coords:
        ds_amoc_out = ds_amoc_out.assign_coords(time=sst_da['time'])

    ds_amoc_out.to_netcdf(args.out_amoc)
    print(f"Wrote detrended AMOC series to {args.out_amoc}")


if __name__ == '__main__':
    main()

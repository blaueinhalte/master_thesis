import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.gridspec import GridSpec
from matplotlib import colors


try:
    # Newer scikit-learn
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # Older scikit-learn: emulate root_mean_squared_error via mean_squared_error
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred, *args, **kwargs):
        """
        Backwards-compatible replacement for sklearn.metrics.root_mean_squared_error
        using mean_squared_error.
        """
        # Default: RMSE => squared=False
        squared = kwargs.pop("squared", False)

        # Try to pass squared, but older versions may not accept it
        try:
            return mean_squared_error(
                y_true, y_pred, squared=squared, *args, **kwargs
            )
        except TypeError:
            # Oldest versions: no `squared` kwarg – compute sqrt manually
            mse = mean_squared_error(y_true, y_pred, *args, **kwargs)
            return np.sqrt(mse) if not squared else mse


class sst:
    """
    Class to handle, compare, and visualize SST datasets (historical, assimilation_oce, observations, etc.)
    with ocean basin masks derived from GR15L40 grid.
    """

    def __init__(
        self, 
        sst_paths: list,
        maskfile='/work/uo1075/u241372/GR15L40_fx_rbek_mean_remap.nc',
        names : list = None
    ):
        
        """
        Initialize SST analyzer with dataset dictionary and optional GR15L40 mask file.

        Parameters
        ----------
        sst_paths : list
            [
        "...hist_sst.nc",
        "...ass_sst.nc",
        "...ass_atm_sst.nc",
        "...obsv_sst.nc"
        ]
            Reference for difference plot must be last in list.
        maskfile : str
            Path to GR15L40 mask file containing 'rbek' region indices.
        names : list
            list of names for the dictionary and suptitles. Must be in same order as paths.
        """
        if names==None:
            self.names = ['historical_ens', 'assimilation_atm', 'assimilation_oce_ens', 'observations']
        else:
            self.names = names
        sst_dict = {name: sst_path for name, sst_path in zip(self.names, sst_paths)}
        self.sst_files = sst_dict
        self.datasets = {name: xr.open_dataset(path).tos for name, path in sst_dict.items()}
        #self.datasets_yearly = {name: self.datasets[name].resample(time='1YE').mean(dim='time') for name in self.datasets.keys()}


        # Align all times to observations dataset -> different montly mean time stamps
        ref_time = self.datasets[[*self.datasets][0]]['time']
        for name, _ in self.datasets.items():
            self.datasets[name]['time'] = ref_time
            if 'longitude' in self.datasets[name].dims:
                self.datasets[name] = self.datasets[name].rename({'longitude': 'lon', 'latitude': 'lat'})
            if 'depth' in self.datasets[name].dims:
                self.datasets[name] = self.datasets[name].mean(dim='depth')
                
        # Load GR15L40 mask and define basins
        grid = xr.open_dataset(maskfile).mean(dim='time')

        atl_n = grid.rbek.where(grid.rbek == 4, other=0).astype(bool)
        atl_s = grid.rbek.where(grid.rbek == 5, other=0).astype(bool)
        self.atl_mask = atl_n | atl_s

        indopac_main = grid.rbek.where(grid.rbek == 7, other=0).astype(bool)
        indopac_strip = grid.rbek.where(grid.rbek == 8, other=0).astype(bool)
        self.indopac_mask = indopac_main | indopac_strip

        self.so_mask = grid.rbek.where(grid.rbek == 6, other=0).astype(bool)

        self.masks = {
            'Atlantic': self.atl_mask,
            'IndoPacific': self.indopac_mask,
            'SouthernOcean': self.so_mask
        }

        self.colors = ['crimson', 'forestgreen', 'mediumblue', 'darkorange']
        titles = []
        for name in self.names:
            n = name.capitalize().replace('_', ' ')
            for rep1, rep2 in zip(['atm', 'oce'], ['Atmosphere', 'Ocean']):
                if rep1 in name:
                    n = n.replace(rep1, rep2)
                if 'ens' in name:
                    n = n.replace('ens', 'Ensemble')
            if 'Observations' in n:
                n = 'HadISST Reference'
            titles.append(n)     
        self.titles = titles

        titles_vert = []
        for name in self.names:
            if name == 'observations':
                n_vert = 'HadISST'
            n_vert = name.split('_')
            if 'historical' in n_vert:
                n_vert = n_vert[0][:4].upper()
            elif "atm" in n_vert or "oce" in n_vert:
                n_vert = n_vert[1][:3].upper()
            titles_vert.append(n_vert)
        self.titles_vert = titles_vert
        

    # ----------------------------------------------------
    #  Basic Processing
    # ----------------------------------------------------
    def compute_diff(self, ref='observations'):
        """Compute differences of all datasets relative to a reference dataset (default: observations)."""
        ref_data = self.datasets[ref]
        self.diff = {name: ds - ref_data for name, ds in self.datasets.items() if name != ref}
        return self.diff

    def compute_diff_mean(self, ref='observations'):
        ref_data = self.datasets[ref]
        self.diff_mean = {name: ds.mean(dim='time') - ref_data.mean(dim='time') for name, ds in self.datasets.items() if name != ref}
        return self.diff_mean

    def rmse_corr_spatial(self, ds, ref='observations'):
        """
        Compute RMSE and correlation maps between datasets and a reference.
    
        Parameters
        ----------
        ds  : dict
            Dictionary of datasets {name: xr.DataArray}
            Each DataArray must have a 'time' dimension.
        ref : str
            Name of reference dataset.
    
        Returns
        -------
        tuple of dict
            (rmse_dict, corr_dict), each containing xr.DataArray maps
        """
        ref_da = ds[ref]
    
        rmse_dict = {}
        corr_dict = {}
    
        for name in [n for n in self.names if n != ref]:
            da = ds[name]
    
            # Align time dimension
            da, ref_aligned = xr.align(da, ref_da)
    
            # RMSE over time
            rmse = np.sqrt(((da - ref_aligned) ** 2).mean(dim='time'))
    
            # Correlation over time (gridpoint-wise)
            corr = xr.corr(da, ref_aligned, dim='time')
    
            rmse_dict[name] = rmse
            corr_dict[name] = corr
    
        return rmse_dict, corr_dict


    def global_mean(self, ds):
        """Compute global mean (lon/lat)."""
        return ds.mean(dim=['lat', 'lon'])

    def yearly_mean(self, ds):
        """Compute yearly mean from monthly data."""
        if isinstance(ds, xr.DataArray):
            return ds.resample(time='1YE').mean(dim='time')
        if isinstance(ds, dict):
            ds_yearly = {name: ds[name].resample(time='1YE').mean(dim='time') for name in ds.keys()}
            return ds_yearly
        

    def rmse_corr(self, ds, ref='observations'):
        """
        Compute RMSE and correlation coefficients between multiple datasets and a reference dataset.
    
        Parameters
        ----------
        ds  : dict
            Dictionary of datasets to compare {name: xr.DataArray or np.ndarray}.
        ref : string
            Key (name) of reference dataset to compare against.
    
        Returns
        -------
        tuple of dict
            (rmse_dict, corr_dict)
        """
        ref = ds[ref]
        rmse_dict, corr_dict = {}, {}
        b = np.array(ref).flatten()
    
        for name in self.names[:3]:
            a = np.array(ds[name]).flatten()
            mask = np.isfinite(a) & np.isfinite(b)
            if np.any(mask):
                a, b_masked = a[mask], b[mask]
                rmse_dict[name] = root_mean_squared_error(a, b_masked)
                corr_dict[name] = np.corrcoef(a, b_masked)[0, 1]
            else:
                rmse_dict[name], corr_dict[name] = np.nan, np.nan

        return rmse_dict, corr_dict


    # ----------------------------------------------------
    #  Basin Analysis
    # ----------------------------------------------------
    def basin_mean(self, ds, mask):
        """Compute mean SST over a masked region."""
        if isinstance(ds, xr.DataArray):
            return ds.where(mask).mean(dim=['lat', 'lon'])
        if isinstance(ds, dict):
            return {name: val.where(mask).mean(dim=['lat', 'lon']) for name, val in ds.items()}
        return 

    # ----------------------------------------------------
    #  Plotting
    # ----------------------------------------------------
    def plot_mean_maps(self, diff=False, corr=False, ref='observations', vert=False):
        """Plot mean SST maps for all runs."""
    
        if diff and not corr:
            data_mean = self.compute_diff_mean(ref=ref)
            cmap = 'seismic'
            tos_min = -6
            tos_max = 6
            csteps = 4*tos_max+1
            cssteps = csteps 
            n = len(self.names) - 1
            clabel = 'SST difference (K)'
            levs = np.linspace(tos_min, tos_max, csteps)
            
        elif corr and not diff:
            _, data_mean = self.rmse_corr_spatial(ds=self.datasets, ref=ref)
            cmap = 'PRGn'
            tos_min = -1
            tos_max = 1
            cssteps = 0
            n = len(self.names) - 1
    
            csteps = 21
            cmap = plt.get_cmap('PRGn')
            cmap_alpha = cmap(np.linspace(0, 1, csteps))
            cmap_0 = int((csteps - 1) / 2)
            idx_03 = int(np.round(0.3 / (2 / (csteps - 1)), 1))
        
            cmap_alpha[cmap_0 - idx_03:cmap_0 + idx_03 + 1, -1] = 0.3
            cmap = colors.ListedColormap(cmap_alpha)
            clabel = 'Correlation coefficient (-)'
            levs = np.linspace(tos_min, tos_max, csteps)
            
            
        elif corr and diff:
            raise KeyError("Can't plot correlation and difference at the same time")
        
        else:
            data_mean = {name: ds.mean(dim='time') for name, ds in self.datasets.items()}
            cmap = 'RdYlBu_r'
            min_vals = [float(ds.min().values) for ds in data_mean.values()]
            max_vals =  [float(ds.max().values) for ds in data_mean.values()]
            tos_min = 0 #(np.floor(min(min_vals)))
            tos_max = 29#(np.ceil(max(max_vals)))
            csteps = 100
            cssteps = 10
            levs = np.linspace(tos_min, tos_max, 99)
            n = len(self.names)
            clabel = 'SST (°C)'
    
        
        if n==3:
            w,h = [3,1]
        elif n>3:
            w,h = [2,2]
        elif n==2:
            w,h = [2,1]
        elif n==1:
            w,h = [1,1]
    
        if vert:
            w,h = h,w
    
            
        fsize = (10*w, 6*h) 
        fig = plt.figure(figsize=fsize)
    
        height_ratios = [6] * h + ([0.075, .75] if vert else [])
        width_ratios = [10]*w + ([] if  vert else [0.75])
    
        
        
        if n > 3:
            gs = GridSpec(2, int(np.ceil(n/2)), figure=fig, height_ratios=height_ratios, width_ratios = width_ratios, hspace=0.2, wspace=0.12) 
    
        elif n == 3:
            gs = GridSpec(len(height_ratios),len(width_ratios), figure=fig, height_ratios=height_ratios, width_ratios = width_ratios, hspace=0.2, wspace=0.12) 
    
        else:
            gs = GridSpec(1,n, figure=fig, height_ratios=height_ratios, width_ratios = width_ratios, hspace=0.2, wspace=0.12) 
    
    
        
        for i, (name, title, col) in enumerate(zip(self.names[:n], self.titles_vert[:n] if vert else self.titles[:n], self.colors[:n])):
            if n==3 and not vert:
                ax = fig.add_subplot(gs[0,i], projection=ccrs.PlateCarree())
                row, col_idx = 0, i
            elif n==3 and vert:
                ax = fig.add_subplot(gs[i,0], projection=ccrs.PlateCarree())
                row, col_idx = i, 0
            else:
                ax = fig.add_subplot(gs[i // 2, i % 2], projection=ccrs.PlateCarree())
                row, col_idx = i // 2, i % 2
                
            ax.add_feature(cfeature.COASTLINE, edgecolor='lightgrey', linewidth=0.6)
            ax.add_feature(cfeature.LAND, color='whitesmoke')
        
            ds = data_mean[name]
        
            # add cyclic point to avoid seam at 0°/360°
            z_cyc, lon_cyc = add_cyclic_point(ds.values, coord=ds.lon.values)
        
            c = ax.contourf(
                lon_cyc, ds.lat.values, z_cyc,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                levels=levs,
                extend='both' if not corr else 'neither'
            )
        
            cs = ax.contour(
                lon_cyc, ds.lat.values, z_cyc,
                transform=ccrs.PlateCarree(),
                levels=np.linspace(tos_min, tos_max, cssteps),
                colors='grey',
                linewidths=0.5
            )
    
            # Add ticks to all subplots
            ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
            
            # Add x-axis labels to last row only
            if row == h - 1:  # last row
                labels = ['180°','120°W','60°W', '0°','60°E','120°E','180°']
                ax.set_xticklabels(labels)
                # ax.set_xlabel('Longitude')
            else:
                ax.set_xticklabels([])
            
            # Add y-axis labels to first column only
            if col_idx == 0 and not corr:  # first column
                labels = ['90°S','60°S','30°S', '0°','30°N','60°N','90°N']
                ax.set_yticklabels(labels)
                # ax.set_ylabel('Latitude')
            else:
                ax.set_yticklabels([])
    
            if not vert:
                ax.set_title(title, color=col, fontweight='bold')
    
            elif vert and not corr:
                ax.text(
                    -0.17, 0.5, title,
                    color=col,
                    transform=ax.transAxes,
                    rotation=90,
                    va='center', ha='center',
                    fontsize=28, fontweight='bold'
                )
        
        # cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
        cbar_pos = gs[-1,:] if vert else gs[:,-1]
        cbar_ax = fig.add_subplot(cbar_pos)
        cbar = fig.colorbar(c, cax=cbar_ax, orientation="horizontal" if vert else 'vertical')
        cbar.set_label(clabel)
        if not diff and not corr:
            cbar.set_ticks(np.arange(0,30,5))  # Set the tick positions
        
        #plt.tight_layout()
        return fig



    def plot_global_timeseries(self, ylim=False):
        """Plot yearly global mean SST for all datasets."""

        data = {name: self.global_mean(ds) for name, ds in self.datasets.items()}
        time = data['observations'].time

        rmse, corr = self.rmse_corr(data)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, col, tit in zip(self.names, self.colors, self.titles):
            label = f'{tit} \nRMSE={rmse[name]:.3f}, r={corr[name]:.3f}' if name != 'observations' else tit
            ax.plot(time, data[name], label=label, color=col)

        legend = ax.legend()
        for label, col in zip(legend.get_texts(), self.colors):
            label.set_color(col)
            
        ax.set_title('Global Mean SST (Yearly)')
        ax.set_ylabel('Temperature [°C]')
        if ylim:
            ax.set_ylim(ylim)
        
        plt.tight_layout()
        return fig

    
    def plot_basin_timeseries(self):
        """Plot yearly mean SST per ocean basin."""
        time = self.datasets['observations'].time
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        for ax, (bname, mask) in zip(axes, self.masks.items()):
            axmax = 0
            ds_basins = self.basin_mean(self.datasets, mask)
            rmse, corr = self.rmse_corr(ds_basins)
            for name, col, tit in zip(self.names, self.colors, self.titles):
                ds_basin = ds_basins[name]
                label=f'{tit} \nRMSE={rmse[name]:.3f}, r={corr[name]:.3f}' if name != 'observations' else tit
                ax.plot(time, ds_basin, label=label, color=col)
                axmax = np.max(ds_basin) if np.max(ds_basin)>axmax else axmax          
            
            ax.set_ylim([np.round(axmax,2)-1.88, np.round(axmax,2)+0.12])
            ax.set_title(f'{bname} Basin Mean SST')

            legend = ax.legend()
            for label, col in zip(legend.get_texts(), self.colors):
                label.set_color(col)
                
        plt.tight_layout()
        return fig

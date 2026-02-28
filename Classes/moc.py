import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt

from scipy.signal import argrelextrema
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

class moc:

### __init__
    def __init__(
        self,
        file,
        typ,
        suptitle
    ):
        self.file = file
        self.typ = typ
        self.suptitle = suptitle

        self.timespan = slice('1958-01-01', '2014-12-31')

        if type(file)==str:
            self.data = xr.open_dataset(self.file)
        else:
            self.data = file

        mean_dims = []
        if 'lon' in self.data.dims:
            mean_dims.append('lon')
        if 'depth' in self.data.dims:
            if len(self.data.depth.values)==1:
                mean_dims.append('depth')
        
        if len(mean_dims)>0:
            self.data = self.data.mean(dim=mean_dims)

        if len(self.data.time) > 57:
            self.data = self.data.resample(time='1Y').mean(dim='time')
            if 'Multi' not in self.suptitle:
                self.data = self.data.sel(time=self.timespan)
            

        # assimilation and scenario (?)
        if self.typ=='ass':
            self.moc_global = self.data.global_moc/1e9
            self.moc_atlantic = self.data.atlantic_moc/1e9
            self.moc_indopac = self.data.indopacific_moc/1e9
            self.depth_name = 'depth_2'

            # for calculating amoc max
            self.min_amoc_max_lat = 25 # minimum latitude for amoc_max
            self.max_amoc_min_lat = 25 # maximum latitude for amoc_min

        # historical runs and piControl
        if self.typ=='hist':
            # self.data['lat'] = np.linspace(-89.5, 89.5, 180) # delete this line later when fixed
            self.moc_global = self.data.msftmz.isel(basin=0)/1e9
            self.moc_atlantic = self.data.msftmz.isel(basin=1)/1e9
            self.moc_indopac = self.data.msftmz.isel(basin=2)/1e9
            self.depth_name = 'lev'

            # for calculating amoc max
            self.min_amoc_max_lat = 0
            self.max_amoc_min_lat = 0


        self.SOlat = -35.5 # latitude for plotting Southern (global) Ocean south from here (-38.6 for ACC)
        self.vmax = 54 # max value of the streamfunction (for colorbar)
        self.clevels = np.arange(-self.vmax, self.vmax, 2) # contour lines every 2 Sv

        # for calculating amoc max
        self.min_depth = 500

        if 'Historical' in self.suptitle or 'HIST' in self.suptitle:
            self.suptitle_color = 'crimson'
        elif 'Atmosphere' in self.suptitle or 'ATM' in self.suptitle:
            self.suptitle_color = 'forestgreen'
        elif 'Ocean' in self.suptitle or 'OCE' in self.suptitle:
            self.suptitle_color = 'mediumblue'
        elif 'Observation' in self.suptitle:
            self.suptitle_color = 'darkorange'
        elif 'Multi' in self.suptitle:
            self.suptitle_color = 'indigo'

    def sel_timespan(self):
        self.moc_global = self.moc_global.sel(time=self.timespan)
        self.moc_atlantic = self.moc_atlantic.sel(time=self.timespan)
        self.moc_indopac = self.moc_indopac.sel(time=self.timespan)
        return self.moc_global, self.moc_atlantic, self.moc_indopac


### calc_mean_moc
    def calc_mean_moc(self, glo=None, atlantic=None, indopac=None):
        """
        Compute the 57-year mean for each MOC (Meridional Overturning Circulation) dataset.

        Args:
        lat_zoom (slice, optional): A slice object to select a specific latitudinal range (e.g., slice(30, 60)).

        Returns:
        tuple: A tuple containing three xarray.DataArray objects:
            - moc_global_mean: The mean of the global MOC dataset.
            - moc_atlantic_mean: The mean of the Atlantic MOC dataset.
            - moc_indopac_mean: The mean of the Indo-Pacific MOC dataset.
        """
        if any(x is None for x in [glo, atlantic, indopac]):
            glo = self.moc_global
            atlantic = self.moc_atlantic
            indopac = self.moc_indopac
        
        moc_global_mean = glo.mean(dim=['time'])
        moc_atlantic_mean = atlantic.mean(dim=['time'])
        moc_indopac_mean = indopac.mean(dim=['time'])
        return moc_global_mean, moc_atlantic_mean, moc_indopac_mean

    def calc_points(self, glo=None, atlantic=None, indopac=None, mean=False):
        """
        Compute AMOC and related diagnostic points (max, min, 34.5°S) for
        each time step if 'time' is in the dataset dimensions.
        Fully vectorized using xarray.apply_ufunc for efficiency.
        """

        if mean:
            if any(x is None for x in [glo, atlantic, indopac]):
                glo, atlantic, indopac = self.calc_mean_moc()
        else:
            if any(x is None for x in [glo, atlantic, indopac]):
                glo, atlantic, indopac = [self.moc_global, self.moc_atlantic, self.moc_indopac]
    
        def _calc_amoc_points(glo_t, atl_t, indo_t, lat, depth, SOlat, min_lat, min_depth):
    
            # Restrict depth slices
            depth_mask = depth >= min_depth
    
            # --- AMOC MAX ---
            lat_mask_north = lat >= min_lat # north of minimum latitude for AMOC max
            atl_north = atl_t[np.ix_(lat_mask_north, depth_mask)] # get north atlantic MOC
            amoc_depth_max = np.nanmax(atl_north, axis=1) # maximum at each latitude (?)
            lat_max_idx = np.nanargmax(amoc_depth_max) # lat idx of max
            lat_max = lat[lat_mask_north][lat_max_idx] # lat of max
            depth_max_idx = np.nanargmax(atl_north[lat_max_idx, :]) # depth idx of max
            depth_max = depth[depth_mask][depth_max_idx] # depth of max
            amoc_max = atl_north[lat_max_idx, depth_max_idx] # AMOC max
    
            # --- AMOC MIN ---
            lat_mask_south = (lat >= SOlat) & (lat <= lat_max) # AMOC min south of AMOC max
            atl_south = atl_t[np.ix_(lat_mask_south, depth_mask)] 
            amoc_depth_min = np.nanmax(atl_south, axis=1) # max at each lat
            lat_min_idx = np.nanargmin(amoc_depth_min) # ixd of min of max at each lat
            lat_min = lat[lat_mask_south][lat_min_idx] # lat of min
            depth_min_idx = np.nanargmax(atl_south[lat_min_idx, :]) # depth idx of min
            depth_min = depth[depth_mask][depth_min_idx] # depth of min
            amoc_min = atl_south[lat_min_idx, depth_min_idx] # AMOC min
     
            # --- AMOC 34.5°S ---
            so_lat_idx = np.argmin(np.abs(lat - SOlat))
            atl_34s = atl_t[so_lat_idx, depth_mask]
            depth_amoc_345_idx = np.nanargmax(atl_34s)
            depth_amoc_345 = depth[depth_mask][depth_amoc_345_idx]
            amoc_345 = atl_34s[depth_amoc_345_idx]
    
            # --- SO upper ---
            glo_34s = glo_t[so_lat_idx, depth_mask]
            depth_so_345_idx = np.nanargmax(glo_34s)
            depth_so_345 = depth[depth_mask][depth_so_345_idx]
            so_345 = glo_34s[depth_so_345_idx]
    
            # --- PMOC zAMOC345 & PMOC_34S ---
            indo_34s = indo_t[so_lat_idx, depth_mask]
            pmoc_zAMOC345 = indo_t[so_lat_idx, np.argmin(np.abs(depth - depth_amoc_345))]
            depth_pmoc_345_idx = np.nanargmax(np.abs(indo_34s))
            depth_pmoc_345 = depth[depth_mask][depth_pmoc_345_idx]
            pmoc_345 = indo_34s[depth_pmoc_345_idx]
            pmoc_zSO345 = indo_t[so_lat_idx, np.argmin(np.abs(depth - depth_so_345))]


            # ---- PMOC z_SouthAtlanticLocalBottom ---
            # --- Calculate depth for South Atlantic local bottom ---
            # Slice atlantic along depth from depth_amoc_345 to bottom
            # atlantic_lat_34s = atl_t[np.ix_(lat == SOlat, depth_mask)]  # Atlantic at 34.5°S
            
            # Apply the condition where AMOC values are less than or equal to AMOC min
            depth_below_amoc345_mask = depth[depth_mask]>depth_amoc_345
            depth_south_atlantic_local_bottom_mask = atl_34s[depth_below_amoc345_mask] <= amoc_min
        
            # Find the shallowest depth where the condition is true
            depth_south_atlantic_local_bottom_idx = np.nanargmin(depth_south_atlantic_local_bottom_mask)
            depth_south_atlantic_local_bottom = depth[depth_mask][depth_below_amoc345_mask][depth_south_atlantic_local_bottom_idx]
        
            # --- PMOC at depth_south_atlantic_local_bottom ---
            pmoc_zsouth_atlantic_local_bottom = indo_34s[depth_south_atlantic_local_bottom_idx]

            
        
            return np.array([
                lat_max, depth_max, amoc_max,
                lat_min, depth_min, amoc_min,
                depth_amoc_345, amoc_345,
                depth_so_345, so_345,
                pmoc_zAMOC345, depth_pmoc_345, pmoc_345,
                depth_south_atlantic_local_bottom, pmoc_zsouth_atlantic_local_bottom,
                pmoc_zSO345
            ])

    
        # --------------------------------------------------
        # Use xarray.apply_ufunc to broadcast over 'time'
        # --------------------------------------------------
        results = xr.apply_ufunc(
            _calc_amoc_points,
            glo, atlantic, indopac,
            glo["lat"], glo[self.depth_name],
            kwargs=dict(
                SOlat=self.SOlat,
                min_lat=self.min_amoc_max_lat,
                min_depth=self.min_depth,
            ),
            input_core_dims=[["lat", self.depth_name]] * 3 + [["lat"], [self.depth_name]],
            output_core_dims=[["metric"]],
            vectorize=True,  # allow iteration over time and other dims
            dask="parallelized",  # works with dask-backed arrays
            output_dtypes=[float],
        )
    
        # Label outputs
        metric_names = [
            "lat_max", "depth_max", "amoc_max",
            "lat_min", "depth_min", "amoc_min",
            "depth_amoc_345", "amoc_345",
            "depth_so_345", "so_345",
            "pmoc_zAMOC345", "depth_pmoc_345", "pmoc_345",
            "depth_SAlb", "pmoc_zSAlb",
            "pmoc_zSO345"
        ]
        results = results.assign_coords(metric=metric_names)
    
        # Convert to dataset with one variable per metric
        points = results.to_dataset(dim="metric")

        
        return points


### plot_panel
    def plot_panel(self, glo=None, atlantic=None, indopac=None, 
                   split=True, weddel=None, scatter=False, arrows=False, layers=None, lat_zoom=None,
                   savefig=False, folder=None, run=None):
        """
        Plot the Meridional Overturning Circulation (MOC) streamfunction for different ocean basins.

        Args:
        - atlantic (xarray.DataArray, optional): MOC data for the Atlantic Ocean. If not provided, this will be computed as the 57-year mean.
        - indopac (xarray.DataArray, optional): MOC data for the Indo-Pacific Ocean. If not provided, this will be computed as the 57-year mean.
        - glo (xarray.DataArray, optional): MOC data for the Southern Ocean. If not provided, this will be computed as the 57-year mean.
        - split (bool, optional): Whether to create a two-panel plot (Atlantic/Indopac with Southern Ocean south of self.SOlat) [True] or plot each basin separately [False].
        - weddel (bool, optional): Whether to include Weddell Sea difference plot in the Atlantic basin (only of split is False).
        - scatter (bool, optional): Whether to include scatter points for MOC maxima/minima (Baker index points).
        - arrows (bool, optional): Whether to include arrows showing flow direction and magnitude along self.SOlat.
        - layers (int or slice, optional): Select specific depth layers for plotting. If int, layers will be plotted from the surface until the given depth index.
        - lat_zoom (slice, optional): Latitude range to zoom in on.
        - savefig (bool, optional): Whether to save the plot to a file.
        - folder (str, optional): Folder to save the figure in (if savefig is True).
        - run (str, optional): A string to include in the file name if saving the figure.

        Returns:
        tuple: A tuple containing the figure and axis objects (fig, ax).
        
        """

        # Check that correct arguments were given
        if split is False and any(arg is not False for arg in [arrows]):
            raise ValueError("Argument and 'arrows' can only be used when 'split' is True.")
        if savefig and any(arg is None for arg in [folder, run]):
            raise ValueError("Arguments 'folder' and 'run' must be given to save the figure correctly.")

        if split:
            weddel = False
        elif split is False and weddel is None:
            weddel = True

        if any(x is None for x in [glo, atlantic, indopac]):
            glo, atlantic, indopac = self.calc_mean_moc()
            if scatter:
                points = self.calc_points(glo, atlantic, indopac)
        
        if not isinstance(glo, list):
            atlantic = [atlantic]
            indopac = [indopac]
            glo = [glo]

        n_sim = len(glo) # number of simulations for comparison
        depth = glo[0][self.depth_name].values

        # Create figure
        n = 2 if split else 3 # number of panels
        
        fig = plt.figure(figsize=(12*n_sim, 6*n))
        gs = GridSpec(n, n_sim)
        
        for sim in range(n_sim):

            glo_sim = glo[sim]
            atlantic_sim = atlantic[sim]
            indopac_sim = indopac[sim]

            if scatter:
                points = self.calc_points(glo_sim, atlantic_sim, indopac_sim)       
            
            if lat_zoom is not None:
                # compute points if specified
                if isinstance(lat_zoom, slice):
                    glo_sim = glo_sim.sel(lat=lat_zoom)
                    atlantic_sim = atlantic_sim.sel(lat=lat_zoom)
                    indopac_sim = indopac_sim.sel(lat=lat_zoom)
                else:
                    raise ValueError("'lat_zoom' must be of type slice.")
            
            # Slice layers if specified
            if layers is not None:
                if isinstance(layers, int):
                    layers=slice(0,layers)
                if not isinstance(layers, slice):
                    raise ValueError("'layers' must be of type int or slice.")
                glo_sim = glo_sim.isel({self.depth_name: layers})
                atlantic_sim = atlantic_sim.isel({self.depth_name: layers})
                indopac_sim = indopac_sim.isel({self.depth_name: layers})
                depth = glo_sim[self.depth_name].values
        
            basin_title=["Atlantic", "Indo-Pacific", "Southern Ocean"]
    
            # Define for different cases (2 or 3 panels)
            mocs = [atlantic_sim.sel(lat=slice(self.SOlat, None)), indopac_sim.sel(lat=slice(self.SOlat, None))] if split else [atlantic_sim, indopac_sim, glo_sim] # mocs to plot in each panel
            glo_sim = glo_sim.sel(lat=slice(None, self.SOlat)) if split else glo_sim # slice Southern Ocean if needed

            
            for y, moc, title in zip(np.arange(n), mocs, basin_title[:n]):
                ax = fig.add_subplot(gs[y, sim])
                cf = ax.contourf(moc['lat'], depth, moc, levels=self.clevels, cmap='RdBu_r', vmin=-self.vmax, vmax=self.vmax) # contour colors
                cs = ax.contour(moc['lat'], depth, moc, levels=self.clevels, colors='grey', linewidths=0.5) # contour lines
                ax.contour(moc['lat'], depth, moc, levels=[0], colors='black', linewidths=1.0)  # 0 contour line as thicker black
                ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f') # contour labels
    
                # only plot Southern Ocean if SOlat is within the plotted latitudes
                if split and (lat_zoom is None or lat_zoom.start<self.SOlat):
                    ax.contourf(glo_sim['lat'], depth, glo_sim, levels=self.clevels, cmap='RdBu_r', vmin=-self.vmax, vmax=self.vmax) # contour colors
                    cs_so = ax.contour(glo_sim['lat'], depth, glo_sim, levels=self.clevels, colors='grey', linewidths=0.5) # contour lines 
                    ax.contour(glo_sim['lat'], depth, glo_sim, levels=[0], colors='black', linewidths=1.0) # 0 contour as thicker black
                    ax.axvline(self.SOlat, color='black', linestyle='dotted') # vertical line to separate Sounthern Ocean from Atlantic/Indopacific
                    ax.clabel(cs_so, inline=True, fontsize=8, fmt='%1.0f') # contour labels
    
                ax.invert_yaxis()
                ax.set_title(title, fontsize=16)
                ax.set_xlabel('Latitude', fontsize=14)
                ax.set_ylabel('Depth (m)', fontsize=14)
                
                
                cbar = fig.colorbar(cf, ax=ax, orientation='vertical')
                cbar.set_label(label='Meridional Transport (Sv)', fontsize=14)
                              
            
                # Optional Weddel sea
                if weddel and y==0:
                    nan_mask = atlantic_sim.isnull() & (atlantic_sim['lat'] < -40)
                    wed = glo_sim.where(nan_mask) - indopac_sim.where(nan_mask)
                    if layers:
                        wed = wed.isel({self.depth_name: layers})
                    ax.contourf(atlantic_sim['lat'], depth, wed,
                                   levels=self.clevels, cmap='RdBu_r', vmin=-self.vmax, vmax=self.vmax)
                    cs_wed = ax.contour(atlantic_sim['lat'], depth, wed,
                                  levels=self.clevels, colors='grey', linewidths=0.5)
                    ax.contour(atlantic_sim['lat'], depth, wed,
                                  levels=[0], colors='black', linewidths=1.0)
                    ax.clabel(cs_wed, inline=True, fontsize=8, fmt='%1.0f')

            
                # Scatter Baker index points if desired (only in Atlantic and Indopacific panels)
                if scatter:
                    s = 100
                    if y==0:
                        ax.scatter(points.lat_max, points.depth_max, c='purple', s=s, label=f'AMOC max={points.amoc_max:.2f}') # AMOC max
                        ax.scatter(points.lat_min, points.depth_min, c='deepskyblue', s=s, label=f'AMOC min={points.amoc_min:.2f}') # AMOC min
                        ax.scatter(self.SOlat, points.depth_amoc_345, c='red', s=s, label=f'AMOC 34.5°S={points.amoc_345:.2f}') # AMOC 34.5
                        ax.scatter(self.SOlat, points.depth_so_345, c='darkorange', s=s, label=f'SO 34.5°S={points.so_345:.2f}') # SO 34.5
                        ax.legend(loc='lower right')
                    if y==1:
                        c,e = ('yellowgreen','w') if points.pmoc_zAMOC345 > 0 else ('honeydew','yellowgreen') # pmoc_zAMOC345 only goes into original Baker calculation if > 0
                        ax.scatter(self.SOlat, points.depth_amoc_345, c=c, edgecolors=e, s=s, label=f'PMOC zAMOC 34.5°S={points.pmoc_zAMOC345:.2f}') # SO 34.5
                        # ax.scatter(self.SOlat, points.depth_pmoc_345, c='darkgreen', s=s, label=f'PMOC 34.5°S={points.pmoc_345:.2f}') # PMOC max 34.5
                        ax.legend(loc='lower right')
                        
            
                # Arrows for flow direction
                if arrows:
                    if y==0:
                        self.draw_arrows([ax], atlantic_sim, basin="atl")
                        self.draw_arrows([ax], glo_sim, basin="glo")
                    if y==1:
                        self.draw_arrows([ax], indopac_sim, basin="indopac")
                        self.draw_arrows([ax], glo_sim, basin="glo")
        
                if layers is None:
                    ax.set_ylim([6000, 0])
                else:
                    ylims = [depth[layers.stop-1], depth[layers.start]]
                    ax.set_ylim(ylims)
        
                if lat_zoom is not None:
                    ax.set_xlim([lat_zoom.start, lat_zoom.stop])
        
        
        if len(atlantic)==1:
            ax_pos = ax.get_position()
            fig.suptitle(self.suptitle, color=self.suptitle_color, x=ax_pos.x0 + ax_pos.width / 2)
        
        plt.tight_layout()

        # Save figure if requested
        if savefig:
            os.makedirs(folder, exist_ok=True)
            timestamp = atlantic_sim.time.dt.year.values if 'time' in atlantic_sim.dims else 0
            fname = f"moc_split_mean_{run}_{timestamp}.png"
            plt.savefig(os.path.join(folder, fname), dpi=300, bbox_inches='tight')
            if ax is None:
                plt.close(fig) # Close only if we created the figure
        
        return fig, ax


    def plot_panel_new(self, glo=None, atlantic=None, indopac=None, 
                       split=True, weddel=None, scatter=False, arrows=False, layers=None, lat_zoom=None,
                       savefig=False, folder=None, run=None): 
        """
        Plot the Meridional Overturning Circulation (MOC) streamfunction for different ocean basins.

        Args:
        - atlantic (xarray.DataArray, optional): MOC data for the Atlantic Ocean. If not provided, this will be computed as the 57-year mean.
        - indopac (xarray.DataArray, optional): MOC data for the Indo-Pacific Ocean. If not provided, this will be computed as the 57-year mean.
        - glo (xarray.DataArray, optional): MOC data for the Southern Ocean. If not provided, this will be computed as the 57-year mean.
        - split (bool, optional): Whether to create a two-panel plot (Atlantic/Indopac with Southern Ocean south of self.SOlat) [True] or plot each basin separately [False].
        - weddel (bool, optional): Whether to include Weddell Sea difference plot in the Atlantic basin (only of split is False).
        - scatter (bool, optional): Whether to include scatter points for MOC maxima/minima (Baker index points).
        - arrows (bool, optional): Whether to include arrows showing flow direction and magnitude along self.SOlat.
        - layers (int or slice, optional): Select specific depth layers for plotting. If int, layers will be plotted from the surface until the given depth index.
        - lat_zoom (slice, optional): Latitude range to zoom in on.
        - savefig (bool, optional): Whether to save the plot to a file.
        - folder (str, optional): Folder to save the figure in (if savefig is True).
        - run (str, optional): A string to include in the file name if saving the figure.

        Returns:
        tuple: A tuple containing the figure and axis objects (fig, ax).
        
        """

        # Check that correct arguments were given
        if split is False and any(arg is not False for arg in [arrows]):
            raise ValueError("Argument and 'arrows' can only be used when 'split' is True.")
        if savefig and any(arg is None for arg in [folder, run]):
            raise ValueError("Arguments 'folder' and 'run' must be given to save the figure correctly.")

        if split:
            weddel = False
        elif split is False and weddel is None:
            weddel = True

        if any(x is None for x in [glo, atlantic, indopac]):
            glo, atlantic, indopac = self.calc_mean_moc()
            if scatter:
                points = self.calc_points(glo, atlantic, indopac)
        
        if not isinstance(glo, list):
            atlantic = [atlantic]
            indopac = [indopac]
            glo = [glo]

        n_sim = len(glo) # number of simulations for comparison
        depth = glo[0][self.depth_name].values

        # Create figure
        n = 2 if split else 3 # number of panels

        # Decide whether we want a single shared colorbar
        add_cbar = (n_sim > 1) or (n_sim == 1 and isinstance(self.suptitle, str) and ("Ocean" in self.suptitle)) or (arrows)
        
        # Create figure + gridspec (add an extra column for the colorbar when needed)
        if n_sim < 4:
            n_rows = 2 if split else 3
            n_cols = n_sim + (1 )#if add_cbar else 0)
        elif n_sim == 50:
            n_rows = 20 if split else 30
            n_cols = 5 + 1
        elif n_sim ==25:
            n_rows = 10 if split else 15
            n_cols = 5 +1
        elif n_sim == 16:
            n_rows = 8 if split else 12
            n_cols = 4 + 1
        
        # Make figure slightly wider only when we actually add a colorbar
        fig_w = 12 * (n_cols-1) + (0.05 if add_cbar else 0.0)
        fig_h = 6 * n_rows
        fig = plt.figure(figsize=(fig_w, fig_h))

        
        
        if True:#add_cbar:        
            width_ratios = [1] * (n_cols-1) + [0.05]  # last column reserved for colorbar
            gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios, wspace=0.15)
        else:
            gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=[1]*n_sim, wspace=0.15)
            cax = None
        
        # Optional: keep axes if you want later access
        axes = [[None for _ in range(n_sim)] for _ in range(n_rows)]
        
        # We'll use a single ScalarMappable so the shared colorbar is consistent
        #sm = ScalarMappable(norm=Normalize(vmin=-self.vmax, vmax=self.vmax), cmap="RdBu_r")
        #sm.set_array([])
        
        for sim in range(n_sim):
            row = (sim // (n_cols-1)) *2
        
            glo_sim = glo[sim]
            atlantic_sim = atlantic[sim]
            indopac_sim = indopac[sim]
        
            if scatter:
                points = self.calc_points(glo_sim, atlantic_sim, indopac_sim)                   
        
            if lat_zoom is not None:
                # compute points if specified
    
                if isinstance(lat_zoom, slice):
                    glo_sim = glo_sim.sel(lat=lat_zoom)
                    atlantic_sim = atlantic_sim.sel(lat=lat_zoom)
                    indopac_sim = indopac_sim.sel(lat=lat_zoom)
                else:
                    raise ValueError("'lat_zoom' must be of type slice.")
            
            # Slice layers if specified
            if layers is not None:
                if isinstance(layers, int):
                    layers=slice(0,layers)
                if not isinstance(layers, slice):
                    raise ValueError("'layers' must be of type int or slice.")
                glo_sim = glo_sim.isel({self.depth_name: layers})
                atlantic_sim = atlantic_sim.isel({self.depth_name: layers})
                indopac_sim = indopac_sim.isel({self.depth_name: layers})
                depth = glo_sim[self.depth_name].values
            
            basin_title=["Atlantic", "Indo-Pacific", "Southern Ocean"]
        
            mocs = (
                [atlantic_sim.sel(lat=slice(self.SOlat, None)),
                 indopac_sim.sel(lat=slice(self.SOlat, None))]
                if split else
                [atlantic_sim, indopac_sim, glo_sim]
            )
            glo_sim = glo_sim.sel(lat=slice(None, self.SOlat)) if split else glo_sim

            for y, moc, title in zip(np.arange(n_rows), mocs, basin_title[:n_rows]):
                ax = fig.add_subplot(gs[y+row, sim%(n_cols-1)])
                axes[y+row][sim//n_cols] = ax
                
                cf = ax.contourf(
                    moc["lat"], depth, moc,
                    levels=self.clevels, cmap="RdBu_r",
                    vmin=-self.vmax, vmax=self.vmax, extend='both'
                )
                cs = ax.contour(moc["lat"], depth, moc, levels=self.clevels, colors="grey", linewidths=0.5)
                ax.contour(moc["lat"], depth, moc, levels=[0], colors="black", linewidths=1.0)
                ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0f")
        
                if split and (lat_zoom is None or lat_zoom.start < self.SOlat):
                    ax.contourf(
                        glo_sim["lat"], depth, glo_sim,
                        levels=self.clevels, cmap="RdBu_r",
                        vmin=-self.vmax, vmax=self.vmax
                    )
                    cs_so = ax.contour(glo_sim["lat"], depth, glo_sim, levels=self.clevels, colors="grey", linewidths=0.5)
                    ax.contour(glo_sim["lat"], depth, glo_sim, levels=[0], colors="black", linewidths=1.0)
                    ax.axvline(self.SOlat, color="black", linestyle="dotted")
                    ax.clabel(cs_so, inline=True, fontsize=8, fmt="%1.0f")
        
                ax.invert_yaxis()
                ax.set_title(f'{title} r{sim+1}' if n_sim>3 else title, fontsize=16)
                ax.set_xlabel("Latitude", fontsize=14)
                ax.set_ylabel("Depth (m)", fontsize=14)
        
                # REMOVE the per-axis colorbar entirely
                # (no fig.colorbar(cf, ax=ax, ...) here)
        
                # Optional Weddel sea
                if weddel and y==0:
                    nan_mask = atlantic_sim.isnull() & (atlantic_sim['lat'] < -40)
                    wed = glo_sim.where(nan_mask) - indopac_sim.where(nan_mask)
                    if layers:
                        wed = wed.isel({self.depth_name: layers})
                    ax.contourf(atlantic_sim['lat'], depth, wed,
                                   levels=self.clevels, cmap='RdBu_r', vmin=-self.vmax, vmax=self.vmax)
                    cs_wed = ax.contour(atlantic_sim['lat'], depth, wed,
                                  levels=self.clevels, colors='grey', linewidths=0.5)
                    ax.contour(atlantic_sim['lat'], depth, wed,
                                  levels=[0], colors='black', linewidths=1.0)
                    ax.clabel(cs_wed, inline=True, fontsize=8, fmt='%1.0f')
    
            
                # Scatter Baker index points if desired (only in Atlantic and Indopacific panels)
                if scatter:
                    s = 100
                    if y%2==0:
                        ax.scatter(points.lat_max, points.depth_max, c='purple', s=s, label=f'AMOC max={points.amoc_max:.2f}') # AMOC max
                        ax.scatter(points.lat_min, points.depth_min, c='deepskyblue', s=s, label=f'AMOC min={points.amoc_min:.2f}') # AMOC min
                        ax.scatter(self.SOlat, points.depth_amoc_345, c='red', s=s, label=f'AMOC 34.5°S={points.amoc_345:.2f}') # AMOC 34.5
                        ax.scatter(self.SOlat, points.depth_so_345, c='darkorange', s=s, label=f'SO 34.5°S={points.so_345:.2f}') # SO 34.5
                        ax.legend(loc='lower right')
                    if y%2==1:
                        # c,e = ('yellowgreen','w') if points.pmoc_zAMOC345 > 0 else ('honeydew','yellowgreen') # pmoc_zAMOC345 only goes into original Baker calculation if > 0
                        c,e = ('yellowgreen','w') if points.pmoc_zAMOC345 > 0 else ('yellowgreen','w') # pmoc_zAMOC345 only goes into original Baker calculation if > 0
                        ax.scatter(self.SOlat, points.depth_amoc_345, c=c, edgecolors=e, s=s, label=f'PMOC zAMOC 34.5°S={points.pmoc_zAMOC345:.2f}') # SO 34.5
                        # ax.scatter(self.SOlat, points.depth_pmoc_345, c='darkgreen', s=s, label=f'PMOC 34.5°S={points.pmoc_345:.2f}') # PMOC max 34.5
                        ax.legend(loc='lower right')
                        
            
                # Arrows for flow direction
                if arrows:
                    if y==0:
                        self.draw_arrows([ax], atlantic_sim, basin="atl")
                        self.draw_arrows([ax], glo_sim, basin="glo")
                    if y==1:
                        self.draw_arrows([ax], indopac_sim, basin="indopac")
                        self.draw_arrows([ax], glo_sim, basin="glo")
        
                if layers is None:
                    ax.set_ylim([6000, 0])
                else:
                    ylims = [depth[layers.stop - 1], depth[layers.start]]
                    ax.set_ylim(ylims)
        
                if lat_zoom is not None:
                    ax.set_xlim([lat_zoom.start, lat_zoom.stop])
        
        # Add ONE shared colorbar (only if add_cbar is True)
        if add_cbar: #and cax is not None:
            cax = fig.add_subplot(gs[:, -1])     # one colorbar axis spanning all rows
            cbar = fig.colorbar(cf, cax=cax, orientation="vertical")
            cbar.set_label("Meridional Transport (Sv)", fontsize=14)
        
        # Suptitle centering: keep your current logic, but it’s usually cleaner now:
        if n_sim == 1:
            # center over the plot column (not the colorbar column)
            ax_pos = axes[0][0].get_position()
            fig.suptitle(self.suptitle, color=self.suptitle_color, x=ax_pos.x0 + ax_pos.width / 2, y=0.94)
        
        #plt.tight_layout()
        
        # Save figure if requested (unchanged)
        if savefig:
            os.makedirs(folder, exist_ok=True)
            timestamp = atlantic_sim.time.dt.year.values if "time" in atlantic_sim.dims else 0
            fname = f"moc_split_mean_{run}_{timestamp}.png"
            plt.savefig(os.path.join(folder, fname), dpi=300, bbox_inches="tight")
            if ax is None:
                plt.close(fig)
        
        return fig, ax

    def plot_panel_v3(self, glo=None, atlantic=None, indopac=None, 
                           split=True, weddel=None, scatter=False, arrows=False, layers=None, lat_zoom=None,
                           savefig=False, folder=None, run=None, add_cbar=False): 
            """
            Plot the Meridional Overturning Circulation (MOC) streamfunction for different ocean basins.
    
            Args:
            - atlantic (xarray.DataArray, optional): MOC data for the Atlantic Ocean. If not provided, this will be computed as the 57-year mean.
            - indopac (xarray.DataArray, optional): MOC data for the Indo-Pacific Ocean. If not provided, this will be computed as the 57-year mean.
            - glo (xarray.DataArray, optional): MOC data for the Southern Ocean. If not provided, this will be computed as the 57-year mean.
            - split (bool, optional): Whether to create a two-panel plot (Atlantic/Indopac with Southern Ocean south of self.SOlat) [True] or plot each basin separately [False].
            - weddel (bool, optional): Whether to include Weddell Sea difference plot in the Atlantic basin (only of split is False).
            - scatter (bool, optional): Whether to include scatter points for MOC maxima/minima (Baker index points).
            - arrows (bool, optional): Whether to include arrows showing flow direction and magnitude along self.SOlat.
            - layers (int or slice, optional): Select specific depth layers for plotting. If int, layers will be plotted from the surface until the given depth index.
            - lat_zoom (slice, optional): Latitude range to zoom in on.
            - savefig (bool, optional): Whether to save the plot to a file.
            - folder (str, optional): Folder to save the figure in (if savefig is True).
            - run (str, optional): A string to include in the file name if saving the figure.
    
            Returns:
            tuple: A tuple containing the figure and axis objects (fig, ax).
            
            """
    
            # Check that correct arguments were given
            if split is False and any(arg is not False for arg in [arrows]):
                raise ValueError("Argument and 'arrows' can only be used when 'split' is True.")
            if savefig and any(arg is None for arg in [folder, run]):
                raise ValueError("Arguments 'folder' and 'run' must be given to save the figure correctly.")
    
            if split:
                weddel = False
            elif split is False and weddel is None:
                weddel = True
    
            if any(x is None for x in [glo, atlantic, indopac]):
                glo, atlantic, indopac = self.calc_mean_moc()
                if scatter:
                    points = self.calc_points(glo, atlantic, indopac)
            
            if not isinstance(glo, list):
                atlantic = [atlantic]
                indopac = [indopac]
                glo = [glo]
    
            n_sim = len(glo) # number of simulations for comparison
            depth = glo[0][self.depth_name].values
    
            # number of basin-columns
            n_basins = 2 if split else 3
            basin_title = ["Atlantic", "Indo-Pacific", "Southern Ocean"]
    
            # Determine whether we want a single shared colorbar
            if add_cbar==False:
                add_cbar = (n_sim > 1) or (n_sim == 1 and isinstance(self.suptitle, str) and ("Ocean" in self.suptitle or "OCE" in self.suptitle)) or (arrows) 
    
            # Make figure size
            fig_w = 10 * n_basins
            # Add a small extra vertical space if we will place colorbar below
            fig_h = 5 * n_sim + (1.5) # if add_cbar else 0)
            fig = plt.figure(figsize=(fig_w, fig_h))
    
            # GridSpec: rows = n_sim (plots) + (1 if add_cbar else 0) for colorbar
            gs_rows = n_sim + (1 ) # if add_cbar else 0)
            # Use a small height ratio for the colorbar row (last row)
            if True: # add_cbar:
                height_ratios = [5] * n_sim + [.75]
                gs = GridSpec(gs_rows, n_basins, figure=fig, height_ratios=height_ratios, hspace=0.4, wspace=0.12) 
            else:
                gs = GridSpec(gs_rows, n_basins, figure=fig, wspace=0.12) #hspace=0.4,
    
            # Optional: keep axes if you want later access; shape [n_sim][n_basins]
            axes = np.array([[None for _ in range(n_basins)] for _ in range(n_sim)])
    
            for r, sim in enumerate(range(n_sim)):
                glo_sim = glo[sim]
                atlantic_sim = atlantic[sim]
                indopac_sim = indopac[sim]
            
                if scatter:
                    points = self.calc_points(glo_sim, atlantic_sim, indopac_sim)                   
            
                if lat_zoom is not None:
                    # compute points if specified
                    if isinstance(lat_zoom, slice):
                        glo_sim = glo_sim.sel(lat=lat_zoom)
                        atlantic_sim = atlantic_sim.sel(lat=lat_zoom)
                        indopac_sim = indopac_sim.sel(lat=lat_zoom)
                    else:
                        raise ValueError("'lat_zoom' must be of type slice.")
                
                # Slice layers if specified
                if layers is not None:
                    if isinstance(layers, int):
                        layers = slice(0, layers)
                    if not isinstance(layers, slice):
                        raise ValueError("'layers' must be of type int or slice.")
                    glo_sim = glo_sim.isel({self.depth_name: layers})
                    atlantic_sim = atlantic_sim.isel({self.depth_name: layers})
                    indopac_sim = indopac_sim.isel({self.depth_name: layers})
                    depth = glo_sim[self.depth_name].values
    
                # Build the list of mocs for the columns (basins)
                if split:
                    # For split mode, show Atlantic & Indo-Pacific as columns, and overlay Southern Ocean south of SOlat
                    mocs = [
                        atlantic_sim.sel(lat=slice(self.SOlat, None)),
                        indopac_sim.sel(lat=slice(self.SOlat, None))
                    ]
                else:
                    mocs = [atlantic_sim, indopac_sim, glo_sim]
    
                for col, (moc, title) in enumerate(zip(mocs, basin_title[:n_basins])):
                    ax = fig.add_subplot(gs[sim, col])
                    axes[sim][col] = ax
    
                    cf = ax.contourf(
                        moc["lat"], depth, moc,
                        levels=self.clevels, cmap="RdBu_r",
                        vmin=-self.vmax, vmax=self.vmax, extend='both'
                    )
                    cs = ax.contour(moc["lat"], depth, moc, levels=self.clevels, colors="grey", linewidths=0.5)
                    ax.contour(moc["lat"], depth, moc, levels=[0], colors="black", linewidths=1.0)
                    ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0f")
    
                    # If split, overlay the Southern Ocean portion (south of SOlat) on each basin column
                    if split:
                        # glo_sim is the Southern Ocean; select latitudes south of SOlat (lower lat values)
                        so_part = glo_sim.sel(lat=slice(None, self.SOlat))
                        # Only overlay if lat_zoom isn't exclusive north of SOlat
                        if lat_zoom is None or lat_zoom.start < self.SOlat:
                            ax.contourf(
                                so_part["lat"], depth, so_part,
                                levels=self.clevels, cmap="RdBu_r",
                                vmin=-self.vmax, vmax=self.vmax
                            )
                            cs_so = ax.contour(so_part["lat"], depth, so_part, levels=self.clevels, colors="grey", linewidths=0.5)
                            ax.contour(so_part["lat"], depth, so_part, levels=[0], colors="black", linewidths=1.0)
                            ax.axvline(self.SOlat, color="black", linestyle="dotted")
                            ax.clabel(cs_so, inline=True, fontsize=8, fmt="%1.0f")
    
                    ax.invert_yaxis()
                    
                    if n_sim>3:
                        ax.set_title(f'{title} r{sim+1}', fontsize=14)

                    if "Historical" in self.suptitle or "HIST" in self.suptitle or arrows:
                        ax.set_title(title, pad=20)
                    
                    if add_cbar:
                        ax.set_xlabel("Latitude")
                    if col==0:
                        ax.set_ylabel("Depth (m)")
    
                    # Optional Weddel sea (only meaningful when not split and only in Atlantic column)
                    if weddel and (not split) and col == 0:
                        nan_mask = atlantic_sim.isnull() & (atlantic_sim['lat'] < -40)
                        wed = glo_sim.where(nan_mask) - indopac_sim.where(nan_mask)
                        if layers:
                            wed = wed.isel({self.depth_name: layers})
                        ax.contourf(atlantic_sim['lat'], depth, wed,
                                       levels=self.clevels, cmap='RdBu_r', vmin=-self.vmax, vmax=self.vmax)
                        cs_wed = ax.contour(atlantic_sim['lat'], depth, wed,
                                      levels=self.clevels, colors='grey', linewidths=0.5)
                        ax.contour(atlantic_sim['lat'], depth, wed,
                                      levels=[0], colors='black', linewidths=1.0)
                        ax.clabel(cs_wed, inline=True, fontsize=8, fmt='%1.0f')
    
                    # Scatter Baker index points if desired (only in Atlantic and Indo-Pacific columns)
                    if scatter:
                        s = 100
                        # Points refer to current simulation's computed points
                        if col in (0,):  # Atlantic column: show max/min and AMOC 34.5 etc.
                            ax.scatter(points.lat_max, points.depth_max, c='purple', s=s, label=f'AMOC max={points.amoc_max:.2f}')
                            ax.scatter(points.lat_min, points.depth_min, c='deepskyblue', s=s, label=f'AMOC min={points.amoc_min:.2f}')
                            ax.scatter(self.SOlat, points.depth_amoc_345, c='red', s=s, label=f'AMOC 34.5°S={points.amoc_345:.2f}')
                            ax.scatter(self.SOlat, points.depth_so_345, c='darkorange', s=s, label=f'SO 34.5°S={points.so_345:.2f}')
                            ax.legend(loc='lower right')
                        if col in (1,):  # Indo-Pacific column: show PMOC-related point(s)
                            c,e = ('yellowgreen','w') if points.pmoc_zAMOC345 > 0 else ('yellowgreen','w')
                            ax.scatter(self.SOlat, points.depth_amoc_345, c=c, edgecolors=e, s=s, label=f'PMOC zAMOC 34.5°S={points.pmoc_zAMOC345:.2f}')
                            ax.legend(loc='lower right')
    
                    # Arrows for flow direction: only allowed when split=True (checked above)
                    if arrows and split:
                        if col == 0:
                            self.draw_arrows([ax], atlantic_sim, basin="atl")
                            self.draw_arrows([ax], glo_sim, basin="glo")
                        if col == 1:
                            self.draw_arrows([ax], indopac_sim, basin="indopac")
                            self.draw_arrows([ax], glo_sim, basin="glo")
    
                    if layers is None:
                        ax.set_ylim([6000, 0])
                    else:
                        ylims = [depth[layers.stop - 1], depth[layers.start]]
                        ax.set_ylim(ylims)
    
                    if lat_zoom is not None:
                        ax.set_xlim([lat_zoom.start, lat_zoom.stop])

                    
    
            # Add ONE shared horizontal colorbar below the plots (only if add_cbar is True)
            if add_cbar:
                cax = fig.add_subplot(gs[-1, :])  # last row, span all basin columns
                cbar = fig.colorbar(cf, cax=cax, orientation="horizontal")
                cbar.set_label("Meridional Transport (Sv)")
    
            # Suptitle centering: center across full figure
            # if isinstance(self.suptitle, str) and self.suptitle:
            #    fig.suptitle(self.suptitle, color=self.suptitle_color, x=0.5, y=0.96, fontsize=16)

            # vertical row titles (leftmost column only)     

            for r, ax in enumerate(axes[:,0]):
                if axes.shape[0]==1:
                    suptitle = [self.suptitle]
                    color = [self.suptitle_color]
                else:
                    suptitle = ['HIST', 'ATM', 'OCE']
                    color = ['crimson', 'forestgreen', 'mediumblue']
                ax.text(
                    -0.17, 0.5, suptitle[r],
                    color=color[r],
                    transform=ax.transAxes,
                    rotation=90,
                    va='center', ha='center',
                    fontsize=28, fontweight='bold'
                )
    
            # Save figure if requested
            if savefig:
                os.makedirs(folder, exist_ok=True)
                timestamp = atlantic_sim.time.dt.year.values if "time" in atlantic_sim.dims else 0
                fname = f"moc_split_mean_{run}_{timestamp}.png"
                plt.savefig(os.path.join(folder, fname), dpi=300, bbox_inches="tight")
                if ax is None:
                    plt.close(fig)

            return fig, ax

    def plot_panel_members(self, glo=None, atlantic=None, indopac=None, 
                           layers=None, lat_zoom=None, savefig=False, folder=None, run=None):
        """
        Plot the Meridional Overturning Circulation (MOC) streamfunction for multiple ensemble members.
        
        Args:
        - atlantic (list of xarray.DataArray): MOC data for the Atlantic Ocean for each member.
        - indopac (list of xarray.DataArray): MOC data for the Indo-Pacific Ocean for each member.
        - glo (list of xarray.DataArray): MOC data for the Southern Ocean for each member.
        - layers (int or slice, optional): Select specific depth layers for plotting.
        - lat_zoom (slice, optional): Latitude range to zoom in on.
        - savefig (bool, optional): Whether to save the plot to a file.
        - folder (str, optional): Folder to save the figure in (if savefig is True).
        - run (str, optional): A string to include in the file name if saving the figure.
        
        Returns:
        list: A list containing the figure objects.
        
        """
        
        if savefig and any(arg is None for arg in [folder, run]):
            raise ValueError("Arguments 'folder' and 'run' must be given to save the figure correctly.")
        
        if any(x is None for x in [glo, atlantic, indopac]):
            glo, atlantic, indopac = self.calc_mean_moc()
        
        if not isinstance(glo, list):
            atlantic = [atlantic]
            indopac = [indopac]
            glo = [glo]
        
        n_members = len(glo)
        depth = glo[0][self.depth_name].values
        
        # Determine layout based on number of members
        if n_members == 50:
            # Two figures: 13x4 and 12x4
            configs = [(12, 4), (13, 4)]
            member_ranges = [range(0, 24), range(24, 50)]
        elif n_members == 16:
            # One figure: 8x4
            configs = [(8, 4)]
            member_ranges = [range(0, 16)]
        else:
            raise ValueError(f"Number of members must be 16 or 50, got {n_members}")
        
        figs = []
        
        for fig_idx, (nrows, ncols) in enumerate(configs):
            member_start = member_ranges[fig_idx].start
            member_end = member_ranges[fig_idx].stop
            n_members_in_fig = member_end - member_start
            
            # Make figure size (2 basins per member, so ncols = 4 with spacing)
            fig_w = 10 * ncols + 2  # Extra space for column spacing
            fig_h = 5 * nrows + 2  # Extra space for top titles and colorbar
            fig = plt.figure(figsize=(fig_w, fig_h))
            
            # GridSpec with extra space for column titles at top, plot rows, and colorbar at bottom
            # height_ratios: space for titles, plot rows, space for colorbar
            height_ratios = [0.5] + [5] * nrows + [0.2, 0.75]
            # width_ratios: normal spacing except extra space between col 2 and 3
            width_ratios = [5, 5, 0.6, 5, 5] if ncols == 4 else [5] * ncols
            
            gs = GridSpec(nrows + 3, ncols + 1, figure=fig, 
                          height_ratios=height_ratios, 
                          width_ratios=width_ratios,
                          hspace=0.1, wspace=0.1)
            
            n_members_per_row = ncols // 2
            
            # Track axes for title placement and last contourf for colorbar
            axes_first_row = {}
            last_cf = None
            
            for member_idx in range(member_start, member_end):
                member_num = member_idx + 1  # 1-indexed for display
                
                glo_member = glo[member_idx]
                atlantic_member = atlantic[member_idx]
                indopac_member = indopac[member_idx]
                
                if lat_zoom is not None:
                    if isinstance(lat_zoom, slice):
                        glo_member = glo_member.sel(lat=lat_zoom)
                        atlantic_member = atlantic_member.sel(lat=lat_zoom)
                        indopac_member = indopac_member.sel(lat=lat_zoom)
                    else:
                        raise ValueError("'lat_zoom' must be of type slice.")
                
                # Slice layers if specified
                if layers is not None:
                    if isinstance(layers, int):
                        layers_slice = slice(0, layers)
                    elif isinstance(layers, slice):
                        layers_slice = layers
                    else:
                        raise ValueError("'layers' must be of type int or slice.")
                    glo_member = glo_member.isel({self.depth_name: layers_slice})
                    atlantic_member = atlantic_member.isel({self.depth_name: layers_slice})
                    indopac_member = indopac_member.isel({self.depth_name: layers_slice})
                    depth_member = glo_member[self.depth_name].values
                else:
                    depth_member = depth
                
                # Position within the figure
                relative_idx = member_idx - member_start
                row = relative_idx // n_members_per_row + 1  # +1 for title row
                col_offset = (relative_idx % n_members_per_row) * 2
                
                # Adjust column offset for spacing between col 2 and 3
                if col_offset >= 2:
                    gs_col_offset = col_offset + 1
                else:
                    gs_col_offset = col_offset
                
                # Determine member ID format
                if n_members == 50:
                    member_id = f"r{member_num}i1p1f1"
                else:  # 16 members
                    member_id = f"r{member_num}i8p4"
                
                # Plot Atlantic (first column of the pair)
                ax_atl = fig.add_subplot(gs[row, gs_col_offset])
                cf_atl = ax_atl.contourf(
                    atlantic_member["lat"], depth_member, atlantic_member,
                    levels=self.clevels, cmap="RdBu_r",
                    vmin=-self.vmax, vmax=self.vmax, extend='both'
                )
                last_cf = cf_atl  # Keep reference for colorbar
                
                cs_atl = ax_atl.contour(atlantic_member["lat"], depth_member, atlantic_member, 
                                        levels=self.clevels, colors="grey", linewidths=0.5)
                ax_atl.contour(atlantic_member["lat"], depth_member, atlantic_member, 
                               levels=[0], colors="black", linewidths=1.0)
                ax_atl.clabel(cs_atl, inline=True, fontsize=8, fmt="%1.0f")
                ax_atl.invert_yaxis()
                
                # Overlay Southern Ocean portion
                so_part = glo_member.sel(lat=slice(None, self.SOlat))
                if lat_zoom is None or lat_zoom.start < self.SOlat:
                    ax_atl.contourf(
                        so_part["lat"], depth_member, so_part,
                        levels=self.clevels, cmap="RdBu_r",
                        vmin=-self.vmax, vmax=self.vmax
                    )
                    cs_so = ax_atl.contour(so_part["lat"], depth_member, so_part, 
                                           levels=self.clevels, colors="grey", linewidths=0.5)
                    ax_atl.contour(so_part["lat"], depth_member, so_part, 
                                   levels=[0], colors="black", linewidths=1.0)
                    ax_atl.axvline(self.SOlat, color="black", linestyle="dotted")
                    ax_atl.clabel(cs_so, inline=True, fontsize=8, fmt="%1.0f")
                
                # Only add labels on first column
                
                if True: #col_offset == 0:
                    ax_atl.set_ylabel("Depth (m)", fontsize=20)
                    ax_atl.set_yticklabels(np.arange(0,6000,1000), fontsize=14)
                else:
                    ax_atl.set_yticklabels([])

                if row==nrows:
                    ax_atl.set_xlabel("Latitude", fontsize=20)
                    ax_atl.set_xticklabels(np.arange(-80,80,20), fontsize=14)
                else:
                    ax_atl.set_xticklabels([])
                
                if layers is None:
                    ax_atl.set_ylim([6000, 0])
                else:
                    ylims = [depth_member[layers_slice.stop - 1], depth_member[layers_slice.start]]
                    ax_atl.set_ylim(ylims)
                
                if lat_zoom is not None:
                    ax_atl.set_xlim([lat_zoom.start, lat_zoom.stop])
                
                # Add member ID label to the left of Atlantic column
                ax_atl.text(
                    -0.2, 0.5, member_id,
                    color='black',
                    transform=ax_atl.transAxes,
                    rotation=90,
                    va='center', ha='center',
                    fontsize=20, fontweight='bold'
                )
                
                # Store first row axes for title placement
                if row == 1:
                    axes_first_row[f'atl_{col_offset}'] = ax_atl
                
                
                # Plot Indo-Pacific (second column of the pair)
                ax_indopac = fig.add_subplot(gs[row, gs_col_offset + 1])
                cf_indopac = ax_indopac.contourf(
                    indopac_member["lat"], depth_member, indopac_member,
                    levels=self.clevels, cmap="RdBu_r",
                    vmin=-self.vmax, vmax=self.vmax, extend='both'
                )
                last_cf = cf_indopac  # Keep reference for colorbar

                indopac_member = indopac_member.sel(lat=slice(self.SOlat, None))
                cs_indopac = ax_indopac.contour(indopac_member["lat"], depth_member, indopac_member, 
                                                levels=self.clevels, colors="grey", linewidths=0.5)
                ax_indopac.contour(indopac_member["lat"], depth_member, indopac_member, 
                                   levels=[0], colors="black", linewidths=1.0)
                ax_indopac.clabel(cs_indopac, inline=True, fontsize=8, fmt="%1.0f")
                ax_indopac.invert_yaxis()
                
                # Overlay Southern Ocean portion
                so_part = glo_member.sel(lat=slice(None, self.SOlat))
                if lat_zoom is None or lat_zoom.start < self.SOlat:
                    ax_indopac.contourf(
                        so_part["lat"], depth_member, so_part,
                        levels=self.clevels, cmap="RdBu_r",
                        vmin=-self.vmax, vmax=self.vmax
                    )
                    cs_so = ax_indopac.contour(so_part["lat"], depth_member, so_part, 
                                           levels=self.clevels, colors="grey", linewidths=0.5)
                    ax_indopac.contour(so_part["lat"], depth_member, so_part, 
                                   levels=[0], colors="black", linewidths=1.0)
                    ax_indopac.axvline(self.SOlat, color="black", linestyle="dotted")
                    ax_indopac.clabel(cs_so, inline=True, fontsize=8, fmt="%1.0f")
                
                
                
                if layers is None:
                    ax_indopac.set_ylim([6000, 0])
                else:
                    ylims = [depth_member[layers_slice.stop - 1], depth_member[layers_slice.start]]
                    ax_indopac.set_ylim(ylims)
                
                if lat_zoom is not None:
                    ax_indopac.set_xlim([lat_zoom.start, lat_zoom.stop])
                
                
                ax_indopac.set_yticklabels([])

                if row==nrows:
                    ax_indopac.set_xlabel("Latitude", fontsize=20)
                    ax_indopac.set_xticklabels(np.arange(-80,80,20), fontsize=14)
                else:
                    ax_indopac.set_xticklabels([])
                    
                
                # Store first row axes for title placement
                if row == 1:
                    axes_first_row[f'indopac_{col_offset}'] = ax_indopac
            
            # Add column titles only on the first row
            for pair_idx in range(n_members_per_row):
                gs_col = pair_idx * 2
                if gs_col >= 2:
                    gs_col += 1
                
                # Atlantic title
                if f'atl_{pair_idx * 2}' in axes_first_row:
                    ax = axes_first_row[f'atl_{pair_idx * 2}']
                    ax.text(0.5, 1.15, "Atlantic", transform=ax.transAxes,
                           fontsize=25, fontweight='bold', ha='center', va='bottom')
                
                # Indo-Pacific title
                if f'indopac_{pair_idx * 2}' in axes_first_row:
                    ax = axes_first_row[f'indopac_{pair_idx * 2}']
                    ax.text(0.5, 1.15, "Indo-Pacific", transform=ax.transAxes,
                           fontsize=25, fontweight='bold', ha='center', va='bottom')
            
            # Add colorbar at the bottom spanning all columns
            if last_cf is not None:
                cax = fig.add_subplot(gs[-1, :])
                cbar = fig.colorbar(last_cf, cax=cax, orientation="horizontal")
                cbar.set_label("Meridional Transport (Sv)", fontsize=20)
            
            figs.append(fig)
        
        plt.show()
        
        # Save figures if requested
        if savefig:
            os.makedirs(folder, exist_ok=True)
            for fig_idx, fig in enumerate(figs):
                timestamp = atlantic[0].time.dt.year.values if "time" in atlantic[0].dims else 0
                fname = f"moc_members_{run}_{fig_idx+1}_of_{len(figs)}_{timestamp}.png"
                fig.savefig(os.path.join(folder, fname), dpi=300, bbox_inches="tight")
                plt.close(fig)
        
        return figs


### draw_arrows
    def draw_arrows(self, axs, moc, basin="glo"):
        """
        Draw directional arrows along 34.5°S for a given MOC profile.
        Automatically uses extrema-flipping for Indo-Pacific or falls back to basic max/min logic.
        
        Parameters
        ----------
        axs : list of matplotlib axes
            Axes to plot arrows on.
        profile : xarray.DataArray
            MOC profile at 34.5°S.
        basin : str
            "glo" for Southern Ocean, "atl" for Atlantic, "ip" for Indo-Pacific.
        """
        profile = moc.sel(lat=self.SOlat)
        
        depth_vals = profile[self.depth_name].values
        values = profile.values
        lat_target = self.SOlat
        scale = 0.9

        # Find local extrema
        local_min_idx = argrelextrema(values, np.less, order=2)[0]
        local_max_idx = argrelextrema(values, np.greater, order=2)[0]
        extrema_idx = np.sort(np.concatenate([local_min_idx, local_max_idx]))

        # Decide mode: use extrema mode only if multiple extrema found
        use_extrema_mode = (len(local_min_idx) > 1) or (len(local_max_idx) > 1)

        if not use_extrema_mode:  # fallback to basic max/min logic
            depth_pos_max = depth_vals[np.nanargmax(values)] if np.any(values > 0) else None
            depth_neg_max = depth_vals[np.nanargmin(values)] if np.any(values < 0) else None

            for d, v in zip(depth_vals, values):
                if np.isnan(v):
                    continue
                dx_mag = scale * abs(v)

                if basin == "glo":  # anchored left
                    if v > 0 and (depth_pos_max is not None and d < depth_pos_max):
                        dx, start_lat, ac = dx_mag, lat_target - dx_mag, 'g'
                    elif v > 0:
                        dx, start_lat, ac = -dx_mag, lat_target, 'm'
                    elif v < 0 and (depth_neg_max is not None and d < depth_neg_max):
                        dx, start_lat, ac = -dx_mag, lat_target, 'm'
                    else:
                        dx, start_lat, ac = dx_mag, lat_target - dx_mag, 'g'
                
                else:  # Atlantic/IP anchored right
                    if v > 0 and (depth_pos_max is not None and d < depth_pos_max):
                        dx, start_lat, ac = dx_mag, lat_target, 'g'
                    elif v > 0:
                        dx, start_lat, ac = -dx_mag, lat_target + dx_mag, 'm'
                    elif v < 0 and (depth_neg_max is not None and d < depth_neg_max):
                        dx, start_lat, ac = -dx_mag, lat_target + dx_mag, 'm'
                    else:
                        dx, start_lat, ac = dx_mag, lat_target, 'g'

                for axx in axs:
                    axx.arrow(start_lat, d, dx, 0,
                              head_width=50, head_length=1.5,
                              fc=ac, ec=ac, length_includes_head=True)

                    axx.scatter(self.SOlat-2 if basin=='glo' else self.SOlat+2, depth_pos_max, c='lightcoral' if basin =='glo' else 'r', s=100, alpha=0.7)
                    axx.scatter(self.SOlat-2 if basin=='glo' else self.SOlat+2, depth_neg_max, c='cornflowerblue' if basin=='glo' else 'b', s=100, alpha=0.7)
                    
        else:  # extrema-flipping mode
            direction = -1  # start with left
            for i in range(len(depth_vals)):
                d = depth_vals[i]
                v = values[i]
                if np.isnan(v):
                    continue
                if i in extrema_idx:
                    direction *= -1
                dx_mag = scale * abs(v)
                if basin == "glo":
                    if direction == -1:  # left arrow
                        dx, start_lat, ac = dx_mag, lat_target - dx_mag, 'g'
                    else:  # right arrow
                        dx, start_lat, ac = -dx_mag, lat_target, 'm'
                else: # Atlantic/IP
                    if direction == -1:  # left arrow
                        dx, start_lat, ac = dx_mag, lat_target, 'g'
                    else:  # right arrow
                        dx, start_lat, ac = -dx_mag, lat_target + dx_mag, 'm'
                
                for axx in axs:
                    axx.arrow(start_lat, d, dx, 0,
                              head_width=50, head_length=1.5,
                              fc=ac, ec=ac, length_includes_head=True)
                    if isinstance(local_max_idx,np.ndarray):
                        xpos = np.array([self.SOlat-2 if basin=='glo' else self.SOlat+2]*len(local_max_idx))
                    else:
                        xpos = self.SOlat-2 if basin=='glo' else self.SOlat+2
                    if isinstance(local_min_idx, np.ndarray):
                        xneg = np.array([self.SOlat-2 if basin=='glo' else self.SOlat+2]*len(local_min_idx))
                    else:
                        xneg = self.SOlat-2 if basin=='glo' else self.SOlat+2
                    axx.scatter(xpos, depth_vals[local_max_idx], c='lightcoral' if basin=='glo' else 'r', s=100, alpha=.7, label='Maximum SO' if basin == 'glo' else 'Maximum Atl/IndoPac')
                    axx.scatter(xneg, depth_vals[local_min_idx], c='cornflowerblue' if basin == 'glo' else 'b', s=100, alpha=.7, label='Minimum SO' if basin == 'glo' else 'Minimum Atl/IndoPac')
                    #axx.legend(loc='lower right')

        return

### baker_upwelling
    def baker_upwelling(self, glo=None, atlantic=None, indopac=None, mean=True, plot=False, suptitle=None, multidec=False, for_regression=False, add=False):
        """
        Calculate upwelling pathways according to Baker et al. methodology.
        
        Parameters:
        -----------
        moc_global : xarray.DataArray, optional
            Global meridional overturning circulation data. Can be all monthly / yearly data or temporal mean with no time dimension.
        moc_atlantic : xarray.DataArray, optional
        a    Atlantic meridional overturning circulation data. Can be all monthly / yearly data or temporal mean with no time dimension.
        moc_indopac : xarray.DataArray, optional
            Indo-Pacific meridional overturning circulation data. Can be all monthly / yearly data or temporal mean with no time dimension.
            
        Returns:
        --------
        list or tuple
            Upwelling pathway components, either as a list of tuples (for multiple time steps)
            or a single tuple (for a single time step)
        """
        
        # Check if any MOC data is None and calculate mean if necessary
        if mean and any(x is None for x in [glo, atlantic, indopac]):
            glo, atlantic, indopac = self.calc_mean_moc()

        elif not mean and any(x is None for x in [glo, atlantic, indopac]):
            glo = self.moc_global
            atlantic = self.moc_atlantic 
            indopac = self.moc_indopac

        # Check if the provided MOC variables have a time dimension or length > 1
        time_dim_exists = 'time' in glo.dims
        time_length = glo.sizes.get('time', 1)
        
        if time_dim_exists:
            if time_length > 57 and not multidec:
                glo_Ymean = glo.resample(time='1Y').mean(dim='time')
                atlantic_Ymean = atlantic.resample(time='1Y').mean(dim='time')
                indopac_Ymean = indopac.resample(time='1Y').mean(dim='time')
            elif time_length == 57:
                glo_Ymean = glo
                atlantic_Ymean = atlantic
                indopac_Ymean = indopac
            elif multidec:
                glo_Ymean = glo.resample(time='1Y').mean(dim='time')
                atlantic_Ymean = atlantic.resample(time='1Y').mean(dim='time')
                indopac_Ymean = indopac.resample(time='1Y').mean(dim='time')

            if for_regression:
                time = glo_Ymean.time
            else:
                time = glo_Ymean.time.dt.year
            results = []
            for t in range(glo_Ymean.sizes.get('time', 1)):
                glo_mean_t = glo_Ymean.isel(time=t)
                atlantic_mean_t = atlantic_Ymean.isel(time=t)
                indopac_mean_t = indopac_Ymean.isel(time=t)
                # Run the rest of the logic for the current time step
                result = self._calculate_for_time_step(
                    glo_mean_t, atlantic_mean_t, indopac_mean_t
                )
                results.append(result)

            
            if not mean and plot:
                if suptitle == None:
                    suptitle = self.suptitle
                    
                mean_pathways = self._calculate_for_time_step(*self.calc_mean_moc(glo, atlantic, indopac))
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(1,1,1)
                ax.plot(time, [r[0] for r in results], c='purple', label=f'AMOC max')# (mean: {mean_pathways[0]:.2f})')
                ax.plot(time, [r[1] for r in results], c='deepskyblue', label=f'Atlantic Up')# (mean: {mean_pathways[1]:.2f})')
                if add:
                    ax.plot(time, [r[2]+r[3] for r in results], c='y', label=f'Southern Ocean + Indo-Pacific res Up')
                else:
                    ax.plot(time, [r[2] for r in results], c='darkorange', label=f'Southern Ocean Up')# (mean: {mean_pathways[2]:.2f})')
                    ax.plot(time, [r[3] for r in results], c='yellowgreen', label=f'Indo-Pacific residual Up')# (mean: {mean_pathways[3]:.2f})')

                
                ax.set_ylim([0, 38])
                ax.set_xlim([time[0], time[-1]])
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Pathway Strength (Sv)')
                if 'Historical' not in suptitle:
                    ax.yaxis.label.set_color('none')
        
                ax.legend(loc='upper left')
                ax.grid(which='major', linestyle='-', alpha=0.7)
                ax.grid(which='minor', linestyle=':', alpha=0.3)

                fig.suptitle(suptitle, fontsize=24, color=self.suptitle_color)
                plt.tight_layout()  # leave some space at the bottom for the text
                
        else:
            time = [0]
            results = []
            results.append(self._calculate_for_time_step(glo, atlantic, indopac))

        upwelling = xr.Dataset(
            data_vars={
                'amoc_max': ('time', [r[0] for r in results]),
                'atlantic_Up': ('time', [r[1] for r in results]),
                'SO_Up': ('time', [r[2] for r in results]),
                'indopac_resUp': ('time', [r[3] for r in results]),
            },
            coords={'time': time})


        if plot:
            return fig, ax, upwelling
        else:
            return upwelling

    
    def _calculate_for_time_step(self, glo, atlantic, indopac):
        """
        Helper function to calculate the AMOC upwelling pathways for a single time step.
        """

        # Get the points from the calculation
        points = self.calc_points(glo, atlantic, indopac)

        # Calculate depth for South Atlantic local bottom
        depth_south_atlantic_local_bottom = atlantic.sel(
            {self.depth_name:slice(points.depth_amoc_345,None)})[self.depth_name].where(  # slice depth of amoc at 34.5°S until bottom 
                atlantic.sel(
                    {'lat':self.SOlat, self.depth_name:slice(points.depth_amoc_345,None)}) <= points.amoc_min  # slice latitude at 34.5°S and depth of amoc at 34.5°S until bottom, find where it's values are smaller than AMOC min
        ).min(dim=self.depth_name).values # take minimum value of this -> highest / shallowest depth

        # Calculate components of the localised South Atlantic circulation
        south_atlantic_local_total = points.amoc_345 - points.amoc_min # definitively positive. waters that are downwelled inside the Atlantic
        south_atlantic_local_indopacup = min(south_atlantic_local_total,
            (indopac.sel({'lat':self.SOlat, self.depth_name: depth_south_atlantic_local_bottom}).clip(max=0) # indopac at 34.5°S and depth of SA loc bottom 
            - indopac.sel({'lat':self.SOlat, self.depth_name: points.depth_amoc_345}).clip(max=0) # pmoc_zAMOC345: indopac at 34.5°S and depth of Atl max at 34.5°S (AMOC_345) --> higher
            ).clip(min=0)) # only relevant if flow is northward i.e. > 0. otherwise waters would enter SO at a depth where they still need to be upwelled. we want to know if any of the SA loc waters are upwelled in the IndoPac.
        # Ensures south_atlantic_local_indopacup is zero if net flow is southward in Indo-Pacific over depth of localised South Atlantic circulation
        
        south_atlantic_local_windup = (south_atlantic_local_total - south_atlantic_local_indopacup).clip(min=0)  # unnecessary clip as indopacup can max be total

        ##
        # ‘South_Atlantic_local’, determined from equation (2), is reduced if a component of the localized South Atlantic waters enters an 
        # anticlockwise overturning cell in the Indo-Pacific Ocean, upwells and rejoins the northwards branch of the localized South Atlantic circulation. 
        # This is because these waters do not upwell in the SO and thus do not reduce SouthernOcean_Up.
        ##

        # Calculate Atlantic upwelling pathway
        atlantic_Up = (points.amoc_max - points.amoc_min).clip(min=0) # unnecessary clip bc amoc_min is smaller than amoc_max by definition.

        # Calculate PMOC strength at depth of the maximum AMOC strength at 34.5S
        if (indopac.sel({'lat':self.SOlat, self.depth_name: points.depth_amoc_345}) > 0).any(): # indopac at 34.5°S and depth of AMOC at 34.5°S
            pmoc_zAMOC345 = points.pmoc_zAMOC345
        else:
            pmoc_zAMOC345 = 0
        # could be replaced by a .clip(min=0)

        # Calculate Southern Ocean upwelling pathway, removing PMOC upwelling in Southern Ocean that cannot be connected to AMOC
        # SO_Up = (glo.sel(lat=self.SOlat).max() - pmoc_zAMOC345).clip(min=0) # pmoc_zAMOC345 is 0 in all of my cases
        SO_Up = (points.so_345 - pmoc_zAMOC345).clip(min=0) # pmoc_zAMOC345 is 0 in all of my cases
        
        ## HERE HERE HERE
        if south_atlantic_local_windup > 0: # as opposed to = 0
            self.south_atlantic_local_windup = south_atlantic_local_windup
            SO_Up = (SO_Up - south_atlantic_local_windup).clip(min=0)
        
        #if SO_Up<points.amoc_min:
        #    if hasattr(glo, 'time'):
        #        print(glo.time.dt.year.values)
        #if SO_Up>points.amoc_min:
        #    print(SO_Up.values, points.amoc_min.values)
        
        SO_Up = min(SO_Up, points.amoc_min) # if SO_Up > amoc_min, pathway is reduced to amoc_min

        # Calculate Indo-Pacific residual upwelling pathway
        indopac_resUp = points.amoc_max - atlantic_Up - SO_Up

        return points.amoc_max, atlantic_Up, SO_Up, indopac_resUp


    def hovmoeller(self, lats, basins=False, contour=False): # works only for one simulation and single lat (not slice) at a time right now
        
        if not isinstance(lats, list):
            lats = [lats]

        figs = []
        basin_titles = ['Atlantic', 'Indo-Pacific', 'Southern Ocean']
        for l, lat in enumerate(lats):

            if basins and lat <= -34.5:
                nrows = 3
            elif basins and lat > -34.5:
                nrows = 2
            else:
                nrows = 1

            figs.append(plt.figure(figsize=(10, 6 * nrows)))
            
            fig = figs[l]
            last_cf = None  # will keep a reference to the last contourf mappable

            # for j, m in enumerate(self.members):
            #mean_dims = ['lon']
            #if isinstance(lat, slice):
            #    mean_dims.append('lat')

            if basins:
                datas = [self.moc_atlantic, self.moc_indopac, self.moc_global]
                data = [d.sel(lat=lat, method='nearest') for d in datas]
                plot_title = basin_titles
            else:
                data = [self.moc_global.sel(lat=lat, method='nearest')]
                plot_title = ['Global']

            for i, (basin_data, basin_title) in enumerate(zip(data, plot_title)):  

                if nrows==2 and 'Southern' in basin_title:
                    continue

                ax = fig.add_subplot(nrows, 1, i +1)
    
                cf = ax.contourf(
                    basin_data.time,
                    basin_data[self.depth_name],
                    basin_data.transpose(self.depth_name, 'time'),
                    levels=self.clevels,
                    cmap='RdBu_r',
                    extend='both'
                )

                if contour:
                    cs = ax.contour(
                        basin_data.time,
                        basin_data[self.depth_name],
                        basin_data.transpose(self.depth_name, 'time'),
                        levels=self.clevels,
                        colors='grey',
                        alpha= 0.6,
                        linewidths=0.5
                    )
    
                    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f') # contour labels


                ax.invert_yaxis()
                ax.set_xlabel('Time')
                ax.set_ylabel('Depth (m)')
                ax.set_title(f'{lat}°')
                fig.suptitle(self.suptitle)
    
                last_cf = cf  # keep updating to have a valid mappable for the colorbar
            
                if basins:
                    self._set_basin_titles(fig, lat=lat)
            
                # Colorbar axis params: vertical, 0.8 of figure height
                cbar_width = 0.05 * 1
                cbar_height = 0.6
                cbar_left = 0.94
                cbar_bottom = (1.0 - cbar_height) / 2.0
            
                # Only add the colorbar if we captured a mappable
                if last_cf is not None:
                    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
                    cbar = fig.colorbar(last_cf, cax=cax, orientation='vertical')
                    cbar.set_label('Meridional Transport (Sv)')

        
        plt.show()
            
        return figs

    def _set_basin_titles(self,fig, lat=None):
        # Add vertical row titles (one per basin) on the left, centered vertically next to each row.
        subplot_axes = [ax for ax in fig.axes if hasattr(ax, 'get_subplotspec')]
    
        if lat is not None and lat > -34.5:
            bt = ['Atlantic', 'Indo-Pacific']
        else:
            bt = ['Atlantic', 'Indo-Pacific', 'Southern Ocean']
            
        for i, basin_title in enumerate(bt):
            # Collect axes that belong to row i by checking their SubplotSpec row span
            row_axes = []
            for ax in subplot_axes:
                ss = ax.get_subplotspec()
                # ss.rowspan is a slice: rows = range(start, stop)
                row_start = ss.rowspan.start
                row_stop = ss.rowspan.stop
                if i >= row_start and i < row_stop:
                    row_axes.append(ax)
    
            if not row_axes:
                continue
    
            # Compute vertical center from the positions of axes in this row
            y_centers = [ax.get_position().y0 + 0.5 * ax.get_position().height for ax in row_axes]
            y_center = float(np.mean(y_centers))
    
            # compute x position just left of the left-most axis in the row
            x_lefts = [ax.get_position().x0 for ax in row_axes]

            x_offset = 0.125
            x_pos = float(min(x_lefts) - x_offset)
            if x_pos < 0.01:
                x_pos = 0.01
    
            # place the vertical text; ha='center' so it's centered on x_pos, va='center' vertically
            fig.text(x_pos, y_center, basin_title,
                     va='center', ha='center', rotation='vertical',
                     fontsize=20, fontweight='bold')

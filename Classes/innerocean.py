import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from cartopy.util import add_cyclic_point
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

class MyDataHandler:
    def __init__(self, data): self.data = data
    def ensure_time_averaged(self):
        if 'time' in self.data.dims:
            self.data = self.data.mean(dim='time')
        return self.data
    def ensure_dimensions(self):
        if len(self.data.lat)!=180 or len(self.data.lon)!=360:
            lat = np.arange(-89.5,90.5,1, dtype=np.float32)
            lon = np.arange(1,361,1, dtype=np.float32)
            self.data = self.data.interp({'lat': lat, 'lon': lon}, method="linear")
        return self.data

class innerocean:
    """
    files      : list[str|xr.Dataset]
    varname    : str (e.g. 'thetao', 'so')
    typs       : list[str] ('ass' -> depth='depth', 'hist' -> depth='lev')
    suptitles  : list[str] (one per file)
    """
    def __init__(self, files, varname, typs, suptitles):
        if isinstance(files, list):
            assert len(files) == len(typs) == len(suptitles), "files, typs, suptitles must match in length"
            self.files, self.typs, self.suptitles = files, typs, suptitles
        else:
            self.files, self.typs, self.suptitles = [files], [typs], [suptitles]

        self.n = len(self.files)
        self.varname = varname
        fallback = None
        if self.varname == 'thetao':
            fallback = 'temperature'
        elif self.varname == 'so':
            fallback = 'salinity'
                    
        if varname == 'temperature':
            K = -273.15
        else:
            K = 0
    
        # load + time-average all datasets; extract variable and depth name

        titles_vert = []
        for name in self.suptitles:
            if name == 'observations':
                n_vert = 'EN4'
            n_vert = name.split('_')
            if 'historical' in n_vert:
                n_vert = n_vert[0][:4].upper()
            elif "atm" in n_vert or "oce" in n_vert:
                n_vert = n_vert[1][:3].upper()
            titles_vert.append(n_vert)


        
        self.members = []
        for f, t, title, title_short in zip(self.files, self.typs, self.suptitles, titles_vert):
            ds = xr.open_dataset(f) if isinstance(f, str) else f

            handler = MyDataHandler(ds)
            ds = handler.ensure_dimensions()
            ds_tm = handler.ensure_time_averaged()
            depth_name = 'depth' if t == 'ass' else 'lev'

            try:
                datavar = ds[self.varname] + K
                datavar_tm = ds_tm[self.varname] + K
            except KeyError:
                if fallback is None:
                    raise KeyError(f"Variable '{self.varname}' not found in dataset and no fallback defined.")
    
                try:
                    datavar = ds[fallback] + K
                    datavar_tm = ds_tm[fallback] + K
                except KeyError:
                    raise KeyError(
                        f"Neither '{self.varname}' nor fallback '{fallback}' were found in dataset {f}."
                    )

            self.members.append({
                'ds': ds,
                'ds_timemean' : ds_tm,
                'data': datavar,
                'data_timemean' : datavar_tm,
                'depth_name': depth_name,
                'title': title,
                'color': self._get_color(title),
                'title_short': title_short
            })

        # basin masks (assume same grid as data; if not, regrid beforehand)
        grid = xr.open_dataset('/work/uo1075/u241372/GR15L40_fx_rbek_mean_remap.nc').mean(dim='time')
        
        med_baltic_mask = (((grid.lon>=354) & (grid.lat>=30) & (grid.lat<=40)) |
                          ((grid.lon>=0) & (grid.lon<=10) & (grid.lat>=30) & (grid.lat<=48)) |
                          ((grid.lon>=10) & (grid.lon<=50) & (grid.lat>=20) & (grid.lat<=70)))
        
        atl_mask = (grid.rbek.isin([4,5]) & (grid.lat >= -34.5)) & (~med_baltic_mask)
        self.atl_mask = atl_mask.astype(bool)



        indopac_mask = (grid.rbek.isin([7,8]) & (grid.lat >= -34.5)) 
        self.indopac_mask = indopac_mask.astype(bool)

        #self.so_mask = grid.rbek.where(grid.rbek == 6, other=0).astype(bool)
        so_mask = (grid.rbek.isin([5,6,7]) & (grid.lat <= -34.5))
        self.so_mask = so_mask.astype(bool)

        # titles per basin
        self.basin_titles = [
            'Atlantic',
            'Indo-Pacific',
            'Southern Ocean'
        ]

        # colour levels
        if self.varname == 'thetao' or self.varname == 'temperature':
            self.label = 'Potential temperature (°C)'
            self.clevels = np.arange(0, 15, 0.5)

            colors = [
                (0.1, 0.2, 0.4),    # Dunkelviolett
                (0.8, 0.1, 0.4),   # Violett
                (1.0, 0.6, 0.2),   # Orange
                (1.0, 0.8, 0.3),
                (1.0, 1.0, 200/256)   # Gelb   
            ]
            
            # Erstelle Colormap
            cmap_thetao = LinearSegmentedColormap.from_list("yellow_to_darkpurple", colors, N=256)
            self.cmap = cmap_thetao
            
        elif self.varname == 'so' or self.varname == 'salinity':
            self.label = "Salinity (PSU)"
            self.clevels = np.arange(34.3, 36, 0.1)
            
            colors = np.array([
                (68, 1, 84),
                (59, 82, 139),    # Dunkelviolett
                (33, 145, 140),   # Violett
                (94, 201, 98),   # Orange
                (253, 231, 37),
                (256, 256, 200)   # Gelb   
            ]) /256
            
            # Erstelle Colormap
            cmap_so = LinearSegmentedColormap.from_list("viridis_bright", colors, N=256)
            self.cmap = cmap_so
            
        elif self.varname == 'rhopoto':
            self.label = "Potential Density ()"
            self.clevels = np.arange(1030,1045,.5)
            self.cmap = 'PRGn'

    def _get_color(self, suptitle):
        if 'Historical' in suptitle:
            return 'crimson'
        elif 'Atmosphere' in suptitle:
            return 'forestgreen'
        elif 'Ocean' in suptitle:
            return 'mediumblue'
        elif 'Observation' in suptitle:
            return 'darkorange'
        else:
            return 'darkorange'

    def _mask_basins(self, da):
        return (
            da.where(self.atl_mask),
            da.where(self.indopac_mask),
            da.where(self.so_mask)
        )

    def plot_basins(self):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
        ax.add_feature(cfeature.COASTLINE, edgecolor='lightgrey', linewidth=0.6)
        ax.add_feature(cfeature.LAND, color='whitesmoke')
        atl_cyc, lon_cyc = add_cyclic_point(self.atl_mask.values, coord=self.atl_mask.lon.values)
        indopac_cyc, lon_cyc = add_cyclic_point(self.indopac_mask.values, coord=self.indopac_mask.lon.values)
        so_cyc, lon_cyc = add_cyclic_point(self.so_mask.values, coord=self.so_mask.lon.values)

        ax.contourf(
            lon_cyc, self.atl_mask.lat, atl_cyc,
            levels=[0.5, 1.5], cmap='Blues', alpha=0.8, transform=ccrs.PlateCarree()) 
        ax.contourf(
            lon_cyc, self.indopac_mask.lat, indopac_cyc,
            levels=[0.5, 1.5], cmap='Greens', alpha=0.8, transform=ccrs.PlateCarree())
        ax.contourf(
            lon_cyc, self.so_mask.lat, so_cyc,
            levels=[0.5, 1.5], cmap='Oranges', alpha=0.8, transform=ccrs.PlateCarree())
        
        ax.set_global()
        # ax.set_title('Ocean basin masks')

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(range(-90, 91, 30))
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # Add legend with colored boxes
    
        legend_elements = [
            Patch(facecolor='steelblue', alpha=0.8, label='Atlantic Ocean'),
            Patch(facecolor='green', alpha=0.8, label='Indo-Pacific Ocean'),
            Patch(facecolor='darkorange', alpha=0.8, label='Southern Ocean')
        ]
        ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9)
    
        return fig
        

    # -------- surface (depth index 0) --------
    def plot_depth0(self):
        """
        Plot surface (depth index 0) for each member in self.members in a 1 x n row of maps.
        Produce a single vertical colorbar on the right side of the figure, with height 0.8
        (80% of the figure height). The subplots area is tightened to leave space for that colorbar.
        """
        # Figure: leave room on the right for the colorbar (we'll adjust with tight_layout rect)
        fig = plt.figure(figsize=(12 * self.n, 6))
        proj = ccrs.PlateCarree()
    
        last_cf = None  # will store the last contourf/QuadMesh for the colorbar
    
        for j, m in enumerate(self.members):
            ax = plt.subplot(1, self.n, j + 1, projection=proj)
            ax.add_feature(cfeature.COASTLINE, edgecolor='lightgrey')
            ax.set_xlim([-180, 180])
    
            datavar_atl, datavar_indopac, datavar_so = self._mask_basins(m['data_timemean'])
            depth0 = {m['depth_name']: 0}
    
            # overlay 3 basins on the same axes
            for basin_data in (datavar_atl, datavar_indopac, datavar_so):
                # plot and keep reference to the returned mappable (ContourSet or QuadMesh)
                cf = ax.contourf(
                    basin_data.lon, basin_data.lat,
                    basin_data.isel(depth0),
                    levels=self.clevels, cmap=self.cmap, extend='both'
                )
                # keep the last mappable so we can use it for the single colorbar
                last_cf = cf
    
                # try drawing a mask boundary; continue if it errors
                try:
                    ax.contour(
                        basin_data.lon, basin_data.lat,
                        ~np.isnan(basin_data.isel(depth0)),
                        levels=[0.5], colors='black', linewidths=0.8
                    )
                except Exception:
                    # ignore boundary drawing errors here; plotting should continue
                    pass
    
            ax.set_title(m['title'], color=m['color'], fontweight='bold')
    
        # Reserve space on the right for a single colorbar:
        # Use tight_layout with rect to leave the right 8% of the figure free (rect's right=0.92).
        # This keeps subplots from overlapping the colorbar region.
        fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    
        # Create a new axes on the right for the colorbar:
        cbar_width = 0.02 * 1/self.n
        cbar_height = 0.8
        cbar_left = 0.94  # start near the right edge (0..1)
        cbar_bottom = (1.0 - cbar_height) / 2.0
    
        # Only add the colorbar if we captured a mappable
        if last_cf is not None:
            cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            # Use the figure-level colorbar so it spans the full height of cax
            cbar = fig.colorbar(last_cf, cax=cax, orientation='vertical')
            cbar.set_label(self.varname)
    
        return fig



    # -------- zonal mean (lat vs depth), 3 rows × N columns --------
    def plot_zonalmean(self, clevels=None, cmap=None):
        """
        Plot zonal-mean (lat x depth) panels arranged in 3 x n subplots and add a single
        vertical colorbar on the right that is 0.8 of the figure height. Column titles
        (member titles) remain above each column; basin titles are shown once per row as
        vertical row labels on the left (correctly centered and not overlapping axis labels).
        """
        if clevels is None:
            clevels = self.clevels

        if cmap is None:
            cmap = self.cmap
            
        fig = plt.figure(figsize=(10 * self.n, 12))
    
        last_cf = None  # will keep a reference to the last contourf mappable
    
        # Create subplots in the same order as before
        for j, m in enumerate(self.members):
            datavar_atl, datavar_indopac, datavar_so = self._mask_basins(m['data_timemean'])
            for i, (basin_data, basin_title) in enumerate(
                zip([datavar_atl, datavar_indopac], self.basin_titles[:2])
            ):
                ax = plt.subplot(2, self.n, i * self.n + j + 1)
    
                cf = ax.contourf(
                    basin_data.lat,
                    basin_data[m['depth_name']],
                    basin_data.mean(dim='lon'),
                    levels=clevels,
                    cmap=cmap,
                    extend='both'
                )
                ax.contourf(
                    datavar_so.lat,
                    datavar_so[m['depth_name']],
                    datavar_so.mean(dim='lon'),
                    levels=clevels,
                    cmap=cmap,
                    extend='both'
                )

                if self.varname == 'xx': #'thetao':
                    ax.contour(
                        basin_data.lat,
                        basin_data[m['depth_name']],
                        basin_data.mean(dim='lon'),
                        levels=[4],
                        colors='white',
                        linewidths=2.0)
                    ax.contour(
                        datavar_so.lat,
                        datavar_so[m['depth_name']],
                        datavar_so.mean(dim='lon'),
                        levels=[4],
                        colors='white',
                        linewidths=2.0)

                ax.axvline(-34.5, color='black', linestyle='dotted')
                
                ax.invert_yaxis()
                self._set_labels(ax,m,j,i, y='Depth (m)', x='Latitude')
                
                last_cf = cf  # keep updating to have a valid mappable for the colorbar
    
        # Reserve space on the right for a single colorbar and add margins for row labels/xlabels:
        fig.tight_layout(rect=[0.08, 0.06, 0.92, 0.95])
    
        self._set_basin_titles(fig)
    
        # Colorbar axis params: vertical, 0.8 of figure height
        cbar_width = 0.05 * 1/self.n
        cbar_height = 0.6
        cbar_left = 0.94
        cbar_bottom = (1.0 - cbar_height) / 2.0
    
        # Only add the colorbar if we captured a mappable
        if last_cf is not None:
            cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            cbar = fig.colorbar(last_cf, cax=cax, orientation='vertical')
            cbar.set_label(self.label)
        
        return fig


    # -------- zonal mean (lat vs depth), 3 rows × N columns --------
    def plot_zonalmean_v2(self, clevels=None, cmap=None):
        """
        Plot zonal-mean (lat x depth) panels arranged in 3 x n subplots and add a single
        vertical colorbar on the right that is 0.8 of the figure height. Column titles
        (member titles) remain above each column; basin titles are shown once per row as
        vertical row labels on the left (correctly centered and not overlapping axis labels).
        """
        if clevels is None:
            clevels = self.clevels

        if cmap is None:
            cmap = self.cmap


        # Make figure size
        fig_w = 20
        fig_h = 5 * self.n + (1.5) # if add_cbar else 0)
        fig = plt.figure(figsize=(fig_w, fig_h))

        # GridSpec: rows = n_sim (plots) + (1 if add_cbar else 0) for colorbar
        gs_rows = self.n + 1+ 1 # if add_cbar else 0)
        # Use a small height ratio for the colorbar row (last row)
        height_ratios = [5] * self.n + [0.075] +[.75]
        gs = GridSpec(gs_rows, 2, figure=fig, height_ratios=height_ratios, hspace=0.2, wspace=0.12) 

        last_cf = None  # will keep a reference to the last contourf mappable
        axes = np.array([[None for _ in range(2)] for _ in range(self.n)])
    
        # Create subplots in the same order as before
        for j, m in enumerate(self.members):
            datavar_atl, datavar_indopac, datavar_so = self._mask_basins(m['data_timemean'])
            for i, (basin_data, basin_title) in enumerate(
                zip([datavar_atl, datavar_indopac], self.basin_titles[:2])
            ):

                ax = fig.add_subplot(gs[j, i])
                axes[j][i] = ax
    
                cf = ax.contourf(
                    basin_data.lat,
                    basin_data[m['depth_name']],
                    basin_data.mean(dim='lon'),
                    levels=clevels,
                    cmap=cmap,
                    extend='both'
                )
                ax.contourf(
                    datavar_so.lat,
                    datavar_so[m['depth_name']],
                    datavar_so.mean(dim='lon'),
                    levels=clevels,
                    cmap=cmap,
                    extend='both'
                )

                if self.varname == 'xx': #'thetao':
                    ax.contour(
                        basin_data.lat,
                        basin_data[m['depth_name']],
                        basin_data.mean(dim='lon'),
                        levels=[4],
                        colors='white',
                        linewidths=2.0)
                    ax.contour(
                        datavar_so.lat,
                        datavar_so[m['depth_name']],
                        datavar_so.mean(dim='lon'),
                        levels=[4],
                        colors='white',
                        linewidths=2.0)

                ax.axvline(-34.5, color='black', linestyle='dotted')
                
                ax.invert_yaxis()
                # self._set_labels(ax,m,j,i, y='Depth (m)', x='Latitude')

                if j==3:
                    ax.set_xlabel('Latitude')                    
                if i==0: 
                    ax.set_ylabel('Depth (m)')
                
                last_cf = cf  # keep updating to have a valid mappable for the colorbar

            
    
        # Reserve space on the right for a single colorbar and add margins for row labels/xlabels:
        # fig.tight_layout(rect=[0.08, 0.06, 0.92, 0.95])
    
        # self._set_basin_titles(fig)

        for c, ax in enumerate(axes[0,:]):
            ax.set_title(self.basin_titles[c], pad=20)

        suptitle = ['HIST', 'ATM', 'OCE', 'EN4']
        color = [m['color'] for m in self.members]
        for r, ax in enumerate(axes[:,0]):
            ax.text(
                -0.17, 0.5, suptitle[r],
                color=color[r],
                transform=ax.transAxes,
                rotation=90,
                va='center', ha='center',
                fontsize=28, fontweight='bold'
            )
    
        # Only add the colorbar if we captured a mappable
        if last_cf is not None:
            cax = fig.add_subplot(gs[-1, :])
            cbar = fig.colorbar(last_cf, cax=cax, orientation='horizontal', pad=.3)
            cbar.set_label(self.label)
        
        return fig

    
    # -------- zonal mean (lat vs depth), 3 rows × N columns --------
    def plot_zonalmean_diff(self, clevels=None):
        """
        Plot zonal-mean (lat x depth) panels arranged in 3 x n subplots and add a single
        vertical colorbar on the right that is 0.8 of the figure height. Column titles
        (member titles) remain above each column; basin titles are shown once per row as
        vertical row labels on the left (correctly centered and not overlapping axis labels).
        """
        if clevels is None:
            clevels = self.clevels

        n = self.n -1
        fig = plt.figure(figsize=(10 * n, 12))
    
        last_cf = None  # will keep a reference to the last contourf mappable


        if self.varname == 'so':
            clevs = np.linspace(-2,2,141)
        elif self.varname == 'thetao':
            clevs = np.linspace(-4,4,117)
            
        
        # Create subplots in the same order as before
        for j, m in enumerate(self.members):
            
            if 'observations' in m['title'] or 'EN4' in m['title']:
                #depth = self.members[0]['data_timemean']['lev']
                #zonalmean_obsv = self.members[-1]['data_timemean'].interp(depth=depth).drop_vars('depth')
                #depth_name = 'lev'
                #cmap = self.cmap
                continue
            else:
                depth = m['data_timemean'][m['depth_name']]
                zonalmean_obsv = self.members[-1]['data_timemean'].interp(depth=depth).drop_vars('depth')
                depth_name = m['depth_name']
            obsv_atl, obsv_indopac, obsv_so = self._mask_basins(zonalmean_obsv)
            obsv_atl, obsv_indopac, obsv_so = obsv_atl.mean(dim='lon'), obsv_indopac.mean(dim='lon'), obsv_so.mean(dim='lon')

            if 'observations' in m['title'] or 'EN4' in m['title']:
                continue
                #atl_diff, indopac_diff, so_diff = obsv_atl, obsv_indopac, obsv_so
            else:
                datavar_atl, datavar_indopac, datavar_so = self._mask_basins(m['data_timemean'])
                zonalmean_atl, zonalmean_indopac, zonalmean_so = datavar_atl.mean(dim='lon'), datavar_indopac.mean(dim='lon'), datavar_so.mean(dim='lon')
                atl_diff, indopac_diff, so_diff = zonalmean_atl-obsv_atl, zonalmean_indopac-obsv_indopac, zonalmean_so-obsv_so


            for i, (basin_diff, basin_title) in enumerate(
                zip([atl_diff, indopac_diff], self.basin_titles[:2])
            ):
                ax = plt.subplot(2, n, i * n + j + 1)
    
                cf = ax.contourf(
                    basin_diff.lat,
                    basin_diff[depth_name],
                    basin_diff,
                    levels=clevs,
                    cmap='seismic',
                    extend='both'
                )
                ax.contourf(
                    so_diff.lat,
                    so_diff[depth_name],
                    so_diff,
                    levels=clevs,
                    cmap='seismic',
                    extend='both'
                )

                if self.varname == 'xx': #'thetao':
                    ax.contour(
                        basin_diff.lat,
                        basin_diff[depth_name],
                        basin_diff.mean(dim='lon'),
                        levels=[4],
                        colors='white',
                        linewidths=2.0)
                    ax.contour(
                        so_diff.lat,
                        so_diff[depth_name],
                        so_diff.mean(dim='lon'),
                        levels=[4],
                        colors='white',
                        linewidths=2.0)

                ax.axvline(-34.5, color='black', linestyle='dotted')
                
                ax.invert_yaxis()
                self._set_labels(ax,m,j,i, y='Depth (m)', x='Latitude')
                
                last_cf = cf  # keep updating to have a valid mappable for the colorbar
    
        # Reserve space on the right for a single colorbar and add margins for row labels/xlabels:
        fig.tight_layout(rect=[0.08, 0.06, 0.92, 0.95])
    
        self._set_basin_titles(fig)
    
        # Colorbar axis params: vertical, 0.8 of figure height
        cbar_width = 0.05 * 1/n
        cbar_height = 0.6
        cbar_left = 0.94
        cbar_bottom = (1.0 - cbar_height) / 2.0
    
        # Only add the colorbar if we captured a mappable
        if last_cf is not None:
            cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            cbar = fig.colorbar(last_cf, cax=cax, orientation='vertical')
            cbar.set_label(self.label)
    
        return fig

    def plot_zonalmean_combined(self, clevels=None, diff_clevs=None, cmap=None, per_basin=True):
        """
        Single combined zonal-mean plotting function.
    
        Layout:
          - self.n columns (one per member)
          - 4 rows:
            row 1: Atlantic data for each member (and Southern Ocean overlaid)
            row 2: Atlantic difference to observational reference (members[:-1] show diff,
                   the reference column is left empty)
            row 3: Indo-Pacific data for each member (and Southern Ocean overlaid)
            row 4: Indo-Pacific difference to observational reference (members[:-1] show diff,
                   the reference column is left empty)
    
        Notes:
          - Observational/reference member is assumed to be self.members[-1].
          - Southern Ocean data (so) are always plotted/overlaid, either as data or as difference.
          - A single vertical colorbar is placed on the right (uses the last valid mappable).
        """
        if clevels is None:
            clevels = self.clevels
        if cmap is None:
            cmap = self.cmap
    
        ncols = self.n
        nrows = 4
        fig = plt.figure(figsize=(10 * max(1, ncols), 24))
    
        # Determine observational/reference member (assumed last)
        obs_member = self.members[-1]
        obs_da = obs_member['data_timemean']
    
        # Keep separate mappables for data and difference colorbars
        last_data_cf = None   # for main data colormap (right colorbar)
        last_diff_cf = None   # for seismic difference colormap (left colorbar)
    
        # Default diff levels if not provided (falls back to var-specific rules)
        if diff_clevs is None:
            if getattr(self, 'varname', '') == 'so':
                diff_clevs = np.linspace(-1.5, 1.5, 141)
            elif getattr(self, 'varname', '') == 'thetao':
                diff_clevs = np.linspace(-4, 4, 117)
            else:
                diff_clevs = None  # will let contourf auto-handle if needed
    
        for j, m in enumerate(self.members):
            # per-member data (may be observational or model)
            datavar_atl, datavar_indopac, datavar_so = self._mask_basins(m['data_timemean'])
    
            # mean over longitude for the main fields (we plot lat vs depth)
            atl_mean = datavar_atl.mean(dim='lon')
            indopac_mean = datavar_indopac.mean(dim='lon')
            so_mean = datavar_so.mean(dim='lon')
    
            # prepare interpolated observational reference aligned to current member depths
            # keep same depth coordinate name as member (e.g., 'lev' or 'depth')
            depth_name = m['depth_name']
            obs_interp = obs_da.interp(depth=m['data_timemean'][depth_name]).drop_vars('depth')
    
            obs_atl = self._mask_basins(obs_interp)[0].mean(dim='lon')
            obs_indopac = self._mask_basins(obs_interp)[1].mean(dim='lon')
            obs_so = self._mask_basins(obs_interp)[2].mean(dim='lon')

            if per_basin:
                ax2_idx = ncols + 1 + j
                ax3_idx = 2 * ncols + 1 + j
            else:
                ax2_idx = 2 * ncols + 1 + j
                ax3_idx = ncols + 1 + j
                
            # Row 1: Atlantic data
            ax1 = plt.subplot(nrows, ncols, 1 + j)
            cf1 = ax1.contourf(
                atl_mean.lat,
                atl_mean[depth_name],
                atl_mean,
                levels=clevels,
                cmap=cmap,
                extend='both'
            )
            # always overlay SO on same axis (data)
            ax1.contourf(
                so_mean.lat,
                so_mean[depth_name],
                so_mean,
                levels=clevels,
                cmap=cmap,
                extend='both'
            )
            ax1.axvline(-34.5, color='black', linestyle='dotted')
            ax1.invert_yaxis()
            # set labels and title for top row
            if hasattr(self, '_set_labels'):
                self._set_labels(ax1, m, j, 0, y='Depth (m)', x='Latitude')
            # ax1.set_title(m.get('title', '') if isinstance(m, dict) else '')
            last_data_cf = cf1
    
            # Row 2: Atlantic difference to observations
            ax2 = plt.subplot(nrows, ncols, ax2_idx)
            if m is obs_member:
                # leave reference column empty for differences
                ax2.axis('off')
            else:
                atl_diff = atl_mean - obs_atl
                so_diff = so_mean - obs_so
                cf2 = ax2.contourf(
                    atl_diff.lat,
                    atl_diff[depth_name],
                    atl_diff,
                    levels=diff_clevs,
                    cmap='seismic',
                    extend='both'
                )
                # overlay SO difference
                ax2.contourf(
                    so_diff.lat,
                    so_diff[depth_name],
                    so_diff,
                    levels=diff_clevs,
                    cmap='seismic',
                    extend='both'
                )
                ax2.axvline(-34.5, color='black', linestyle='dotted')
                ax2.invert_yaxis()
                if hasattr(self, '_set_labels'):
                    self._set_labels(ax2, m, j, 1, y='Depth (m)', x='Latitude')
                last_diff_cf = cf2
    
            # Row 3: Indo-Pacific data
            ax3 = plt.subplot(nrows, ncols, ax3_idx)
            cf3 = ax3.contourf(
                indopac_mean.lat,
                indopac_mean[depth_name],
                indopac_mean,
                levels=clevels,
                cmap=cmap,
                extend='both'
            )
            # overlay SO data also on this row
            ax3.contourf(
                so_mean.lat,
                so_mean[depth_name],
                so_mean,
                levels=clevels,
                cmap=cmap,
                extend='both'
            )
            ax3.axvline(-34.5, color='black', linestyle='dotted')
            ax3.invert_yaxis()
            if hasattr(self, '_set_labels'):
                self._set_labels(ax3, m, j, 2, y='Depth (m)', x='Latitude')
            last_data_cf = cf3
    
            # Row 4: Indo-Pacific difference to observations
            ax4 = plt.subplot(nrows, ncols, 3 * ncols + 1 + j)
            if m is obs_member:
                ax4.axis('off')
            else:
                indopac_diff = indopac_mean - obs_indopac
                so_diff = so_mean - obs_so
                cf4 = ax4.contourf(
                    indopac_diff.lat,
                    indopac_diff[depth_name],
                    indopac_diff,
                    levels=diff_clevs,
                    cmap='seismic',
                    extend='both'
                )
                # overlay SO difference
                ax4.contourf(
                    so_diff.lat,
                    so_diff[depth_name],
                    so_diff,
                    levels=diff_clevs,
                    cmap='seismic',
                    extend='both'
                )
                ax4.axvline(-34.5, color='black', linestyle='dotted')
                ax4.invert_yaxis()
                if hasattr(self, '_set_labels'):
                    self._set_labels(ax4, m, j, 3, y='Depth (m)', x='Latitude')
                last_diff_cf = cf4
    
        # Add basin row labels on the left (one per row), centered vertically for each row.
        # The vertical positions are approximations that work with the chosen figure size/tight_layout,
        # adjust if necessary.
        fig.tight_layout(rect=[0.08, 0.06, 0.92, 0.95])
        self._set_basin_titles(fig, diff=True, per_basin=per_basin)
            
        # Row center y positions (approx): top to bottom
        #row_centers = [0.87, 0.66, 0.45, 0.24]
        #row_labels = ['Atlantic (data)', 'Atlantic (model - obs)', 'Indo-Pacific (data)', 'Indo-Pacific (model - obs)']
        #for yc, label in zip(row_centers, row_labels):
        #    fig.text(0.02, yc, label, va='center', rotation='vertical', fontsize=12)
    
        # Left colorbar for difference (seismic) plots
        if last_diff_cf is not None:
            left_cbar_width = 0.02
            left_cbar_height = 0.6
            left_cbar_left = 0.99
            left_cbar_bottom = (1.0 - left_cbar_height) / 2.0
            cax_left = fig.add_axes([left_cbar_left, left_cbar_bottom, left_cbar_width, left_cbar_height])
            cbar_left = fig.colorbar(last_diff_cf, cax=cax_left, orientation='vertical')
            cbar_left.set_label(f"{getattr(self, 'label', '')} (difference)")
    
        # Single colorbar on the right using the last captured mappable for data
        if last_data_cf is not None:
            cbar_width = 0.02
            cbar_height = 0.6
            cbar_left = 0.93
            cbar_bottom = (1.0 - cbar_height) / 2.0
            cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            cbar = fig.colorbar(last_data_cf, cax=cax, orientation='vertical')
            cbar.set_label(getattr(self, 'label', ''))
    
        return fig

    
    def hovmoeller(self, lats, basins=False):
        
        if not isinstance(lats, list):
            lats = [lats]


        figs = []

        
        for l, lat in enumerate(lats):

            if basins and lat <= -34.5:
                nrows = 3
            elif basins and lat > -34.5:
                nrows = 2
            else:
                nrows = 1

            figs.append(plt.figure(figsize=(10 * self.n, 6 * nrows)))
            
            fig = figs[l]
            last_cf = None  # will keep a reference to the last contourf mappable

            axes = np.array([])
            for j, m in enumerate(self.members):
                mean_dims = ['lon']
                if isinstance(lat, slice):
                    mean_dims.append('lat')

                if basins:
                    datas = self._mask_basins(m['data'])
                    data = [d.sel(lat=lat, method='nearest').mean(dim=mean_dims) for d in datas]
                    plot_title = self.basin_titles
                else:
                    data = [m['data'].sel(lat=lat, method='nearest').mean(dim=mean_dims)]
                    plot_title = ['Global']

                for i, (basin_data, basin_title) in enumerate(zip(data, plot_title)):  

                    if nrows==2 and 'Southern' in basin_title:
                        continue

                    ax = fig.add_subplot(nrows, self.n, i * self.n + j + 1)
        
                    cf = ax.contourf(
                        basin_data.time,
                        basin_data[m['depth_name']],
                        basin_data.transpose(m['depth_name'], 'time'),
                        levels=self.clevels,
                        cmap=self.cmap,
                        extend='both'
                    )

                    ax.invert_yaxis()
                    self._set_labels(ax,m,j,i, y='Depth (m)', x='Time', tit=f'{lat}°')

                                        
                    last_cf = cf  # keep updating to have a valid mappable for the colorbar
                
                     # Reserve space on the right for a single colorbar and add margins for row labels/xlabels:
                #fig.tight_layout(rect=[0.08, 0.06, 0.92, 0.95])
            
                if basins:
                    self._set_basin_titles(fig)
            
            
                # Only add the colorbar if we captured a mappable
                if last_cf is not None:
                    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
                    cbar = fig.colorbar(last_cf, cax=cax, orientation='vertical')
                    cbar.set_label(self.label)

        plt.show()
            
        return figs


    def hovmoeller_v2(self, lats, basins=False):
        
        if not isinstance(lats, list):
            lats = [lats]
    
        figs = []
        
        for l, lat in enumerate(lats):
    
            if basins and lat <= -34.5:
                ncols = 3
            elif basins and lat > -34.5:
                ncols = 2
            else:
                ncols = 1
    
            # Now ncols is based on basins, and nrows is based on members
            nrows = self.n
    
            figs.append(plt.figure(figsize=(7 * ncols, 3.5 * nrows + 1.5)))
            
            fig = figs[l]
            
            # GridSpec: rows = n_sim (plots) + (1 if add_cbar else 0) for colorbar
            gs_rows = self.n + 1+ 1 # if add_cbar else 0)
            # Use a small height ratio for the colorbar row (last row)
            height_ratios = [7] * self.n + [0.2] +[1.05]
            gs = GridSpec(gs_rows, ncols, figure=fig, height_ratios=height_ratios, hspace=0.2, wspace=0.12) 

            axes = np.array([[None for _ in range(ncols)] for _ in range(nrows)])
            last_cf = None  # will keep a reference to the last contourf mappable
    
            for j, m in enumerate(self.members):
                mean_dims = ['lon']
                if isinstance(lat, slice):
                    mean_dims.append('lat')
    
                if basins:
                    datas = self._mask_basins(m['data'])
                    data = [d.sel(lat=lat, method='nearest').mean(dim=mean_dims) for d in datas]
                    plot_title = self.basin_titles
                
                else:
                    data = [m['data'].sel(lat=lat, method='nearest').mean(dim=mean_dims)]
                    plot_title = ['Global']
    
                for i, (basin_data, basin_title) in enumerate(zip(data, plot_title)):  
    
                    if ncols==2 and 'Southern' in basin_title:
                        continue
    
                    # Swapped: now row is j (member index), column is i (basin index)
                    ax = fig.add_subplot(gs[j, i])
                    axes[j,i] = ax

                    time = np.array(basin_data.time)
                    cf = ax.contourf(
                        time,
                        basin_data[m['depth_name']],
                        basin_data.transpose(m['depth_name'], 'time'),
                        levels=self.clevels,
                        cmap=self.cmap,
                        extend='both'
                    )
    
                    ax.invert_yaxis()
                    # Updated label: member name on y-axis, basin/location on x-axis title
                    # self._set_labels(ax, m, j, i, y='Depth (m)', x='Time', tit=f'{basin_title} at {lat}°')

                    # Add ticks to all subplots
                    #ax.set_xticks(np.arange(1960, 2010, 20))
                    ax.set_yticks(np.arange(1000, 6000, 1000))
                    
                    # Add x-axis labels to last row only
                    if j == nrows - 1:  # last row
                        # ax.set_xticklabels(np.arange(1960, 2010, 20))
                        ax.set_xlabel('Year')
                    else:
                        ax.set_xticklabels([])
                    
                    # Add y-axis labels to first column only
                    if i == 0:  # first column
                        ax.set_yticklabels(np.arange(1000, 6000, 1000))
                        ax.set_ylabel('Depth (m)')
                    else:
                        ax.set_yticklabels([])

        
                    last_cf = cf  # keep updating to have a valid mappable for the colorbar
                
            # Add member labels on the left side
            if basins:
                for col, ax in enumerate(axes[0,:]):
                    ax.set_title(self.basin_titles[col], pad=20)
                # self._set_member_labels(fig, nrows)
            
            # Add basin titles on the top
            # if basins:
            #    self._set_basin_column_titles(fig, ncols, self.basin_titles)
            
            suptitle = ['HIST', 'ATM', 'OCE', 'EN4']
            color = [m['color'] for m in self.members]
            for r, ax in enumerate(axes[:,0]):
                ax.text(
                    -.3, 0.5, suptitle[r],
                    color=color[r],
                    transform=ax.transAxes,
                    rotation=90,
                    va='center', ha='center',
                    fontsize=28, fontweight='bold'
                    )
            
            # Only add the colorbar if we captured a mappable
            if last_cf is not None:
                cax = fig.add_subplot(gs[-1,:])
                cbar = fig.colorbar(last_cf, cax=cax, orientation='horizontal')
                cbar.set_label(self.label)

            lat_title = str(lat)+ '°N' if lat>0 else str(np.abs(lat))+'°S'
            fig.suptitle(lat_title)
        
        plt.show()
            
        return figs
    
    
    def _set_member_labels(self, fig, nrows):
        """Add member labels on the left side of the plot."""
        for j, m in enumerate(self.members):
            if j < nrows:
                # Get the first subplot in each row to position the label
                ax = fig.axes[j * len(self.basin_titles)]  # First column of each row
                ax.text(-0.5, 0.5, m['title_short'], color=m['color'],
                        transform=ax.transAxes, 
                        fontsize=24, weight='bold', ha='right', va='center', rotation=90)
    
    
    def _set_basin_column_titles(self, fig, ncols, basin_titles):
        """Add basin titles as column headers at the top of the plot."""
        for i, basin_title in enumerate(basin_titles):
            if i < ncols:
                # Get the first subplot in each column
                ax = fig.axes[i]
                ax.text(0.5, 1.05, basin_title, transform=ax.transAxes,
                       fontsize=12, weight='bold', ha='center', va='bottom')

            

    def _set_labels(self,ax,m,j,i, y='Depth (m)', x='Latitude', tit=None):
        if j == 0:
            ax.set_ylabel(y)
        if i == 2:
            ax.set_xlabel(x)
        # keep the basin_title out of individual subplots; we will add a single row label instead
        if i == 0:
            # column title above each first-row axes (acts as column title)
            ax.text(
                0.5, 1.08, m['title'],
                ha='center', va='bottom',
                transform=ax.transAxes,
                fontsize=24, fontweight='bold',
                color=m['color']
            )
        if tit is not None:
            ax.set_title(tit)

    def _set_basin_titles(self,fig, diff=False, per_basin=False):
        # Add vertical row titles (one per basin) on the left, centered vertically next to each row.
        subplot_axes = [ax for ax in fig.axes if hasattr(ax, 'get_subplotspec')]

        if diff:
            if per_basin:
                bt = ['Atlantic (data)', 'Atlantic (difference)', 'Indo-Pacific (data)', 'Indo-Pacific (difference)']
            else:
                bt = ['Atlantic (data)', 'Indo-Pacific (data)', 'Atlantic (difference)', 'Indo-Pacific (difference)']
        else:
            bt = self.basin_titles[:2]
        
            
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
            if self.n > 1:
                x_offset = 0.04
            elif self.n==1:
                x_offset = 0.125
            x_pos = float(min(x_lefts) - x_offset)
            if x_pos < 0.01:
                x_pos = 0.01
    
            # place the vertical text; ha='center' so it's centered on x_pos, va='center' vertically
            fig.text(x_pos, y_center, basin_title,
                     va='center', ha='center', rotation='vertical',
                     fontsize=20, fontweight='bold')

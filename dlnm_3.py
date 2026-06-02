import xarray as xr
import numpy as np
import pandas as pd



effect_ds = xr.open_dataset('F:/lag_global/dlnm_results/fire_counts/dlnm_effect_northAmerica_180.nc')
results_ds = xr.open_dataset('F:/lag_global/dlnm_results/fire_counts/dlnm_results_northAmerica.nc')



target_vars = ['temp', 'soil', 'precip', 'wind']
param_labels = ['Temperature', 'Soil Moisture', 'Precipitation', 'Wind Speed']
sigma_vars = [f'sigma_{v}' for v in target_vars]


risk_mean_arp = effect_ds.risk.mean(dim='arp', skipna=True)  


lat_vals = effect_ds.lat.values
lon_vals = effect_ds.lon.values
lag_vals = effect_ds.lag.values
n_lat, n_lon, n_lag = len(lat_vals), len(lon_vals), len(lag_vals)


max_risk_grid = np.full((n_lat, n_lon), np.nan)


for i in range(n_lat):
    for j in range(n_lon):
        
        risk_series = risk_mean_arp[i, j, :].values
        
       
        if np.any(~np.isnan(risk_series)):
         
            max_risk = np.nanmax(risk_series)
         
            max_risk_grid[i, j] = max_risk


high_risk_mask = max_risk_grid >= 1.1


params = results_ds.params.sel(param=target_vars)
sigmas = results_ds.params.sel(param=sigma_vars)

sigmas_renamed = sigmas.rename({'param': 'sigma_param'})


raw_effects = params * sigmas_renamed

abs_effects = np.abs(raw_effects)


total_effect = abs_effects.sum(dim='param')

relative_contrib = (abs_effects / total_effect) * 100

contrib_grids = {}
for var in target_vars:
    contrib_grids[var] = np.full((n_lat, n_lon), np.nan)


dominant_grid = np.full((n_lat, n_lon), np.nan)


for i in range(n_lat):
    for j in range(n_lon):
        if high_risk_mask[i, j]:
            contrib_values = []
            
            for var in target_vars:
           
                contrib_val = relative_contrib.sel(param=var, lat=lat_vals[i], lon=lon_vals[j])
                
              
                if hasattr(contrib_val, 'values'):
                    val = contrib_val.values
                   
                    if isinstance(val, np.ndarray):
                        if val.size == 1:
                            contrib_scalar = float(val.item())
                        else:
                            
                            contrib_scalar = float(np.nanmean(val))
                    else:
                        contrib_scalar = float(val)
                else:
                    contrib_scalar = float(contrib_val)
                
                contrib_grids[var][i, j] = contrib_scalar
                contrib_values.append(contrib_scalar)
            

            if not np.all(np.isnan(contrib_values)):
                max_idx = np.nanargmax(contrib_values)
                dominant_grid[i, j] = max_idx 

for var, label in zip(target_vars, param_labels):
    data = contrib_grids[var][~np.isnan(contrib_grids[var])]
    if len(data) > 0:
        mean_val = np.mean(data)
        max_val = np.max(data)
        min_val = np.min(data)
        print(f"  {label}: mean_value={mean_val:.1f}%, value_area={min_val:.1f}%-{max_val:.1f}%, effective points={len(data)}")
    else:
        print(f"  {label}: none")


dominant_counts = {}
for idx, label in enumerate(param_labels):
    count = np.sum(dominant_grid == idx)
    dominant_counts[label] = count
    if np.sum(~np.isnan(dominant_grid)) > 0:
        pct = count / np.sum(~np.isnan(dominant_grid)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    else:
        print(f"  {label}: {count}")
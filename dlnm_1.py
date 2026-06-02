# import xarray as xr
# import dask.array as da
# import numpy as np
# import pandas as pd
# from patsy import dmatrix
# import statsmodels.api as sm
# from statsmodels.genmod.families import Poisson
# import warnings
# warnings.filterwarnings('ignore')

# # ===================================================
# # 1. Data preprocessing
# # ===================================================
# def resample_to_daily(ds, time_dim='time'):
#  
#     time = pd.to_datetime(ds[time_dim].values)
#     ds = ds.assign_coords({time_dim: time})
    
#     
#     daily = xr.Dataset()
#     for var in ds.data_vars:
#         if var in ['arp', 'fire_count']:  
#             daily[var] = ds[var].resample({time_dim: '1D'}).sum()
#         else:                            
#             daily[var] = ds[var].resample({time_dim: '1D'}).mean()
#     return daily

# def load_data(years, lat_range, lon_range):
#     
#     def preprocess(ds):
#         return ds[['arp', 'fire_count', 'temperature', 'soil_moisture', 'wind_speed']].sel(
#             lat=slice(*lat_range),
#             lon=slice(*lon_range)
#         )
    
#     paths = [f'F:/lag_global/northAmerica/{year}.nc' for year in years]
#     ds = xr.open_mfdataset(
#         paths,
#         combine='nested',
#         concat_dim='time',
#         preprocess=preprocess,
#         chunks={'time': 720, 'lat': 50, 'lon': 50},
#         parallel=True
#     )
    
#     
#     ds_daily = resample_to_daily(ds)
    
#     
#     ar_mask = ds_daily['arp'] > 0
    
#     
#     temp_ar = ds_daily['temperature'].where(ar_mask, 0)
#     wind_ar = ds_daily['wind_speed'].where(ar_mask, 0)
    
#     window = 180
#     ds_daily['temp_cum_ar'] = temp_ar.rolling(time=window, min_periods=1).sum()
#     ds_daily['wind_cum_ar'] = wind_ar.rolling(time=window, min_periods=1).sum()
    
#    
#     ds_daily = ds_daily.chunk({'lat': 50, 'lon': 50, 'time': 180})
    
#     return ds_daily

# # ===================================================
# # 2. Cross-base construction
# # ===================================================
# def natural_spline(x, df=3):
#     try:
#         return dmatrix(f"cr(x, df={df}) - 1", {"x": x}, return_type='dataframe').values
#     except:
#         return np.zeros((len(x), df))

# def build_crossbasis_for_var(var, max_lag=180, var_df=3, lag_df=4):
#     n = len(var)
#     var_clean = np.nan_to_num(var, nan=np.nanmean(var))
#     var_basis = natural_spline(var_clean, var_df)
#     lag_basis = natural_spline(np.arange(max_lag+1), lag_df)
#     cross = np.zeros((n, var_df * lag_df))
#     for lag in range(max_lag+1):
#         start = max(0, lag)
#         end = n - (max_lag - lag)
#         if start >= end:
#             continue
#         lagged_var = var_basis[start:end]
#         interaction = lagged_var[:, :, None] * lag_basis[lag][None, :]
#         pos = max_lag - lag
#         cross[pos:pos+end-start] += interaction.reshape(-1, var_df*lag_df)
#     return cross

# # ===================================================
# # 3. Model fitting
# # ===================================================
# def fit_model(data):
#     
#     try:
#         n_days = len(data)
#         dates = pd.date_range(start='2002-01-01', periods=n_days, freq='D')
#         df = pd.DataFrame({
#             'arp': data[:,0],
#             'fire': np.round(data[:,1]).astype(int),
#             'temp_cum_ar': data[:,5],
#             'soil': data[:,3],
#             'wind_cum_ar': data[:,6],
#             'month': dates.month
#         }).dropna()
        
#         if len(df) < 30 or df['fire'].sum() == 0:
#             return np.full(1+12+12+2+1+8, np.nan)
        
#         continuous = ['arp', 'temp_cum_ar', 'soil', 'wind_cum_ar']
#         means = df[continuous].mean().values
#         stds = df[continuous].std().values
#         df[continuous] = (df[continuous] - means) / stds
        
#         cb_arp = build_crossbasis_for_var(df['arp'])
#         cb_soil = build_crossbasis_for_var(df['soil'])
        
#         X = pd.DataFrame(cb_arp).add_prefix('cb_arp_')
#         X_soil = pd.DataFrame(cb_soil).add_prefix('cb_soil_')
#         X = pd.concat([X, X_soil], axis=1)
#         X['temp_cum_ar'] = df['temp_cum_ar']
#         X['wind_cum_ar'] = df['wind_cum_ar']
#         X['season'] = np.sin(2 * np.pi * df['month'] / 12)
#         X = sm.add_constant(X, has_constant='add')
        
#         model = sm.GLM(df['fire'], X, family=Poisson()).fit_regularized(
#             alpha=0.001, L1_wt=0.0, maxiter=2000, cnvrg_tol=1e-3, refit=True
#         )
#         params = model.params.values
#         std_params = np.concatenate([means, stds])
#         return np.concatenate([params, std_params])
#     except Exception:
#         return np.full(1+12+12+2+1+8, np.nan)

# # ===================================================
# # 4. Parallel computing
# # ===================================================
# def parallel_processing(ds):
#     
#     data = da.stack([
#         ds.arp.data,
#         ds.fire_count.data,
#         ds.temperature.data,
#         ds.soil_moisture.data,
#         ds.wind_speed.data,
#         ds.temp_cum_ar.data,
#         ds.wind_cum_ar.data
#     ], axis=-1).transpose(1, 2, 0, 3)  # (lat, lon, time, 7)
    
#     n_total_params = 36
#     def process_chunk(chunk):
#         n_lat, n_lon, n_time, n_var = chunk.shape
#         results = np.full((n_lat, n_lon, n_total_params), np.nan, dtype=np.float32)
#         for i in range(n_lat):
#             for j in range(n_lon):
#                 ts = chunk[i, j, :, :]
#                 if ts.shape != (n_time, 7) or np.isnan(ts).any():
#                     continue
#                 params = fit_model(ts)
#                 if len(params) == n_total_params:
#                     results[i, j] = params
#         return results
    
#     lat_chunks = ds.chunks['lat']
#     lon_chunks = ds.chunks['lon']
#     results = da.map_blocks(
#         process_chunk, data, dtype=np.float32,
#         chunks=(lat_chunks, lon_chunks, (n_total_params,)),
#         drop_axis=[2, 3], new_axis=2
#     )
#     results = results[:len(ds.lat), :len(ds.lon), :]
#     param_names = (
#         ['const'] +
#         [f'cb_arp_{i}' for i in range(12)] +
#         [f'cb_soil_{i}' for i in range(12)] +
#         ['temp_cum_ar', 'wind_cum_ar', 'season'] +
#         ['mu_arp', 'sigma_arp', 'mu_temp_cum_ar', 'sigma_temp_cum_ar',
#          'mu_soil', 'sigma_soil', 'mu_wind_cum_ar', 'sigma_wind_cum_ar']
#     )
#     return xr.Dataset(
#         {'params': (['lat', 'lon', 'param'], results)},
#         coords={'lat': ds.lat.values, 'lon': ds.lon.values, 'param': param_names}
#     )

# # ===================================================
# # 5. Main program
# # ===================================================
# if __name__ == "__main__":
#     data_config = {
#         'years': range(2002, 2024),
#         'lat_range': (22, 40),
#         'lon_range': (-96, -68)
#     }
#     output_config = {
#         'output_path': 'F:/lag_global/dlnm_results/fire_counts/dlnm_results_northAmerica.nc',
#         'encoding': {'params': {'zlib': True, 'complevel': 5, 'chunksizes': (50, 50, 36), 'dtype': 'float32'}}
#     }
#     print("Loading and resampling 6-hourly data to daily...")
#     ds = load_data(**data_config)
#     print(f"Daily data shape: lat={len(ds.lat)}, lon={len(ds.lon)}, time={len(ds.time)}")
#     print("Starting parallel DLNM fitting...")
#     result = parallel_processing(ds)
#     print("Saving results...")
#     result.to_netcdf(output_config['output_path'], encoding=output_config['encoding'], compute=True)
#     print("Done.")
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xarray as xr
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc


effect_ds = xr.open_dataset('F:/lag_global/dlnm_results/fire_counts/dlnm_effect_northAmerica_180.nc')
results_ds = xr.open_dataset('F:/lag_global/dlnm_results/fire_counts/dlnm_results_northAmerica.nc')

target_vars = ['temp', 'soil', 'precip', 'wind']
param_labels = ['Temperature', 'Soil Moisture', 'Precipitation', 'Wind Speed']
sigma_vars = [f'sigma_{v}' for v in target_vars]

risk_mean_arp = effect_ds.risk.mean(dim='arp', skipna=True)


lat_vals = effect_ds.lat.values
lon_vals = effect_ds.lon.values
n_lat, n_lon, n_lag = len(lat_vals), len(lon_vals), len(effect_ds.lag.values)


max_risk_grid = np.full((n_lat, n_lon), np.nan)


for i in range(n_lat):
    for j in range(n_lon):

        risk_series = risk_mean_arp[i, j, :].values
        

        if np.any(~np.isnan(risk_series)):

            max_risk = np.nanmax(risk_series)

            max_risk_grid[i, j] = max_risk

high_risk_mask = max_risk_grid > 1.0

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


dominant_grid_all = np.full((n_lat, n_lon), np.nan)


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
                dominant_grid_all[i, j] = max_idx  

data_list = []

for i in range(n_lat):
    for j in range(n_lon):
        if high_risk_mask[i, j]:
           
            temp_contrib = contrib_grids['temp'][i, j]
            soil_contrib = contrib_grids['soil'][i, j]
            precip_contrib = contrib_grids['precip'][i, j]
            wind_contrib = contrib_grids['wind'][i, j]
            
          
            contrib_values = [temp_contrib, soil_contrib, precip_contrib, wind_contrib]
            if not np.all(np.isnan(contrib_values)):
                
                sorted_indices = np.argsort(contrib_values)[::-1]  
                top1_idx = sorted_indices[0]
                top2_idx = sorted_indices[1]
                
                if top1_idx in [1, 3]:  
                    dominant_factor_binary = top1_idx  
                    
                    if top1_idx == 1:
                        dominant_factor_binary = 0  
                    elif top1_idx ==3:
                            dominant_factor_binary = 1  
                    
                    row_data = {
                        'latitude': lat_vals[i],
                        'longitude': lon_vals[j],
                        'temp_contrib': temp_contrib,
                        'soil_contrib': soil_contrib,
                        'precip_contrib': precip_contrib,
                        'wind_contrib': wind_contrib,
                        'top1_factor': top1_idx,
                        'top2_factor': top2_idx,
                        'dominant_factor_binary': dominant_factor_binary,
                        'max_risk': max_risk_grid[i, j]
                    }
                    data_list.append(row_data)

df_nc = pd.DataFrame(data_list)

dominant_factor_df = df_nc.groupby(['latitude', 'longitude']).agg({
    'temp_contrib': 'mean',
    'soil_contrib': 'mean',
    # 'precip_contrib': 'mean',
    # 'wind_contrib': 'mean',
    'dominant_factor_binary': 'first'  
}).reset_index()

ndvi_path = "E:/map/MOD13C1_NDVI_MEAN.tif"

with rasterio.open(ndvi_path) as src:
    ndvi_data = src.read(1)
    height, width = ndvi_data.shape
    transform = src.transform
    

    x = np.arange(width) * transform.a + transform.c
    y = np.arange(height) * transform.e + transform.f


lats_unique = np.sort(dominant_factor_df['latitude'].unique())
lons_unique = np.sort(dominant_factor_df['longitude'].unique())

lat_grid, lon_grid = np.meshgrid(lats_unique, lons_unique, indexing='ij')
dominant_factor_grid_binary = np.full_like(lat_grid, np.nan, dtype=float)

for _, row in dominant_factor_df.iterrows():
    lat_idx = np.where(lats_unique == row['latitude'])[0][0]
    lon_idx = np.where(lons_unique == row['longitude'])[0][0]
    dominant_factor_grid_binary[lat_idx, lon_idx] = row['dominant_factor_binary']


ndvi_da = xr.DataArray(
    data=ndvi_data,
    dims=['y', 'x'],
    coords={'y': y, 'x': x},
    name='ndvi'
)

ndvi_resampled = ndvi_da.interp(
    y=lats_unique,
    x=lons_unique,
    method='nearest'
).rename({'y': 'lat', 'x': 'lon'})


flat_dominant = dominant_factor_grid_binary.flatten()
flat_ndvi = ndvi_resampled.values.flatten()

df = pd.DataFrame({
    'lat': np.repeat(lats_unique, len(lons_unique)),
    'lon': np.tile(lons_unique, len(lats_unique)),
    'dominant_factor': flat_dominant,
    'ndvi': flat_ndvi
})


df_clean = df.dropna(subset=['dominant_factor', 'ndvi'])
df_clean = df_clean[(df_clean['ndvi'] >= -0.2) & (df_clean['ndvi'] <= 1.0)]

temp_count = (df_clean['dominant_factor'] == 0).sum()
soil_count = (df_clean['dominant_factor'] == 1).sum()

df_clean['lat_norm'] = (df_clean['lat'] - df_clean['lat'].min()) / (df_clean['lat'].max() - df_clean['lat'].min())
df_clean['lon_norm'] = (df_clean['lon'] - df_clean['lon'].min()) / (df_clean['lon'].max() - df_clean['lon'].min())


df_clean['ndvi_squared'] = df_clean['ndvi'] ** 2


df_clean['ndvi_lat_interaction'] = df_clean['ndvi'] * df_clean['lat_norm']
df_clean['ndvi_lon_interaction'] = df_clean['ndvi'] * df_clean['lon_norm']


features = ['ndvi', 'lat_norm', 'lon_norm', 'ndvi_squared', 'ndvi_lat_interaction', 'ndvi_lon_interaction']
X = df_clean[features]
y = df_clean['dominant_factor']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


clf = RandomForestClassifier(n_estimators=200, max_depth=20, 
                            min_samples_split=5, min_samples_leaf=2,
                            random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"{accuracy:.2%}")
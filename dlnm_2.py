import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from patsy import dmatrix

def natural_spline(x, df=3):
    try:
        if len(x) < df:
            x = np.linspace(np.min(x), np.max(x), df*2)
        basis = dmatrix(f"cr(x, df={df}) - 1", {"x": x}, return_type='dataframe')
        return basis.values
    except Exception as e:
        print(f"Error in natural_spline: {str(e)}")
        return np.zeros((len(x), df))

def rebuild_basis(arp_range=(-3, 3), lag_range=(0, 180), var_df=3, lag_df=4):
    arp_values = np.linspace(arp_range[0], arp_range[1], 100)
    lag_values = np.arange(lag_range[0], lag_range[1]+1)
    var_basis = natural_spline(arp_values, df=var_df)
    lag_basis = natural_spline(lag_values, df=lag_df)
    cross_basis = np.zeros((len(arp_values), len(lag_values), var_df*lag_df))
    for i in range(var_df):
        for j in range(lag_df):
            cross_basis[:, :, i*lag_df+j] = np.outer(var_basis[:,i], lag_basis[:,j])
    return arp_values, lag_values, cross_basis

def parse_single_point(point_params, arp_grid, lag_grid, basis_matrix, var_df=3, lag_df=4):
    """
    point_params: 
    const(0), cb_arp(1-12), cb_soil(13-24), temp_cum_ar(25), wind_cum_ar(26), season(27),
	mu_arp(28), sigma_arp(29), mu_temp_cum_ar(30), sigma_temp_cum_ar(31),
	mu_soil(32), sigma_soil(33), mu_wind_cum_ar(34), sigma_wind_cum_ar(35)
    """
    
    cb_params = point_params[1:1+var_df*lag_df] 
    
   
    mu_arp = point_params[28]
    sigma_arp = point_params[29]
    
    
    effect_surface = np.dot(
        basis_matrix.reshape(-1, var_df*lag_df),
        cb_params
    ).reshape(len(arp_grid), len(lag_grid))
    
    risk_surface = np.exp(effect_surface)
    actual_arp = arp_grid * sigma_arp + mu_arp  
    return effect_surface, risk_surface, actual_arp

def process_all_points(ds_result, arp_grid, lag_grid, basis_matrix, var_df=3, lag_df=4):
    params = ds_result.params.values  # (lat, lon, 36)
    lat = ds_result.lat.values
    lon = ds_result.lon.values
    
    effect_cube = np.zeros((len(lat), len(lon), len(arp_grid), len(lag_grid)))
    actual_arp_cube = np.zeros((len(lat), len(lon), len(arp_grid)))
    
    for i in range(len(lat)):
        print(f"Processing latitude {i+1}/{len(lat)}")
        for j in range(len(lon)):
            try:
                eff, _, actual_arp = parse_single_point(
                    params[i, j, :], arp_grid, lag_grid, basis_matrix, var_df, lag_df
                )
                effect_cube[i, j] = eff
                actual_arp_cube[i, j] = actual_arp
            except Exception as e:
                print(f"Error at ({i},{j}): {str(e)}")
                effect_cube[i, j] = np.nan
                actual_arp_cube[i, j] = np.nan
    
    ds_output = xr.Dataset(
        {
            'effect': (['lat', 'lon', 'arp', 'lag'], effect_cube),
            'risk': (['lat', 'lon', 'arp', 'lag'], np.exp(effect_cube)),
            'actual_arp': (['lat', 'lon', 'arp'], actual_arp_cube)
        },
        coords={
            'lat': lat,
            'lon': lon,
            'arp': arp_grid,
            'lag': lag_grid
        }
    )
    return ds_output

if __name__ == "__main__":
   
    result = xr.open_dataset('F:/lag_global/dlnm_results/fire_counts/dlnm_results_northAmerica.nc')
    
    
    params_shape = result.params.shape
    
    var_df = 3
    lag_df = 4
    max_lag = 180
    arp_range = (-3, 3)  
    
    arp_grid, lag_grid, basis_matrix = rebuild_basis(
        arp_range=arp_range, lag_range=(0, max_lag),
        var_df=var_df, lag_df=lag_df
    )
    
    ds_output = process_all_points(result, arp_grid, lag_grid, basis_matrix, var_df, lag_df)
    ds_output.to_netcdf('F:/lag_global/dlnm_results/fire_counts/dlnm_effect_northAmerica_180.nc')
    print("Done")
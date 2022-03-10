import os
import geopandas as gpd
import numpy
import rioxarray
import rasterio as rio
import xarray as xr
from geocube.api.core import make_geocube        
def process_data(admin_df):
    ####  admin boundary shape file 
    admin_df =gpd.read_file('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/IBF_DROUGHT_PIPELINE/pipeline/data/input/ken_admin1.geojson')
    ### import your raser file here , in this case the raster file was stored in a netcdf format 
    ## this can also be in geotiff format 

    ds_disk =rioxarray.open_rasterio("C:/data/icpac/PredictedProbabilityRain_Feb-Apr_Jan2022.nc")
    ### if the raster file has multiple formats select the band which is relevant for the analysis
    precipitation=ds_disk['below'].rio.write_crs("epsg:4326", inplace=True).rio.clip(admin_df.geometry.values, admin_df.crs, from_disk=True).sel(band=1).drop("band")

    precipitation.name = "precipitation"
    ### create a new unique identifier with type integer 
    admin_df['ind'] = admin_df.apply(lambda row: row.ADM1_PCODE[-3:], axis=1)

    admin_df["pcode"] = admin_df.ind.astype(int)

    # make your geo cube 
    out_grid = make_geocube(
        vector_data=admin_df,
        measurements=["pcode"],
        like=precipitation, # ensure the data are on the same grid
    )


    # merge the two together
    out_grid["precipitation"] = (precipitation.dims, precipitation.values, precipitation.attrs, precipitation.encoding)
    out_grid.pcode.plot.imshow()
    out_grid.precipitation.plot()


    grouped_precipitation = out_grid.drop("spatial_ref").groupby(out_grid.pcode)

    grid_mean = grouped_precipitation.mean().rename({"precipitation": "precipitation_mean"})
    grid_median = grouped_precipitation.median().rename({"precipitation": "precipitation_median"})
     
    zonal_stats_df = xr.merge([grid_mean, grid_median]).to_dataframe().reset_index()
     
    zonal_stats_df['ADM1_PCODE'] = zonal_stats_df.apply(lambda row: 'KE'+str(int(row.pcode)).zfill(3), axis=1)
 
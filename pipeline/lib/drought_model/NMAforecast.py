#%%

import pandas as pd
from shapely.geometry import Point
import fiona
import matplotlib.pyplot as plt
import pdfplumber
import geopandas as gpd
import tabula
from tabula import read_pdf
import re
import json
import numpy as np
import os
import pandas as pd
from shapely.geometry import Point
import fiona
import rioxarray
import rasterio as rio
from geocube.api.core import make_geocube
import rioxarray
import xarray as xr
import glob
#%%
admin_df = gpd.read_file('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/ethiopia/admin/admin2.geojson') #self.ADMIN_AREA_GDF   #

tsv_files = glob.glob('C:/data/FMAM_JanInitial_Forecast_Files/NextGen_PRCPPRCP_CCAFCST_mu*.tsv')
forecast_df=[]
for spiforecast in tsv_files: 
    forecat_period=spiforecast.split('.')[0].split('/')[-1].split('_')[-2]
    forecat_time=spiforecast.split('.')[0].split('/')[-1].split('_')[-1]
    with open(spiforecast) as file:
        ## the first two lines are not useful 
        file.readline()
        file.readline()
        # line 3 conaines information on probability, ncol, nrow 
        cpt_items=file.readline().strip('\n').split(',')
        #clim_prob=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:clim_prob'][0]
        ncol=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:ncol'][0]
        nrow=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:nrow'][0]
        X=file.readline().split()
        data=[]
        Lines = file.readlines()
        # loop through the rest of the lines, append precipitation values to a list in a format x,y,precipitation, probability
        for line in Lines:
            temp_data=line.split()
            if all([not line.startswith('cpt' ), len(temp_data)!=int(ncol)]):   
                for j in range(int(ncol)):
                    data.append([float(temp_data[0]),float(X[j]),float(temp_data[j+1])])       
            elif line.startswith('cpt' ):
                cpt_items=line.strip('\n').split(',')
                clim_prob=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:clim_prob'][0]


    #file_name= self.outputPath + '/' + 'spi_forecast.csv'        
    df = pd.DataFrame(data, index =None, columns =['y','x',forecat_time])
    df.replace({'precipitation': -999}, np.nan,inplace=True)
    df.set_index(['y', 'x'],inplace=True)
    precipitation=df.to_xarray()
    forecast_df.append(precipitation)
    
forecast_df_all = xr.merge(forecast_df).rio.write_crs("epsg:4326", inplace=True)
list(forecast_df_all.keys())

### create a new unique identifier with type integer 
admin_df['pcode'] = admin_df.apply(lambda row: row.ADM2_PCODE[-4:], axis=1)
admin_df["pcode"] = admin_df.pcode.astype(int)

#%%

#use first dataframe 

precipitation =forecast_df[0].rio.write_crs("epsg:4326", inplace=True).rio.clip(admin_df.geometry.values, admin_df.crs, from_disk=True)

# make your geo cube 
data_cube = make_geocube(
    vector_data=admin_df,
    measurements=["pcode"],
    like=precipitation, # ensure the data are on the same grid
)

#%%

# merge the two together
data_cube["precipitation"] = (precipitation.dims, precipitation.values, precipitation.attrs, precipitation.encoding)

grouped_precipitation = data_cube.drop("spatial_ref").groupby(data_cube.pcode)

grid_mean = grouped_precipitation.mean().rename({"precipitation": "amount"})
grid_median = grouped_precipitation.median().rename({"precipitation": "precipitation_median"})
    
zonal_stats_df = xr.merge([grid_mean, grid_median]).to_dataframe().reset_index()
    
zonal_stats_df['placeCode'] = zonal_stats_df.apply(lambda row: 'ET'+str(int(row.pcode)).zfill(4), axis=1) 
zonal_stats_df=zonal_stats_df[['placeCode','amount']]        
#stats=population_affected.to_dict(orient='records')
#zonal_stats_df['trigger']=zonal_stats_df['amount'].apply(lambda x: 1 if x>self.TRIGGER_PROBABILITY_RAIN else 0)
#file_name=self.output_filepath.split('.')[0]+'.csv'
#zonal_stats_df.to_csv(file_name)

#%%
#%%
#read observation 
spiforecast='C:/data/FMAM_JanInitial_Forecast_Files/obs_PRCP_Feb-May.tsv'
forecat_period=spiforecast.split('.')[0].split('/')[-1].split('_')[-1]

with open(spiforecast) as file:
    ## the first two lines are not useful 
    file.readline()
    file.readline()
    file.readline()
    cpt_timestamp=file.readline().strip('cpt:T').split('\t')
    # line 3 conaines information on probability, ncol, nrow 
    cpt_items=file.readline().strip('\n').split(',')
    #clim_prob=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:clim_prob'][0]
    ncol=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:ncol'][0]
    nrow=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:nrow'][0]
    clim_time=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:T'][0]
    X=file.readline().split()
    data=[]
    Lines = file.readlines()
    # loop through the rest of the lines, append precipitation values to a list in a format x,y,precipitation, probability
    for line in Lines:
        temp_data=line.split()
        if all([not line.startswith('cpt' ), len(temp_data)!=int(ncol)]):   
            for j in range(int(ncol)):
                data.append([float(temp_data[0]),float(X[j]),float(temp_data[j+1]),clim_time])       
        elif line.startswith('cpt' ):
            cpt_items=line.strip('\n').split(',')
            clim_time=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:T'][0]
    #file_name= self.outputPath + '/' + 'spi_forecast.csv'        
df_observtion = pd.DataFrame(data, index =None, columns =['y','x','precipitation','forecat_time'])
df_observtion.replace({'precipitation': -999}, np.nan,inplace=True)
df_observtion['forecat_time'] = pd.to_datetime(df_observtion['forecat_time'], format='%Y-%m/%d')
df_observtion.set_index(['y', 'x','forecat_time'],inplace=True)
precipitation=df_observtion.to_xarray()
ds_seasonal = precipitation.quantile(q=0.33,dim='forecat_time')
ds_seasonal =ds_seasonal.drop("quantile")

#%%
dff=forecast_df_all['Jan2011'].to_dataset()
dff=dff['Jan2011']-ds_seasonal['precipitation']
dff=dff.where(dff < 0)
dff=dff.to_dataset(name='trigger')
dff.trigger.plot()

#%%

spiforecast='C:/data/FMAM_JanInitial_Forecast_Files/NextGen_PRCPPRCP_CCAFCST_var_Feb-May_Jan2011.tsv'
#%%
with open(spiforecast) as file:
    ## the first two lines are not useful 
    file.readline()
    file.readline()
    # line 3 conaines information on probability, ncol, nrow 
    cpt_items=file.readline().strip('\n').split(',')
    clim_prob=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:clim_prob'][0]
    ncol=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:ncol'][0]
    nrow=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:nrow'][0]
    X=file.readline().split()
    data=[]
    Lines = file.readlines()
    # loop through the rest of the lines, append precipitation values to a list in a format x,y,precipitation, probability
    for line in Lines:
        temp_data=line.split()
        if all([not line.startswith('cpt' ), len(temp_data)!=int(ncol)]):   
            for j in range(int(ncol)):
                data.append([float(temp_data[0]),float(X[j]),float(temp_data[j+1])])       
        elif line.startswith('cpt' ):
            cpt_items=line.strip('\n').split(',')
            clim_prob=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:clim_prob'][0]

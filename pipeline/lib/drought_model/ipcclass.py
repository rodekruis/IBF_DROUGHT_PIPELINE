# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:19:24 2021

@author: ATeklesadik
"""

import subprocess
import requests
import urllib.request
import zipfile
import os
import shutil
import geopandas as gpd
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
import glob
from zipfile import ZipFile
import datetime as dt
import pandas as pd
import json
import rasterio
from rasterio.plot import show
from rasterstats import zonal_stats
from drought_model.settings import *
from datetime import datetime, timedelta,date

try:
    from drought_model.secrets import *
except ImportError:
    print("No secrets file found.")

class IPCCLASS:
    def __init__(
        self,
        countryCodeISO3, 
        admin_area_gdf,       
    ):
        self.countryCodeISO3 = countryCodeISO3
        self.levels = SETTINGS[countryCodeISO3]["levels"]
        self.ipcCountrCode = SETTINGS[countryCodeISO3]["ipcCountrCode"]
        self.PIPELINE_OUTPUT = PIPELINE_OUTPUT
        self.PIPELINE_INPUT = PIPELINE_INPUT
        self.CURRENT_Year=CURRENT_Year
        self.Now_Month_nummeric=Now_Month_nummeric
        self.ADMIN_AREA_GDF=admin_area_gdf
        

    def downloadipc(self):
        """ """
        if self.countryCodeISO3 !='ZMB':
            lencontent=4
            deltadate=0
            CURRENT_DATE = date.today()
            while lencontent < 100: #LOOP to find path for the latest forecast 
                CURRENT_DATE=CURRENT_DATE - timedelta(deltadate) # to use last month forecast
                Now_Month_nummeric = CURRENT_DATE.strftime("%m")
                CURRENT_Year = CURRENT_DATE.year
                yearmonth = f"{CURRENT_Year}-{Now_Month_nummeric}"
                yearmonth_ = f"{CURRENT_Year}{Now_Month_nummeric}"
                #url = f'https://fdw.fews.net/api/ipcpackage/?country_code=ET&collection_date={yearmonth}-01'
                url = f'https://fdw.fews.net/api/ipcpackage/?country_code={self.ipcCountrCode}&collection_date={yearmonth}-01'
                response = requests.get(url)  
                lencontent=len(response.content)
                deltadate=30
                new_ipc_url=url
                    
                filename = (
                    self.PIPELINE_INPUT + 
                    "ipc/raw/"+ 
                    f"{self.ipcCountrCode}_{yearmonth_}.zip"
                    )       

        else:            
            lencontent=4
            deltadate=0
            CURRENT_DATE = date.today()
            while lencontent < 100: #LOOP to find path for the latest forecast 
                CURRENT_DATE=CURRENT_DATE - timedelta(deltadate) # to use last month forecast
                Now_Month_nummeric = CURRENT_DATE.strftime("%m")
                CURRENT_Year = CURRENT_DATE.year
                yearmonth = f"{CURRENT_Year}-{Now_Month_nummeric}"
                yearmonth_ = f"{CURRENT_Year}{Now_Month_nummeric}"
                url = f"https://fdw.fews.net/api/ipcpackage/?country_group=903&collection_date={yearmonth}-01"
                response = requests.get(url)  
                lencontent=len(response.content)
                deltadate=30
                new_ipc_url=url              
                filename = (
                    self.PIPELINE_INPUT + 
                    "ipc/raw/"+ 
                    f"southern_afria_{yearmonth_}.zip"
                    ) 
                
        print("Downloading shapefile {0}".format(filename))

        urllib.request.urlretrieve("{0}".format(new_ipc_url), filename)
        return yearmonth_

    def ipc_proccessing(self):
        yearmonth_=self.downloadipc()
        
        directory_to_extract_to = self.PIPELINE_INPUT + "ipc/raw/"
        
        if self.countryCodeISO3 !='ZMB':
            path_to_zip_file = (self.PIPELINE_INPUT + "ipc/raw/" + f"{self.ipcCountrCode}_{yearmonth_}.zip")

            with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                zip_ref.extractall(directory_to_extract_to)

            ML1_FILE = (
                self.PIPELINE_INPUT
                + "ipc/raw/"
                + f"{self.ipcCountrCode}_{yearmonth_}_ML1.shp"
            )
            ML2_FILE = (
                self.PIPELINE_INPUT
                + "ipc/raw/"
                + f"{self.ipcCountrCode}_{yearmonth_}_ML2.shp"
            )
        
        else:
            path_to_zip_file = (
                self.PIPELINE_INPUT
                + "ipc/raw/"
                + f"southern_afria_{yearmonth_}.zip"
            )

            with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                zip_ref.extractall(directory_to_extract_to)

            ML1_FILE = (
                self.PIPELINE_INPUT
                + "ipc/raw/"
                + f"SA_{yearmonth_}_ML1.shp"
            )
            ML2_FILE = (
                self.PIPELINE_INPUT
                + "ipc/raw/"
                + f"SA_{yearmonth_}_ML2.shp"
            )
            
        ML1_ = gpd.read_file(ML1_FILE)
        ML2_ = gpd.read_file(ML2_FILE)
        
        gdf_lhz_cs_merged = gpd.sjoin(self.ADMIN_AREA_GDF, ML1_, how="left")
        
        df = gdf_lhz_cs_merged[["placeCode", "ML1"]].query("ML1 < 6")
        df = df.iloc[df.reset_index().groupby(["placeCode"])["ML1"].idxmax()]
        # gdf_lhz_cs_merged = gpd.sjoin(moz_lzh, CS_,how='left')
        gdf_lhz_cs_merged = gpd.sjoin(self.ADMIN_AREA_GDF, ML2_, how="left")
        # df=gdf_lhz_cs_merged[['FNID','CS']].query('CS < 6')
        df2 = gdf_lhz_cs_merged[["placeCode", "ML2"]].query("ML2 < 6")
        df2 = df2.iloc[df2.reset_index().groupby(["placeCode"])["ML2"].idxmax()]
        IPCdf = df.set_index("placeCode").join(df2.set_index("placeCode"))
        
        IPCdf.reset_index(inplace=True)
        IPCdf.rename(columns={"ML1":"IPC_forecast_short","ML2": "IPC_forecast_long"},inplace=True)
        df2=IPCdf.copy()
        cols = ["IPC_forecast_short", "IPC_forecast_long"]
        df2[cols] = df2[cols].apply(pd.to_numeric, errors="coerce", axis=1)
        df2 = df2.fillna(0)

      

        ipc_df = pd.merge(
            self.ADMIN_AREA_GDF, df2, how="left", left_on="placeCode", right_on="placeCode"
        )
        

        for indicator in ["IPC_forecast_short", "IPC_forecast_long"]:  # df2.columns:
            for adm_level in self.levels:  # SETTINGS[self.countryCodeISO3]["levels"]:
                df_stats = pd.DataFrame()
                df_stats["placeCode"] =  ipc_df["placeCode"] #ipc_df[f"ADM{adm_level}_PCODE"]                
                df_stats["amount"] = ipc_df[indicator]
                df_stats_levl = df_stats.groupby("placeCode").agg({"amount": "max"})
                df_stats_levl.reset_index(inplace=True)             
                df_stats_levl.dropna(inplace=True)
                df_stats_levl["amount"] = [int(i) for i in df_stats_levl['amount'].values]
                df_stats_levl = df_stats_levl[["amount", "placeCode"]].to_dict(
                    orient="records"
                )
                statsPath = (
                    self.PIPELINE_OUTPUT
                    + "dynamic_indicators/indiator_"
                    # + str(self.leadTimeValue)
                    + "_"
                    + self.countryCodeISO3
                    + "_admin_"
                    + str(adm_level)
                    + "_"
                    + indicator
                    + ".json"
                )

                exposure_data = {
                    "countryCodeISO3": self.countryCodeISO3,
                    "adminLevel": adm_level, 
                    "indicator": indicator, 
                    #"dynamicIndicator": indicator, #"exposurePlaceCodes": df_stats_levl,
                    "dataPlaceCode": df_stats_levl,
                    # "leadTime": self.leadTimeLabel,
                     # + '_affected',
                    
                }

                with open(statsPath, "w") as fp:
                    json.dump(exposure_data, fp)
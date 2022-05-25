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
import urllib.request
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

try:
    from drought_model.secrets import *
except ImportError:
    print("No secrets file found.")


class IPCCLASS:
    def __init__(
        self,
        countryCodeISO3,        
    ):
        self.countryCodeISO3 = countryCodeISO3
        self.levels = SETTINGS[countryCodeISO3]["levels"]
        self.PIPELINE_OUTPUT = PIPELINE_OUTPUT
        self.PIPELINE_INPUT = PIPELINE_INPUT
        self.CURRENT_Year=CURRENT_Year
        self.Now_Month_nummeric=Now_Month_nummeric
        

        admin_woreda_eth = self.PIPELINE_INPUT + "ETH_adm3.geojson"
        self.eth_admin = gpd.read_file(admin_woreda_eth)  # fc.admin_area_gdf


    def downloadipc(self,):
        """ """
        yearmonth = f"{self.CURRENT_Year}-{self.Now_Month_nummeric}"

        filename = (
            self.PIPELINE_INPUT
            + "ipc/raw/"
            + f"east_afria_{self.CURRENT_Year}-{self.Now_Month_nummeric}.zip"
        )

        new_ipc_url = f"https://fdw.fews.net/api/ipcpackage/?country_group=902&collection_date={yearmonth}-01"

        print("Downloading shapefile {0}".format(filename))

        urllib.request.urlretrieve("{0}".format(new_ipc_url), filename)

    def proccessing(self):
        self.downloadipc()
        directory_to_extract_to = self.PIPELINE_INPUT + "ipc/raw/"

        path_to_zip_file = (
            self.PIPELINE_INPUT
            + "ipc/raw/"
            + f"east_afria_{self.CURRENT_Year}-{self.Now_Month_nummeric}.zip"
        )

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        ML1_FILE = (
            self.PIPELINE_INPUT
            + "ipc/raw/"
            + f"EA_{self.CURRENT_Year}{self.Now_Month_nummeric}_ML1.shp"
        )
        ML2_FILE = (
            self.PIPELINE_INPUT
            + "ipc/raw/"
            + f"EA_{self.CURRENT_Year}{self.Now_Month_nummeric}_ML2.shp"
        )

        ML1_ = gpd.read_file(ML1_FILE)
        ML2_ = gpd.read_file(ML2_FILE)
        gdf_lhz_cs_merged = gpd.sjoin(self.eth_admin, ML1_, how="left")
        df = gdf_lhz_cs_merged[["ADM3_PCODE", "ML1"]].query("ML1 < 6")
        df = df.iloc[df.reset_index().groupby(["ADM3_PCODE"])["ML1"].idxmax()]
        # gdf_lhz_cs_merged = gpd.sjoin(moz_lzh, CS_,how='left')
        gdf_lhz_cs_merged = gpd.sjoin(self.eth_admin, ML2_, how="left")
        # df=gdf_lhz_cs_merged[['FNID','CS']].query('CS < 6')
        df2 = gdf_lhz_cs_merged[["ADM3_PCODE", "ML2"]].query("ML2 < 6")
        df2 = df2.iloc[df2.reset_index().groupby(["ADM3_PCODE"])["ML2"].idxmax()]
        IPCdf = df.set_index("ADM3_PCODE").join(df2.set_index("ADM3_PCODE"))

        df2 = pd.DataFrame()
        df2[["IPC_forecast_short", "IPC_forecast_long"]] = IPCdf[["ML1", "ML2"]]
        df2["placeCode"] = IPCdf.index
        cols = ["IPC_forecast_short", "IPC_forecast_long"]
        df2[cols] = df2[cols].apply(pd.to_numeric, errors="coerce", axis=1)
        df2 = df2.fillna(0)

        ipc_df = pd.merge(
            self.eth_admin, df2, how="left", left_on="ADM3_PCODE", right_on="ADM3_PCODE"
        )
        
        url = IBF_API_URL + "/api/admin-area-data/upload/json"
        # login
        login_response = requests.post(
            f"{IBF_API_URL}/api/user/login",
            data=[("email", ADMIN_LOGIN), ("password", ADMIN_PASSWORD)],
        )
        token = login_response.json()["user"]["token"]
        

        for indicator in ["IPC_forecast_short", "IPC_forecast_long"]:  # df2.columns:
            for adm_level in self.levels:  # SETTINGS[self.countryCodeISO3]["levels"]:
                df_stats = pd.DataFrame()
                df_stats["placeCode"] = ipc_df[f"ADM{adm_level}_PCODE"]
                df_stats["amount"] = ipc_df[indicator]
                df_stats_levl = df_stats.groupby("placeCode").agg({"amount": "max"})
                df_stats_levl.reset_index(inplace=True)
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

                upload_response = requests.post(
                    url,
                    json=exposure_data,
                    headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                print(upload_response)

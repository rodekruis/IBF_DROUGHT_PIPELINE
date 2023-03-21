# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:19:24 2021

@author: ATeklesadik
"""

import subprocess
import requests
import urllib.request
import os
import shutil
import geopandas as gpd
import urllib.request
from os import listdir
from os.path import isfile, join
import glob
import datetime as dt
import pandas as pd
import json
from drought_model.settings import *

try:
    from drought_model.secrets import *
except ImportError:
    print("No secrets file found.")
import logging
logger = logging.getLogger(__name__)

class HOTSPOTCLASS:
    def __init__(
        self,
        countryCodeISO3,
    ):
        self.countryCodeISO3 = countryCodeISO3
        self.levels = SETTINGS[countryCodeISO3]["levels"]
        self.PIPELINE_OUTPUT = PIPELINE_OUTPUT
        self.PIPELINE_INPUT = PIPELINE_INPUT



    ##################### hotspot
    def processing(self):
        hs_woreda_data = (
            self.PIPELINE_INPUT
            + "National level Hotspot Woreda classification Jan2021-Final.xlsx"
        )
        hs = pd.read_excel(
            hs_woreda_data, sheet_name="Hotspot classification", header=1
        )
        df2 = pd.DataFrame()
        cols = [
            "Hotspot_General",
            "Hotspot_Water",
            "Hotspot_Nutrition",
            "Hotspot_Health",
        ]
        df2[
            [
                "ADM3_PCODE",
                "Hotspot_General",
                "Hotspot_Water",
                "Hotspot_Nutrition",
                "Hotspot_Health",
            ]
        ] = hs[
            [
                "P-Code",
                "Final National  level  Hotspots",
                "Water",
                "Nutrition",
                "Health",
            ]
        ]
        df2[cols] = df2[cols].apply(pd.to_numeric, errors="coerce", axis=1)
        df2 = df2.fillna(0.1)

        admin_woreda_eth = self.PIPELINE_INPUT + "ETH_adm3.geojson"

        eth_admin = gpd.read_file(admin_woreda_eth)  # fc.admin_area_gdf

        hs_df = pd.merge(
            eth_admin,
            df2,
            how="left",
            left_on="ADM3_PCODE",
            right_on="ADM3_PCODE",
        ) 
        url = IBF_API_URL + "admin-area-data/upload/json"
        # login
        login_response = requests.post(
            f"{IBF_API_URL}user/login",
            data=[("email", ADMIN_LOGIN), ("password", ADMIN_PASSWORD)],
        )
        token = login_response.json()["user"]["token"]

        for indicator in [
            "Hotspot_General",
            "Hotspot_Water",
            "Hotspot_Nutrition",
            "Hotspot_Health",
        ]:  # df2.columns:
            for adm_level in [1,2,3]:#self.levels:  # SETTINGS[self.countryCodeISO3]["levels"]:
                df_stats = pd.DataFrame()
                df_stats["placeCode"] = hs_df[f"ADM{adm_level}_PCODE"]
                df_stats["amount"] = hs_df[indicator]
                df_stats_levl = df_stats.groupby("placeCode").agg({"amount": "max"})
                df_stats_levl.reset_index(inplace=True)
                df_stats_levl = df_stats_levl.fillna(0.1)
                df_stats_levl = df_stats_levl[[ "placeCode","amount"]].to_dict(
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
                    # "dynamicIndicator": indicator, #"exposurePlaceCodes": df_stats_levl,
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
                        "Accept": "application/json",},
                )
                
                logger.info(f'{upload_response}') 
 
            logger.info(f'Uploaded hotspot class data for indicator: {indicator}')

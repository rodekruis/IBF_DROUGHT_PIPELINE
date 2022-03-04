from drought_model.exposure import Exposure
from drought_model.GetData import GetData
from drought_model.dynamicDataDb import DatabaseManager
from drought_model.settings import *
import pandas as pd
import json
from shapely import wkb, wkt
import geopandas
import os
import logging
logger = logging.getLogger(__name__)


class Forecast:
    def __init__(self, leadTimeLabel, leadTimeValue, countryCodeISO3, admin_level):
        self.leadTimeLabel = leadTimeLabel
        self.leadTimeValue = leadTimeValue
        self.admin_level = admin_level
        self.countryCodeISO3=countryCodeISO3
        self.db = DatabaseManager(leadTimeLabel,leadTimeValue, countryCodeISO3,admin_level)      
        self.levels = SETTINGS[countryCodeISO3]['levels']

        admin_area_json = self.db.apiGetRequest('admin-areas/raw',countryCodeISO3=countryCodeISO3)

        #print(admin_area_json)
        for index in range(len(admin_area_json)):
            admin_area_json[index]['geometry'] = admin_area_json[index]['geom']
            admin_area_json[index]['properties'] = {
                'placeCode': admin_area_json[index]['placeCode'], 
                'placeCodeParent': admin_area_json[index]['placeCodeParent'],                   
                'name': admin_area_json[index]['name'],
                'adminLevel': admin_area_json[index]['adminLevel']
                }
                
        df_admin=pd.DataFrame(admin_area_json) 
        df_admin2=df_admin.filter(['adminLevel','placeCode','placeCodeParent'])
        df_list={}       
        max_iteration=self.admin_level+1
        for adm_level in self.levels:
            df_=df_admin2.query(f"adminLevel == {adm_level}")
            df_.rename(columns={"placeCode": f"placeCode_{adm_level}","placeCodeParent": f"placeCodeParent_{adm_level}"},inplace=True)            
            df_list[adm_level]=df_            
        
        df=df_list[self.admin_level]        
        ################# Create a dataframe with pcodes for each admin level         
        for adm_level in self.levels:
            j=adm_level-1
            if j >0 and len(self.levels)>1:
                df=pd.merge(df.copy(),df_list[j],  how='left',left_on=f'placeCodeParent_{j+1}' , right_on =f'placeCode_{j}')
     
        df=df[[f"placeCode_{i}" for i in self.levels]]      
        self.pcode_df=df[[f"placeCode_{i}" for i in self.levels]]      

        df_admin=df_admin.query(f'adminLevel == {self.admin_level}')
        population_df = self.db.apiGetRequest('admin-area-data/{}/{}/{}'.format(countryCodeISO3, self.admin_level, 'populationTotal'), countryCodeISO3='')
        population_df=pd.DataFrame(population_df) 

        df_admin1=geopandas.GeoDataFrame.from_features(admin_area_json)
        df_admin1=df_admin1.query(f'adminLevel == {self.admin_level}')
        self.admin_area_gdf = df_admin1

        self.population_total =population_df
        #read glofas trigger levels from file
        self.getdata = GetData(leadTimeLabel,leadTimeValue,self.admin_area_gdf,self.population_total, self.countryCodeISO3,self.admin_level)

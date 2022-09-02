"""
Drought pipeline for IBF Kenya
Author: Aklilu Teklesadik (Ateklesadik@redcross.nl)
 
"""
import pandas as pd
from shapely.geometry import Point
import fiona
import pdfplumber
import geopandas as gpd
import tabula
#from tabula import read_pdf
import re
import json
import numpy as np
from drought_model.dynamicDataDb import DatabaseManager
from drought_model.settings import *
try:
    from drought_model.secrets import *
except ImportError:
    print('No secrets file found.')
import os
import logging
logger = logging.getLogger(__name__)
import rioxarray
import rasterio as rio
from geocube.api.core import make_geocube
import rioxarray
import xarray as xr

class GetData:
    def __init__(self, leadTimeLabel, leadTimeValue,admin_area_gdf,population_total, countryCodeISO3,admin_level):
        #self.db = DatabaseManager(leadTimeLabel, countryCodeISO3)
        self.leadTimeLabel = leadTimeLabel
        self.leadTimeValue = leadTimeValue
        self.countryCodeISO3 = countryCodeISO3
        self.admin_level=admin_level
        self.ADMIN_AREA_GDF = admin_area_gdf
        self.PIPELINE_OUTPUT = PIPELINE_OUTPUT
        self.PIPELINE_INPUT = PIPELINE_INPUT
        
        
        self.TRIGGER_PROB = SETTINGS[countryCodeISO3]['TRIGGER_LEVELS']["TRIGGER_PROBABILITY"]
        self.TRIGGER_PROBABILITY_RAIN = SETTINGS[countryCodeISO3]['TRIGGER_LEVELS']["TRIGGER_rain_prob_threshold"][0] 
        self.triggger_prob=SETTINGS[countryCodeISO3]['TRIGGER_LEVELS']["TRIGGER_rain_prob_threshold"][1]
        self.SPI_Threshold_Prob=SETTINGS[countryCodeISO3]['TRIGGER_LEVELS']["SPI_Threshold_Prob"]          
 
        
    
        #self.outputPath = PIPELINE_DATA+'input/'
        
        self.spiforecast=PIPELINE_INPUT+'ond_forecast.csv'   
        self.FILE_PATH=NDRMC_BULLETIN_FILE_PATH 
        
        self.population_df=population_total
        
        self.DYNAMIC_INDICATORS= SETTINGS[countryCodeISO3]['DYNAMIC_INDICATORS']
        self.EXPOSURE_DATA_SOURCES= SETTINGS[countryCodeISO3]['EXPOSURE_DATA_SOURCES']

        self.Icpac_Forecast_FtpPath = Icpac_Forecast_FtpPath
        self.Icpac_Forecast_FilePath = Icpac_Forecast_FilePath
        self.Icpac_Forecast_FtpPath_Rain = Icpac_Forecast_FtpPath_Rain
        self.Icpac_Forecast_FilePath_Rain = Icpac_Forecast_FilePath_Rain

        self.ICPAC_FTP_ADDRESS = ICPAC_FTP_ADDRESS
        self.ICPAC_FTP_USERNAME = ICPAC_FTP_USERNAME
        self.ICPAC_FTP_PASSWORD = ICPAC_FTP_PASSWORD
        
        self.output_filepath=Icpac_Forecast_FilePath_Rain#PIPELINE_DATA+'input/'+ftp_file_path.split('/')[-1]  
        
 
        self.current_date = CURRENT_DATE.strftime('%Y%m%d')

    def processing(self):

        spi_data=self.get_spi_data()        
        #drought_indicators=self.read_bulletin()
        df=self.population_df
        
        df_spi = pd.merge(df,spi_data, how="left", on="placeCode")        
        df_spi['trigger']=df_spi['trigger'].fillna(0)
        
        df_spi['population_affected']=df_spi[['value','trigger']].apply(self.affected_people, axis="columns")
       
        return df_spi
        
    def callAllExposure(self):

        df_total = self.processing()
        drought_indicators=self.read_bulletin()
        for indicator, values in self.DYNAMIC_INDICATORS.items():
            df_stats_levl=drought_indicators[indicator]
            self.statsPath=PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
                        str(self.leadTimeValue) + '_' + self.countryCodeISO3 +'_admin_' +str(self.admin_level) + '_' + indicator + '.json'
            result = {
                'countryCodeISO3': self.countryCodeISO3,
                'exposurePlaceCodes': df_stats_levl,
                'leadTime': self.leadTimeLabel,
                'dynamicIndicator': indicator,
                'adminLevel': self.admin_level
            }
            
            with open(self.statsPath, 'w') as fp:
                json.dump(result, fp)
        
        for indicator, values in self.EXPOSURE_DATA_SOURCES.items():
            try:
                logger.info(f'indicator: {indicator}')
                df_total['amount']=df_total[indicator]                
                population_affected=df_total[['placeCode','amount']]        
                stats=population_affected.to_dict(orient='records')
                df_stats=pd.DataFrame(stats) 
                #stats_dff = pd.merge(df,self.pcode_df,  how='left',left_on='placeCode', right_on = f'placeCode_{self.admin_level}')
                for adm_level in SETTINGS[self.countryCodeISO3]['levels']:  
                    if adm_level==self.admin_level:
                        df_stats_levl=stats
                    else:
                        df_stats_levl =df_stats.groupby(f'placeCode_{adm_level}').agg({'amount': 'sum'})
                        df_stats_levl.reset_index(inplace=True)
                        df_stats_levl['placeCode']=df_stats_levl[f'placeCode_{adm_level}']
                        df_stats_levl=df_stats_levl[['amount','placeCode']].to_dict(orient='records')
                        
                    self.statsPath = PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
                        str(self.leadTimeValue) + '_' + self.countryCodeISO3 +'_admin_' +str(adm_level) + '_' + indicator + '.json'

                    result = {
                        'countryCodeISO3': self.countryCodeISO3,
                        'exposurePlaceCodes': df_stats_levl,
                        'leadTime': self.leadTimeLabel,
                        'dynamicIndicator': indicator,# + '_affected',
                        'adminLevel': adm_level
                    }
                    
                    with open(self.statsPath, 'w') as fp:
                        json.dump(result, fp)
                        
                    if indicator=='population_affected':
                        alert_threshold = list(map(self.get_alert_threshold, df_stats_levl))

                        alert_threshold_file_path = PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
                            str(self.leadTimeValue) + '_' + self.countryCodeISO3 + '_admin_' + str(adm_level) + '_' + 'alert_threshold' + '.json'

                        alert_threshold_records = {
                            'countryCodeISO3': self.countryCodeISO3,
                            'exposurePlaceCodes': alert_threshold,
                            'leadTime': self.leadTimeLabel,
                            'dynamicIndicator': 'alert_threshold',
                            'adminLevel': adm_level
                        }

                        with open(alert_threshold_file_path, 'w') as fp:
                            json.dump(alert_threshold_records, fp)
            except:
                logger.info(f'failed to output for indicator: {indicator}')
                pass


    def affected_people(self,df):
        x=df[0]
        y=df[1]
        return int(x*y)    
    def get_spi_data(self):
        """
        Rainfall forecast for spi comes in a form of tercile probability catagories
        SPI LIMIT =-0.98 and the three ctagories are 0.16354 (will be lower than -0.98),0.3006 ( will be about -.98normal)  
        and 0.53586 (will be higher than -0.98). For drought the interesting figures will be the probability with 0.16    
        """
        with open(self.spiforecast) as file:
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
                        data.append([float(temp_data[0]),float(X[j]),float(temp_data[j+1]),float(clim_prob)])       
                elif line.startswith('cpt' ):
                    cpt_items=line.strip('\n').split(',')
                    clim_prob=[items.split("=")[1] for items in cpt_items if  items.split("=")[0]==' cpt:clim_prob'][0]


        file_name= self.PIPELINE_INPUT + 'spi_forecast.csv'        
        df = pd.DataFrame(data, index =None, columns =['Lon','Lat','precipitation','clim_prob'])
        df_1=df.query('clim_prob=={}'.format(self.SPI_Threshold_Prob)) 
        geometry=[Point(xy) for xy in zip(df_1.iloc[:,1], df_1.iloc[:,0])]
        gdf=gpd.GeoDataFrame(df_1, geometry=geometry)    
        admin = self.ADMIN_AREA_GDF   #
        pointInPoly = gpd.sjoin(gdf, admin, op='within') 
        rainforecast=pointInPoly[["precipitation",'placeCode']].groupby('placeCode').mean()
        rainforecast['trigger']=rainforecast['precipitation'].apply(lambda x: 1 if x>self.TRIGGER_PROB else 0)
        rainforecast.to_csv(file_name)
        return rainforecast
        
    def process_icpac_data(self,):
        ####  admin boundary shape file 
        admin_df =self.ADMIN_AREA_GDF  
        forecast_data =rioxarray.open_rasterio(self.output_filepath)
        ### if the raster file has multiple formats select the band which is relevant for the analysis
        precipitation=forecast_data['below'].rio.write_crs("epsg:4326", inplace=True).rio.clip(admin_df.geometry.values, admin_df.crs, from_disk=True).sel(band=1).drop("band")

        precipitation.name = "precipitation"
        ### create a new unique identifier with type integer 
        admin_df['pcode'] = admin_df.apply(lambda row: row.ADM1_PCODE[-3:], axis=1)
        admin_df["pcode"] = admin_df.pcode.astype(int)

        # make your geo cube 
        data_cube = make_geocube(
            vector_data=admin_df,
            measurements=["pcode"],
            like=precipitation, # ensure the data are on the same grid
        )

        # merge the two together
        data_cube["precipitation"] = (precipitation.dims, precipitation.values, precipitation.attrs, precipitation.encoding)
        grouped_precipitation = data_cube.drop("spatial_ref").groupby(data_cube.pcode)

        grid_mean = grouped_precipitation.mean().rename({"precipitation": "amount"})
        grid_median = grouped_precipitation.median().rename({"precipitation": "precipitation_median"})
         
        zonal_stats_df = xr.merge([grid_mean, grid_median]).to_dataframe().reset_index()
         
        zonal_stats_df['placeCode'] = zonal_stats_df.apply(lambda row: self.countryCodeISO3+str(int(row.pcode)).zfill(3), axis=1) 
        zonal_stats_df=zonal_stats_df[['placeCode','amount']]        
        #stats=population_affected.to_dict(orient='records')
        zonal_stats_df['trigger']=zonal_stats_df['amount'].apply(lambda x: 1 if x>self.TRIGGER_PROBABILITY_RAIN else 0)
        file_name=self.output_filepath.split('.')[0]+'.csv'
        zonal_stats_df.to_csv(file_name)
        
        return zonal_stats_df
        
        
    def get_alert_threshold(self, population_affected):
        # population_total = next((x for x in self.population_total if x['placeCode'] == population_affected['placeCode']), None)
        alert_threshold = 0
        if (population_affected['amount'] > 0):
            alert_threshold = 1
        else:
            alert_threshold = 0
        return {
            'amount': alert_threshold,
            'placeCode': population_affected['placeCode']
        }         
        
    def read_bulletin(self):        
        dfs = tabula.read_pdf(self.FILE_PATH, pages='all',stream=True)
        dfs = [df for df in dfs if not df.empty]    
        
        vci_satus={'Extreme':1,'Severe':2,'Moderate':3,'Normal':4,'Above_normal':5}
        cattle_satus={'poor':1,'fair':2,'good':3}
        drought_satus={'normal':1,'alarm':3,'alert':2,'emergency':4,'recovery':5}   
        df_vci={}
        df_cattle={}
        df_drought={}
            
        for items in range(len(dfs)):
            df=dfs[items]
            drought_identifier=df.columns[0]+' '+ str(df.iloc[0,0])
            if all(elem in list(df.iloc[0].to_dict().keys())  for elem in ['Category', 'County'] ):
                df_c=df.copy()
                df_c.columns = df_c.iloc[0]
                df1=df_c.iloc[0: , :]
                df1['Extreme']=df.iloc[:,0]
                #df1['status']=df1['Extreme'].fillna(method="ffill")
                df1['status'] = df1['Extreme'].map({'Extreme': 'Extreme',
                                                         'Severe vegetation':'Severe',
                                                         'Severe':'Severe',
                                                         'Moderate vegetation': 'Moderate',
                                                         'Moderate':'Moderate',
                                                         'Normal vegetation':'Normal',
                                                         'Normal':'Normal',
                                                         'normal':'Normal',
                                                         'Above normal':'Above_normal',
                                                         'Above':'Above_normal'}) 
                df1['status']=df1['status'].fillna(method="ffill")
                
                df1.iloc[:,2].replace('\(',',',inplace=True,regex=True)
                df1.iloc[:,2].replace('\)',',',inplace=True,regex=True)
                        
        
                t0=list(set(df1.query('status=="Extreme"').iloc[:, 2].dropna().values.flatten()))
                t = [items.split(',') for items in t0]
                df_vci['Extreme'] = [re.sub(r"^\s+|\s+$", "", item) for sublist in t for item in sublist]
                
                t0=list(set(df1.query('status=="Severe"').iloc[:, 2].dropna().values.flatten()))
                t = [items.split(',') for items in t0]
                df_vci['Severe']= [re.sub(r"^\s+|\s+$", "", item) for sublist in t for item in sublist]
                
                t0=list(set(df1.query('status=="Moderate"').iloc[:, 2].dropna().values.flatten()))
                t = [items.split(',') for items in t0]
                df_vci['Moderate']= [re.sub(r"^\s+|\s+$", "", item) for sublist in t for item in sublist]
                
                t0=list(set(df1.query('status=="Normal"').iloc[:, 2].dropna().values.flatten()))
                t = [items.split(',') for items in t0]
                df_vci['Normal']= [re.sub(r"^\s+|\s+$", "", item) for sublist in t for item in sublist]
                
                t0=list(set(df1.query('status=="Above_normal"').iloc[:, 2].dropna().values.flatten()))
                t = [items.split(',') for items in t0]
                df_vci['Above_normal']= [re.sub(r"^\s+|\s+$", "", item) for sublist in t for item in sublist]
                print("yes",items,df_vci)
            elif all(elem in list(df.iloc[0].to_dict().keys())  for elem in  ['Cattle','Goats'] ):
                df_c=df.copy()
                df_c.columns = df_c.iloc[0]
                df_c=df_c.iloc[1: , :]
                
                df_cattle['fair']=list(set(df_c['Fair'].dropna().values.flatten()))
                df_cattle['poor']=list(set(df_c['Poor'].dropna().values.flatten()))
                df_cattle['good']=list(set(df_c['Good'].dropna().values.flatten()))
                print("yes",items,df_cattle)
            elif  drought_identifier in ['Drought status', 'Drought status nan']:
                #df.columns = df.iloc[0]
                df_c=df.copy()
                df_c=df_c.iloc[1: , :]
                #df['status']=df.iloc[:,0]
                #df['Worsening']=df.iloc[:,3]
                df_c['Dr_status']=df_c.iloc[:,0].fillna(method="ffill")
                
                df_drought['normal']=list(set(df_c.query('Dr_status=="Normal"').iloc[:, 4].dropna().values.flatten()))
                df_drought['alarm']=list(set(df_c.query('Dr_status=="Alarm"').iloc[:, 4].dropna().values.flatten()))
                df_drought['alert']=list(set(df_c.query('Dr_status=="Alert"').iloc[:, 4].dropna().values.flatten()))
                df_drought['emergency']=list(set(df_c.query('Dr_status=="Emergency"').iloc[:, 4].dropna().values.flatten()))
                df_drought['recovery']=list(set(df_c.query('Dr_status=="Recovery"').iloc[:, 4].dropna().values.flatten()))
                
                print("yes",items,df_drought)
            else:
                print('no data')
        df_total = {**df_vci, **df_cattle, **df_drought}            
          
        bulletin_updated={}
        
        for k,v in df_total.items():
            v=[item for item in v if len(item)>3]
            bulletin_updated[k]=v

        df=self.ADMIN_AREA_GDF[['name','placeCode']]
      
        # join extracted data for the three indicators with admin layer 
        dfvci={}
        columns_withvalue=[]
        for status in vci_satus.keys():
            print(vci_satus[status])
            if bulletin_updated[status] != []:
                df2=pd.DataFrame(bulletin_updated[status])
                df2.columns=["name"]
                df2[status]=vci_satus[status]
                df = pd.merge(df, df2, how="left", on="name")
                df=df.fillna(0)
                columns_withvalue.append(status)
                
        df['amount'] = df[columns_withvalue].apply(np.nanmax, axis = 1)        
        df=df.groupby('placeCode').agg({'amount': 'max'})
        df = df.astype(int)
        df.reset_index(inplace=True) 
        df=df.query('amount > 0')        
        dfvci['vegetation_condition']=df[['placeCode','amount']].to_dict(orient='records')
        
        df=self.ADMIN_AREA_GDF[['name','placeCode']]
        columns_withvalue=[]
        
        for status in cattle_satus.keys():
            print(cattle_satus[status])
            if bulletin_updated[status] != []:
                df2=pd.DataFrame(bulletin_updated[status])
                df2.columns=["name"]
                df2[status]=cattle_satus[status]
                df = pd.merge(df, df2, how="left", on="name")
                df=df.fillna(0)
                columns_withvalue.append(status)
            
        df['amount'] = df[columns_withvalue].apply(np.nanmax, axis = 1)
        df=df.groupby('placeCode').agg({'amount': 'max'})
        df = df.astype(int)
        df.reset_index(inplace=True) 
        df=df.query('amount > 0')
        dfvci['livestock_body_condition']=df[['placeCode','amount']].to_dict(orient='records')       

        df=self.ADMIN_AREA_GDF[['name','placeCode']]
        columns_withvalue=[]
        for status in drought_satus.keys():
            print(drought_satus[status])
            if bulletin_updated[status] != []:
                df2=pd.DataFrame(bulletin_updated[status])
                df2.columns=["name"]
                df2[status]=drought_satus[status]
                df = pd.merge(df, df2, how="left", on="name") 
                df=df.fillna(0)
                columns_withvalue.append(status)                    
        df['amount'] = df[columns_withvalue].apply(np.nanmax, axis = 1)
        df=df.groupby('placeCode').agg({'amount': 'max'})
        df = df.astype(int)
        df.reset_index(inplace=True)    
        df=df.query('amount > 0')            
        dfvci['drought_phase_classification']=df[['placeCode','amount']].to_dict(orient='records')

        print('pass')

        
        # remove duplicate entries
        indicator_file_path = PIPELINE_OUTPUT + 'calculated_affected/dynamic_drought_indicators.json'
        with open(indicator_file_path, 'w') as fp:
            json.dump(dfvci, fp)  
        
        df_bulletin= dfvci#.groupby('placeCode').agg({'VCI_Status':np.nanmax,'Cattle_Status':np.nanmax,'Drought_Status':np.nanmax}).fillna(0)
        return  df_bulletin
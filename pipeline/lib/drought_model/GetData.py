"""
Author: Aklilu Teklesadik (Ateklesadik@redcross.nl)
 
"""
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
from drought_model.dynamicDataDb import DatabaseManager
from drought_model.settings import *
try:
    from drought_model.secrets import *
except ImportError:
    print('No secrets file found.')
import os
import logging
logger = logging.getLogger(__name__)


class GetData:
    def __init__(self, leadTimeLabel, leadTimeValue,admin_area_gdf,population_total, countryCodeISO3,admin_level):
        #self.db = DatabaseManager(leadTimeLabel, countryCodeISO3)
        self.leadTimeLabel = leadTimeLabel
        self.leadTimeValue = leadTimeValue
        self.countryCodeISO3 = countryCodeISO3
        self.admin_level=admin_level
        self.inputPath = PIPELINE_DATA+'input/'
        self.TRIGGER_PROB=TRIGGER_PROBABILITY
        self.outputPath = PIPELINE_DATA+'input/'
        self.spiforecast=PIPELINE_DATA+'input/ond_forecast.csv'
        self.triggger_prob=TRIGGER_PROBABILITY
        self.ADMIN_AREA_GDF = admin_area_gdf
        self.FILE_PATH=NDRMC_BULLETIN_FILE_PATH
        self.SPI_Threshold_Prob=SPI_Threshold_Prob
        self.population_df=population_total
        self.DYNAMIC_INDICATORS= SETTINGS[countryCodeISO3]['DYNAMIC_INDICATORS']
        
        if not os.path.exists(self.inputPath):
            os.makedirs(self.inputPath)
        if not os.path.exists(self.inputPath):
            os.makedirs(self.inputPath)
        self.current_date = CURRENT_DATE.strftime('%Y%m%d')

    def processing(self):
        spi_data=self.get_spi_data()
        drought_indicators=self.read_bulletin()
        df=self.population_df
        df_spi = pd.merge(df,spi_data, how="left", on="placeCode")        
        df_total = pd.merge(df_spi,drought_indicators,how='left',left_on='placeCode', right_on = 'placeCode')
        df_total['trigger']=df_total['trigger'].fillna(0)
        df_total['population']=df_total[['value','trigger']].apply(self.affected_people, axis="columns")
       
        return df_total
        
    def callAllExposure(self):
        df_total = self.processing()
        for indicator, values in self.DYNAMIC_INDICATORS.items():
            try:
                logger.info(f'indicator: {indicator}')
                df_total['amount']=df_total[values]                
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
                        self.leadTimeLabel + '_' + self.countryCodeISO3 +'_admin_' +str(adm_level) + '_' + indicator + '.json'

                    result = {
                        'countryCodeISO3': self.countryCodeISO3,
                        'exposurePlaceCodes': df_stats_levl,
                        'leadTime': self.leadTimeLabel,
                        'dynamicIndicator': indicator,# + '_affected',
                        'disasterType':'drought',
                        'adminLevel': adm_level
                    }
                    
                    with open(self.statsPath, 'w') as fp:
                        json.dump(result, fp)
                        
                    if indicator=='population_affected':
                        alert_threshold = list(map(self.get_alert_threshold, df_stats_levl))

                        alert_threshold_file_path = PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
                            self.leadTimeLabel + '_' + self.countryCodeISO3 + '_admin_' + str(adm_level) + '_' + 'alert_threshold' + '.json'

                        alert_threshold_records = {
                            'countryCodeISO3': self.countryCodeISO3,
                            'exposurePlaceCodes': alert_threshold,
                            'leadTime': self.leadTimeLabel,
                            'dynamicIndicator': 'alert_threshold',
                            'disasterType':'drought',
                            'adminLevel': adm_level
                        }

                        with open(alert_threshold_file_path, 'w') as fp:
                            json.dump(alert_threshold_records, fp)
            except:
                logger.info(f'failed to output for indicator: {indicator}')
                pass

            # if self.population_total:
                # get_population_affected_percentage_ = functools.partial(self.get_population_affected_percentage, adm_level=adm_level)
                # population_affected_percentage = list(map(get_population_affected_percentage_, df_stats_levl))
                # #population_affected_percentage = list(map(self.get_population_affected_percentage, df_stats,adm_level))
 
                # population_affected_percentage_file_path = PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
                    # self.leadTimeLabel + '_' + self.countryCodeISO3 + '_admin_' + str(adm_level) + '_' + 'population_affected_percentage' + '.json'
                    
                # population_affected_percentage_records = {
                    # 'countryCodeISO3': self.countryCodeISO3,
                    # 'exposurePlaceCodes': population_affected_percentage, 
                    # 'leadTime': self.leadTimeLabel,
                    # 'dynamicIndicator': 'population_affected_percentage',
                    # 'adminLevel': adm_level
                # }

                # with open(population_affected_percentage_file_path, 'w') as fp:
                    # json.dump(population_affected_percentage_records, fp)

                # define alert_threshold layer
            # alert_threshold = list(map(self.get_alert_threshold, df_stats_levl))

            # alert_threshold_file_path = PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
                # self.leadTimeLabel + '_' + self.countryCodeISO3 + '_admin_' + str(adm_level) + '_' + 'alert_threshold' + '.json'

            # alert_threshold_records = {
                # 'countryCodeISO3': self.countryCodeISO3,
                # 'exposurePlaceCodes': alert_threshold,
                # 'leadTime': self.leadTimeLabel,
                # 'dynamicIndicator': 'alert_threshold',
                # 'adminLevel': adm_level
            # }

            # with open(alert_threshold_file_path, 'w') as fp:
                # json.dump(alert_threshold_records, fp)
        
    def affected_people(self,df):
        x=df[0]
        y=df[1]
        return x*y    
    def get_spi_data(self):
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


        file_name= self.outputPath + '/' + 'spi_forecast.csv'        
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
        
        vci_satus={'Extreme':5,'Severe':4,'Moderate':3,'Normal':2,'Above_normal':1}
        cattle_satus={'poor':3,'fair':2,'good':1}
        drought_satus={'normal':1,'alarm':2,'alert':3}   
        df_vci={}
        df_cattle={}
        df_drought={}
            
        for items in range(len(dfs)):
            df=dfs[items]
            drought_identifier=df.columns[0]+' '+ str(df.iloc[0,0])
            if all(elem in list(df.iloc[0].to_dict().keys())  for elem in ['Category', 'County'] ):
                df.columns = df.iloc[0]
                df1=df.iloc[0: , :]
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
                df.columns = df.iloc[0]
                df=df.iloc[1: , :]
                
                df_cattle['fair']=list(set(df['Fair'].dropna().values.flatten()))
                df_cattle['poor']=list(set(df['Poor'].dropna().values.flatten()))
                df_cattle['good']=list(set(df['Good'].dropna().values.flatten()))
                print("yes",items,df_cattle)
            elif  drought_identifier in ['Drought status', 'Drought status nan']:
                #df.columns = df.iloc[0]
                df=df.iloc[1: , :]
                #df['status']=df.iloc[:,0]
                #df['Worsening']=df.iloc[:,3]
                df['Dr_status']=df.iloc[:,0].fillna(method="ffill")
                
                df_drought['normal']=list(set(df.query('Dr_status=="Normal"').iloc[:, 4].dropna().values.flatten()))
                df_drought['alarm']=list(set(df.query('Dr_status=="Alarm"').iloc[:, 4].dropna().values.flatten()))
                df_drought['alert']=list(set(df.query('Dr_status=="Alert"').iloc[:, 4].dropna().values.flatten()))
                print("yes",items,df_drought)
            else:
                print('no data')
        df_total = {**df_vci, **df_cattle, **df_drought}            
        indicator_file_path = PIPELINE_OUTPUT + 'calculated_affected/bulletin_df.json'
        bulletin_updated={}
        for k,v in df_total.items():
            v=[item for item in v if len(item)>3]
            bulletin_updated[k]=v
 

        with open(indicator_file_path, 'w') as fp:
            json.dump(bulletin_updated, fp)       
            
        
        
        df=self.ADMIN_AREA_GDF[['name','placeCode']]
        # join extracted data for the three indicators with admin layer 
        for status in vci_satus.keys():
            print(vci_satus[status])
            df2=pd.DataFrame(df_vci[status])
            df2.columns=["name"]
            df2[status]=vci_satus[status]
            df = pd.merge(df, df2, how="left", on="name")
        df['vegetation_condition'] = df[['Extreme','Severe','Moderate','Normal','Above_normal']].apply(np.nanmax, axis = 1)
        
        df=df[['name','placeCode','vegetation_condition']]
        
        for status in cattle_satus.keys():
            print(cattle_satus[status])
            df2=pd.DataFrame(df_cattle[status])
            df2.columns=["name"]
            df2[status]=cattle_satus[status]
            df = pd.merge(df, df2, how="left", on="name")
            
        df['livestock_condition'] = df[['poor','fair','good']].apply(np.nanmax, axis = 1)
        
        df=df[['name','placeCode','vegetation_condition','livestock_condition']]  
        try:
            for status in drought_satus.keys():
                print(drought_satus[status])
                df2=pd.DataFrame(df_drought[status])
                df2.columns=["name"]
                df2[status]=drought_satus[status]
                df = pd.merge(df, df2, how="left", on="name")  
            df['drought_phase'] = df[['normal','alarm','alert']].apply(np.nanmax, axis = 1)
            df=df[['name','placeCode','vegetation_condition','livestock_condition','drought_phase']] 
            print('pass')
        except:
            print('failed')
        
        # remove duplicate entries
        
        df_bulletin= df#.groupby('placeCode').agg({'VCI_Status':np.nanmax,'Cattle_Status':np.nanmax,'Drought_Status':np.nanmax}).fillna(0)
        return  df_bulletin
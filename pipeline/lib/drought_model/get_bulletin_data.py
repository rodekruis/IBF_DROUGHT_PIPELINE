# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:52:07 2022

@author: ATeklesadik
"""
 
import tabula
from tabula import read_pdf
import re
import pandas as pd
import geopandas as gpd 
import numpy as np


# Read remote pdf  
file_path="https://www.ndma.go.ke/index.php/resource-center/national-drought-bulletin/send/39-drought-updates/6312-national-monthly-drought-updates-january-2022"
admin = gpd.read_file('input/ken_admin1.geojson')  

def read_bulletin(file_path,admin):        
    dfs = tabula.read_pdf(file_path, pages='all',stream=True)
    dfs = [df for df in dfs if not df.empty]    
    
    vci_satus={'Extreme':5,'Severe':4,'Moderate':3,'Normal':2,'Above_normal':1}
    cattle_satus={'Poor':3,'fair':2,'good':1}
    drought_satus={'Normal':1,'Alarm':2,'Alert':3,}  
        
    for items in range(len(dfs)):
        df=dfs[items]
        if all(elem in list(df.iloc[0].to_dict().keys())  for elem in ['Category', 'County'] ):
            df.columns = df.iloc[0]
            df1=df.iloc[0: , :]
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
            df_vci={}
            df1.iloc[:,2].replace('\(',',',inplace=True,regex=True)
            df1.iloc[:,2].replace('\)',',',inplace=True,regex=True)
                    
    
            t0=list(set(df1.query('status=="Extreme"').iloc[:, 2].dropna().values.flatten()))
            t = [items.split(',') for items in t0]
            df_vci['Extreme'] = [re.sub(r"^\s+", "", item) for sublist in t for item in sublist]
            
            t0=list(set(df1.query('status=="Severe"').iloc[:, 2].dropna().values.flatten()))
            t = [items.split(',') for items in t0]
            df_vci['Severe']= [re.sub(r"^\s+", "", item) for sublist in t for item in sublist]
            
            t0=list(set(df1.query('status=="Moderate"').iloc[:, 2].dropna().values.flatten()))
            t = [items.split(',') for items in t0]
            df_vci['Moderate']= [re.sub(r"^\s+", "", item) for sublist in t for item in sublist]
            
            t0=list(set(df1.query('status=="Normal"').iloc[:, 2].dropna().values.flatten()))
            t = [items.split(',') for items in t0]
            df_vci['Normal']= [re.sub(r"^\s+", "", item) for sublist in t for item in sublist]
            
            t0=list(set(df1.query('status=="Above_normal"').iloc[:, 2].dropna().values.flatten()))
            t = [items.split(',') for items in t0]
            df_vci['Above_normal']= [re.sub(r"^\s+", "", item) for sublist in t for item in sublist]
            print("yes",items,df1)
        elif all(elem in list(df.iloc[0].to_dict().keys())  for elem in  ['Cattle','Goats'] ):
            df.columns = df.iloc[0]
            df=df.iloc[1: , :]
            df_cattle={}
            df_cattle['fair']=list(set(df['Fair'].dropna().values.flatten()))
            df_cattle['poor']=list(set(df['Fair'].dropna().values.flatten()))
            df_cattle['good']=list(set(df['Good'].dropna().values.flatten()))
            print("yes",items,df_cattle)
        elif all(elem in list(df.iloc[0].to_dict().keys())  for elem in ['Drought', 'Trend'] ):
            df.columns = df.iloc[0]
            df=df.iloc[1: , :]
            df['Dr_status']=df['status'].fillna(method="ffill")
            df_drought={}
            df_drought['Normal']=list(set(df.query('Dr_status=="Normal"')['Worsening'].dropna().values.flatten()))
            df_drought['Alarm']=list(set(df.query('Dr_status=="Normal"')['Worsening'].dropna().values.flatten()))
            df_drought['Alert']=list(set(df.query('Dr_status=="Alert"')['Worsening'].dropna().values.flatten()))
            print("yes",items,df_drought)
        else:
            print('no data')          
            
    df=admin[['ADM1_EN','ADM1_PCODE']]
    # join extracted data for the three indicators with admin layer 
    for status in vci_satus.keys():
        print(vci_satus[status])
        df2=pd.DataFrame(df_vci[status])
        df2.columns=["ADM1_EN"]
        df2[status]=vci_satus[status]
        df = pd.merge(df, df2, how="left", on="ADM1_EN")
    df['VCI_Status'] = df[['Extreme','Severe','Moderate','Normal','Above_normal']].apply(np.nanmax, axis = 1)
    
    df=df[['ADM1_EN','ADM1_PCODE','VCI_Status']]
    
    for status in cattle_satus.keys():
        print(cattle_satus[status])
        df2=pd.DataFrame(df_cattle[status])
        df2.columns=["ADM1_EN"]
        df2[status]=cattle_satus[status]
        df = pd.merge(df, df2, how="left", on="ADM1_EN")
        
    df['Cattle_Status'] = df[['poor','fair','good']].apply(np.nanmax, axis = 1)
    
    df=df[['ADM1_EN','ADM1_PCODE','VCI_Status','Cattle_Status']]  
     
    for status in drought_satus.keys():
        print(drought_satus[status])
        df2=pd.DataFrame(df_drought[status])
        df2.columns=["ADM1_EN"]
        df2[status]=drought_satus[status]
        df = pd.merge(df, df2, how="left", on="ADM1_EN")        
    df['Drought_Status'] = df[['Normal','Alarm','Alert']].apply(np.nanmax, axis = 1)
    
    # remove duplicate entries
    
    df_bulletin= df.groupby('ADM1_PCODE').agg({'VCI_Status':np.nanmax,
                               'Cattle_Status':np.nanmax,
                               'Drought_Status':np.nanmax}).fillna(0)
    return  df_bulletin


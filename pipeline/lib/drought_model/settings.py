##################
## LOAD SECRETS ##
##################
from datetime import date,timedelta
from dateutil.relativedelta import relativedelta
import ast


# 0. Try to load secrets from a local env-variables 
 
try:
    ADMIN_LOGIN = os.getenv("ADMIN_LOGIN")
    IBF_PASSWORD=os.getenv("IBF_PASSWORD")
    IBF_URL =os.getenv("IBF_URL")
    COUNTRY_CODES_ = ast.literal_eval(os.getenv("COUNTRY_CODES_LIST"))
    ICPAC_FTP_ADDRESS =os.getenv("ICPAC_FTP_ADDRESS")
    ICPAC_FTP_USERNAME =os.getenv("ICPAC_FTP_USERNAME")
    ICPAC_FTP_PASSWORD =os.getenv("ICPAC_FTP_PASSWORD")
    DATALAKE_STORAGE_ACCOUNT_NAME =os.getenv("DATALAKE_STORAGE_ACCOUNT_NAME")
    DATALAKE_STORAGE_ACCOUNT_KEY =os.getenv("DATALAKE_STORAGE_ACCOUNT_KEY")
    DATALAKE_API_VERSION =os.getenv("DATALAKE_API_VERSION")  

except:
     print('No environment variables found localy.')
     
# 1. Try to load secrets from Azure key vault (i.e. when running through Logic App) if user has access


try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    az_credential = DefaultAzureCredential()
    secret_client = SecretClient(
        vault_url="https://ibf-flood-keys.vault.azure.net", credential=az_credential
    )

    ADMIN_LOGIN = secret_client.get_secret("ADMIN-LOGIN").value    
    IBF_URL=secret_client.get_secret("IBF-URL").value    
    IBF_PASSWORD=secret_client.get_secret("IBF-PASSWORD").value  
    
    DATALAKE_STORAGE_ACCOUNT_NAME = secret_client.get_secret("DATALAKE-STORAGE-ACCOUNT-NAME").value
    DATALAKE_STORAGE_ACCOUNT_KEY = secret_client.get_secret("DATALAKE-STORAGE-ACCOUNT-KEY").value
    DATALAKE_API_VERSION = '2018-11-09'    

except Exception as e:
    print("No access to Azure Key vault, skipping.")

#2. Try to load secrets from env-variables (i.e. when using Github Actions)
try:
    import os

    ADMIN_LOGIN = os.environ["ADMIN_LOGIN"]
    GOOGLE_DRIVE_DATA_URL = os.environ["GOOGLE_DRIVE_DATA_URL"]
    IBF_URL=os.environ['IBF_API_URL']
    IBF_PASSWORD=os.environ['IBF_PASSWORD']
    DATALAKE_STORAGE_ACCOUNT_NAME = os.environ["DATALAKE_STORAGE_ACCOUNT_NAME"]
    DATALAKE_STORAGE_ACCOUNT_KEY_ = os.environ["DATALAKE_STORAGE_ACCOUNT_KEY"]
    DATALAKE_STORAGE_ACCOUNT_KEY=f'{DATALAKE_STORAGE_ACCOUNT_KEY_}=='
    DATALAKE_API_VERSION = '2018-11-09'

except Exception as e:
   print("No environment variables found.")

# 3. If 1. and 2. both fail, then assume secrets are loaded via secrets.py file (when running locally). If neither of the 3 options apply, this script will fail.
try:
    from drought_model.secrets import *
except ImportError:
    print("No secrets file found.")


######################
## COUNTRY SETTINGS ##
######################

# Countries to include

COUNTRY_CODES = {
    #'KEN':{'seasons':[1]},
    #'ETH': {'seasons': [2, 3, 4]},
    'UGA':{'seasons':[1,2]},
    }


#CROPPING_ZONES ethiopia = [2, 3, 4]
# Meher     1
# Belg      2
# Southern  4
# Northern  3

SETTINGS = {
    "ETH": {
        "IBF_API_URL": IBF_URL,
        "PASSWORD": IBF_PASSWORD,
        "ADMIN_LOGIN": ADMIN_LOGIN,
        "ipcCountrCode":'ET',
        "SPI_Threshold_Prob" : 'NA',
        "TRIGGER_LEVELS":
            {
            "TRIGGER_PROBABILITY": 40,
            "TRIGGER_SCENARIO": "trigger_treshold_one",
            "TRIGGER_rain_prob_threshold": 45,
            "SPI_Threshold_Prob" : 'NA',
            "quntileThreshold": 0.33,
            "TRIGGER_PROBABILITY_AriaRatio":0.3,           
            },
        "mock": False,
        "if_mock_trigger": False,
        "notify_email": False,
        "lead_times": {
            2: {
                10: ["4-month"],11: ["3-month"],12: ["2-month"],
                1: ["1-month"],2: ["0-month"],
                3: ["0-month", "3-month"],
                4: ["0-month", "2-month"],
                5: ["0-month", "1-month"],
                6: ["0-month"],7: ["0-month"],
                8: ["0-month"],9: ["0-month"],
            },
            3: {
                10: ["5-month"],11: ["4-month"],
                12: ["3-month"],1: ["2-month"],
                2: ["1-month"],3: ["0-month"],
                4: ["0-month", "3-month"],
                5: ["0-month", "2-month"],
                6: ["1-month"],7: ["0-month"],
                8: ["0-month"],9: ["0-month"],
            },
            4: {
                10: ["0-month"],11: ["0-month"],
                12: ["0-month", "3-month"],
                1: ["2-month"],2: ["1-month"],
                3: ["0-month"],4: ["0-month"],
                5: ["0-month"],6: ["0-month"],7: ["3-month"],
                8: ["2-month"],9: ["1-month"],
            },
        },
        "admin_level": 2,
        "pcodeLength": 4,
        "levels": [2],
        "admin_zones":"eth_admin2.geojson",
        "croping_zones_pcode":"eth_croping_zones_pcode.csv",
        "EXPOSURE_DATA_SOURCES": {
            "population_affected": {
                "source": "population/hrsl_eth_pop_resized_100",
                "rasterValue": 1,
            }
        },
        "DYNAMIC_INDICATORS": {
            "ipc_class": "ipc_class",
            "nutrition_need_priority_class": "nutrition_need_priority_class",
            "rainfall_forecast": "rainfall_forecast",
        },
    },
    "UGA": {
        "IBF_API_URL": IBF_URL,
        "PASSWORD": IBF_PASSWORD, 
        "ADMIN_LOGIN": ADMIN_LOGIN,
        "SPI_Threshold_Prob" : 'NA',
        "ipcCountrCode":'UG',
        "TRIGGER_LEVELS": {               
            "TRIGGER_PROBABILITY": 1,
            "TRIGGER_SCENARIO": "trigger_treshold_one", 
            "TRIGGER_rain_prob_threshold": 30, 
            "SPI_Threshold_Prob" : 'NA',
            "quntileThreshold": 0.33,
            "TRIGGER_PROBABILITY_AriaRatio":0.1, 
            },
            
        "mock": False,
        "if_mock_trigger": False,
        "notify_email": False,
        "lead_times": {
            1: {
                1: ["3-month"],2: ["2-month"],
                3: ["1-month"],4: ["0-month"],
                5: ["0-month"],6: ["0-month"],
                7: ["0-month"],8: ["0-month"],
                9: ["0-month"],10: ["0-month"],
                11: ["5-month"],12: ["4-month"],
            },
            2: {
                1: ["2-month"],
                2: ["1-month"],
                3: ["0-month"],
                4: ["0-month"],
                5: ["0-month" ],
                6: ["3-month"],
                7: ["2-month"],
                8: ["1-month"],
                9: ["0-month"],
                10: ["0-month"],
                11: ["0-month",],#5
                12: ["3-month"],#4
            },
            },
        "admin_level": 2,
        "pcodeLength": 6,
        "levels": [2],
        "admin_zones":"uga_admin2.geojson",
        "croping_zones_pcode":"uga_croping_zones_pcode.csv",
        "EXPOSURE_DATA_SOURCES": {
            "population_affected": {
                "source": "population/hrsl_uga_pop_resized_100",
                "rasterValue": 1,
            }
        },
        "DYNAMIC_INDICATORS": {
            "livestock_body_condition": "livestock_body_condition",
            "vegetation_condition": "vegetation_condition",
            "drought_phase_classification": "drought_phase_classification",
        },
    },    
    "ZMB": {
        "IBF_API_URL": IBF_URL,
        "PASSWORD": IBF_PASSWORD,
        "ADMIN_LOGIN": ADMIN_LOGIN,  
        "ipcCountrCode":'ZM',    
        "TRIGGER_LEVELS":
            {
            "TRIGGER_PROBABILITY": 40,
            "TRIGGER_SCENARIO": "trigger_treshold_one",
            "TRIGGER_rain_prob_threshold": 45,
            "SPI_Threshold_Prob" : '0.16354',     
            "quntileThreshold": 0.33,
            "TRIGGER_PROBABILITY_AriaRatio":0.3,
            },            
        "mock": False,
        "if_mock_trigger": False,
        "notify_email": False,
        "lead_times": {
            1: {
                1: ["0-month"],2: ["0-month"],
                3: ["0-month"],4: ["0-month"],
                5: ["6-month"],6: ["5-month"],
                7: ["4-month"],8: ["3-month"],
                9: ["2-month"],10: ["1-month"],
                11: ["0-month"],12: ["0-month"],
            },
        },
        "admin_level": 1,
        "levels": [1],
        "admin_zones":"zmb_admin1.geojson",
        "croping_zones_pcode":"zmb_croping_zones_pcode.csv",
        "EXPOSURE_DATA_SOURCES": {
            "population_affected": {
                "source": "population/hrsl_ken_pop_resized_100",
                "rasterValue": 1,
            }
        },
        "DYNAMIC_INDICATORS": {
            "livestock_body_condition": "livestock_body_condition",
            "vegetation_condition": "vegetation_condition",
            "drought_phase_classification": "drought_phase_classification",
        },
    },
    "KEN": {
        "IBF_API_URL": IBF_URL,
        "PASSWORD": IBF_PASSWORD,
        "ADMIN_LOGIN": ADMIN_LOGIN,
        "ipcCountrCode":'KE',
        "TRIGGER_PROBABILITY": 40,
        "TRIGGER_SCENARIO": "trigger_treshold_one",
        #"TRIGGER_threshold": [-1, 30],          
        "TRIGGER_rain_prob_threshold": [40, 10],
        "SPI_Threshold_Prob" : '0.16354',  
        "TRIGGER_LEVELS":
            {
            "TRIGGER_PROBABILITY": 40,
            "TRIGGER_SCENARIO": "trigger_treshold_one",#"trigger_treshold_one",
            #"TRIGGER_threshold": [-1, 30],
            "TRIGGER_rain_prob_threshold": 45,#[40, 10],
            "SPI_Threshold_Prob" : '0.16354',     
            "quntileThreshold": 0.33,
            "TRIGGER_PROBABILITY_AriaRatio":0.3,
            #"TRIGGER_rain_prob_threshold_percentage":0,
            },
            
        "mock": False,
        "if_mock_trigger": False,
        "notify_email": False,
        "lead_times": {
            1: {
                1: ["1-month"],2: ["0-month"],
                3: ["0-month"],4: ["0-month"],
                5: ["4-month"],6: ["3-month"],
                7: ["2-month"],8: ["1-month"],
                9: ["0-month"],10: ["0-month"],
                11: ["0-month"],12: ["2-month"],
            },
        },
        "admin_level": 1,
        "levels": [1],
        "admin_zones":"ken_admin1.geojson",
        "croping_zones_pcode":"ken_croping_zones_pcode.csv",
        "EXPOSURE_DATA_SOURCES": {
            "population_affected": {
                "source": "population/hrsl_ken_pop_resized_100",
                "rasterValue": 1,
            }
        },
        "DYNAMIC_INDICATORS": {
            "livestock_body_condition": "livestock_body_condition",
            "vegetation_condition": "vegetation_condition",
            "drought_phase_classification": "drought_phase_classification",
        },
    },

}


# Change this date only in case of specific testing purposes


#CURRENT_DATE = date.today()

CURRENT_DATE=date.today() #- timedelta(15) # to use last month forecast
CURRENT_Month = CURRENT_DATE.month
now_month = CURRENT_DATE + relativedelta(months=-1)
Now_Month_nummeric = now_month.strftime("%m")
CURRENT_Year = CURRENT_DATE.year

Debug_flag=True

ecmwfResoluation=0.5
ecmwfLeadTime='120 days'

ecmwfLeadTimeValueList = {
    "0-month":'30 days',
    "1-month":'30 days',
    "2-month":'60 days',
    "3-month":'90 days',
    "4-month":'120 days'
    }


 

'''
### define file path for ICPAC forecast
now_month = CURRENT_DATE + relativedelta(months=-1)
one_months = CURRENT_DATE + relativedelta(months=+0)
three_months = CURRENT_DATE + relativedelta(months=+2)

One_Month = one_months.strftime("%b")
Three_Month = three_months.strftime("%b")
Now_Month = now_month.strftime("%b")
Now_Month_nummeric = now_month.strftime("%m")

CURRENT_Year = date.today().year

file_name = f"{One_Month}-{Three_Month}_{Now_Month}{CURRENT_Year}" #file_name = 'Jun-Aug_May2022'


year_month = f"{CURRENT_Year}{Now_Month_nummeric}"

'''



####################
## OTHER SETTINGS ##
####################

TRIGGER_LEVELS = {"minimum": 0.6, "medium": 0.7, "maximum": 0.8}


###################
## PATH SETTINGS ##
###################
#MAIN_DIRECTORY='/home/fbf/'

MAIN_DIRECTORY='./'#C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/IBF_DROUGHT_PIPELINE/pipeline/'



RASTER_DATA = MAIN_DIRECTORY+"data/raster/"
RASTER_INPUT = RASTER_DATA + "input/"
RASTER_OUTPUT = RASTER_DATA + "output/"
PIPELINE_DATA = MAIN_DIRECTORY+"data/other/"
PIPELINE_INPUT = PIPELINE_DATA + "input/"
PIPELINE_OUTPUT = PIPELINE_DATA + "output/"
 

 

 

#########################
## INPUT DATA SETTINGS ##
#########################

# set this true if KMD forecast is uploaded in input folder 

KMD_FORECAST=False

NDRMC_BULLETIN_FILE_PATH = "https://www.ndma.go.ke/index.php/resource-center/national-drought-bulletin/send/39-drought-updates/6599-national-monthly-drought-bulletin-august-2022"

 
#####################
## ATTRIBUTE NAMES ##
#####################

TRIGGER_LEVEL = "triggerLevel"
LEAD_TIME = "leadTime"


##################
## LOAD SECRETS ##
##################

# 1. Try to load secrets from Azure key vault (i.e. when running through Logic App) if user has access
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    az_credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url='https://ibf-flood-keys.vault.azure.net', credential=az_credential)

    ADMIN_LOGIN = secret_client.get_secret("ADMIN-LOGIN").value
    GLOFAS_USER = secret_client.get_secret("GLOFAS-USER").value
    GLOFAS_PW = secret_client.get_secret("GLOFAS-PW").value
    GOOGLE_DRIVE_DATA_URL = secret_client.get_secret("GOOGLE-DRIVE-DATA-URL").value
    UGA_URL=secret_client.get_secret("UGA-URL").value
    ZMB_URL=secret_client.get_secret("ZMB-URL").value
    ETH_URL=secret_client.get_secret("ETH-URL").value
    KEN_URL=secret_client.get_secret("KEN-URL").value
    UGA_PASSWORD=secret_client.get_secret("UGA-PASSWORD").value
    ZMB_PASSWORD=secret_client.get_secret("ZMB-PASSWORD").value
    ETH_PASSWORD=secret_client.get_secret("ETH-PASSWORD").value
    KEN_PASSWORD=secret_client.get_secret("KEN-PASSWORD").value

except Exception as e:
    print('No access to Azure Key vault, skipping.')

# 2. Try to load secrets from env-variables (i.e. when using Github Actions)
try:
    import os
    
    ADMIN_LOGIN = os.environ['ADMIN_LOGIN']
    GLOFAS_USER = os.environ['GLOFAS_USER']
    GLOFAS_PW = os.environ['GLOFAS_PW']
    GOOGLE_DRIVE_DATA_URL = os.environ['GOOGLE_DRIVE_DATA_URL']
    UGA_URL=os.environ['UGA_URL']
    ZMB_URL=os.environ['ZMB_URL']
    ETH_URL=os.environ['ETH_URL']
    KEN_URL=os.environ['KEN_URL']
    UGA_PASSWORD=os.environ['UGA_PASSWORD']
    ZMB_PASSWORD=os.environ['ZMB_PASSWORD']
    ETH_PASSWORD=os.environ['ETH_PASSWORD']
    KEN_PASSWORD=os.environ['KEN_PASSWORD']

except Exception as e:
    print('No environment variables found.')

# 3. If 1. and 2. both fail, then assume secrets are loaded via secrets.py file (when running locally). If neither of the 3 options apply, this script will fail.
try:
    from drought_model.secrets import *
except ImportError:
    print('No secrets file found.')


######################
## COUNTRY SETTINGS ##
######################

# Countries to include
COUNTRY_CODES = ['KEN']#,'ZMB','ETH','UGA']

SETTINGS = {"KEN": {
        "IBF_API_URL": KEN_URL,
        "PASSWORD": KEN_PASSWORD,
        "mock": False,
        "if_mock_trigger": False,
        "notify_email": False,
        'lead_times': {
        1: '2-month',
        2:'1-month',
        3: '0-month',
        4:'6-month',
        5:'5-month',
        6:'4-month',
        7:'3-month',
        8:'2-month',
        9:'1-month',
        10:'0-month',
        11:'4-month',
        12:'3-month'
        },
        'admin_level': 1,
        'levels':[1],
        'EXPOSURE_DATA_SOURCES': {
            "population": {
                "source": "population/hrsl_ken_pop_resized_100",
                "rasterValue": 1
            }
        },
        'DYNAMIC_INDICATORS': {
            "population_affected":"population",
            "livestock_body_condition ":"livestock_condition ",
            "vegetation_condition":"vegetation_condition",
            "drought_phase_classification ":"drought_phase ",
        }
    },

}



# Change this date only in case of specific testing purposes
import datetime
CURRENT_DATE = datetime.date.today()
CURRENT_Month = datetime.date.today().month

#CURRENT_DATE=date.today() - timedelta(1) # to use yesterday's date




####################
## OTHER SETTINGS ##
####################
TRIGGER_PROBABILITY=40
SPI_Threshold_Prob='0.16354'
TRIGGER_LEVELS = {
    "minimum": 0.6,
    "medium": 0.7,
    "maximum": 0.8
}



###################
## PATH SETTINGS ##
###################
NDRMC_BULLETIN_FILE_PATH="https://www.ndma.go.ke/index.php/resource-center/national-drought-bulletin/send/39-drought-updates/6312-national-monthly-drought-updates-january-2022"
RASTER_DATA = 'data/raster/'
RASTER_INPUT = RASTER_DATA + 'input/'
RASTER_OUTPUT = RASTER_DATA + 'output/'
PIPELINE_DATA = 'data/'
PIPELINE_INPUT = PIPELINE_DATA + 'input/'
PIPELINE_OUTPUT = PIPELINE_DATA + 'output/'

#########################
## INPUT DATA SETTINGS ##
#########################

 
#####################
## ATTRIBUTE NAMES ##
#####################

TRIGGER_LEVEL = 'triggerLevel'
LEAD_TIME = 'leadTime'

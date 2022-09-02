from drought_model.forecast import Forecast
import traceback
import time
import datetime
from drought_model.settings import *
try:
    from drought_model.secrets import *
except ImportError:
    print('No secrets file found.')
from drought_model.exposure import Exposure
from drought_model.dynamicDataDb import DatabaseManager as dbm
import resource
import os
import logging
import zipfile

#from google_drive_downloader import GoogleDriveDownloader as gdd

# Set up logger
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='ex.log')
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


logger = logging.getLogger(__name__)
 

def main():
    startTime = time.time() 
    logger.info(str(datetime.datetime.now()))
    
    #dbm_ = dbm('1-month', 'ETH',3)
    dbm_ = dbm(leadTimeLabel='1-month', leadTimeValue=1,countryCodeISO3='ETH',admin_level=3)     
    filename='data.zip'
    path = 'drought/Gold/ibfdatapipeline/'+ filename
    #admin_area_json1['geometry'] = admin_area_json1.pop('geom')
    DataFile = dbm_.getDataFromDatalake(path)
    if DataFile.status_code >= 400:
        raise ValueError()
    open('./' + filename, 'wb').write(DataFile.content)
    path_to_zip_file='./'+filename
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('./data')
        
    logger.info('finished data download')
 
       
    try:
        for COUNTRY_CODE in COUNTRY_CODES.keys():
            logger.info(f'--------STARTING: {COUNTRY_CODE}' + '--------------------------')  
            for SEASON in COUNTRY_CODES[COUNTRY_CODE]['seasons']:
                logger.info(f'--------STARTING: {SEASON}' + '--------------------------')
                COUNTRY_SETTINGS = SETTINGS[COUNTRY_CODE]
                leadTimeLabels=COUNTRY_SETTINGS['lead_times'][SEASON][CURRENT_Month]
                #TRIGGER_SCENARIO=COUNTRY_SETTINGS['TRIGGER_SCENARIO']
                TRIGGER_SCENARIO=COUNTRY_SETTINGS['TRIGGER_LEVELS']['TRIGGER_SCENARIO']
                for leadTimeLabel in leadTimeLabels:
                    leadTimeValue=int(leadTimeLabel.split('-')[0])
                    logger.info(f'--------STARTING: {leadTimeLabel}' + '--------------------------')
                    fc = Forecast(leadTimeLabel, leadTimeValue, COUNTRY_CODE,SEASON,TRIGGER_SCENARIO,COUNTRY_SETTINGS['admin_level'])
                    if COUNTRY_CODE=='KEN' and KMD_FORECAST:
                        logger.info(f'--------STARTING: {COUNTRY_CODE}' + '--------------------------')
                        #fc.getdata.callAllExposure()        
                        fc.getdata_eth.callAllExposure_kenya()      
                    elif COUNTRY_CODE in ['ETH','KEN']:#=='ETH':
                        logger.info(f'--------STARTING: {COUNTRY_CODE}' + '--------------------------')
                        fc.getdata_eth.callAllExposure() 
                        
                    logger.info('--------Finished exposure data Processing')
                    #fc.getdata.callAllExposure()  
                    #logger.info('--------Finished exposure data Processing')
                    fc.db.upload()
                    logger.info('--------Finished upload')
                #fc.db.sendNotification()
                #logger.info('--------Finished notification')
            if COUNTRY_CODE=='ETH':
                #fc.hotspot() 
                fc.getdata_eth.ipc_proccessing()
                logger.info('--------Finished processing ipc class data')
                #fc.ipc()                
    except Exception as e:
        logger.error("Drought Data PIPELINE ERROR")
        logger.error(e)

    elapsedTime = str(time.time() - startTime)
    logger.info(str(elapsedTime))


if __name__ == "__main__":
    main()

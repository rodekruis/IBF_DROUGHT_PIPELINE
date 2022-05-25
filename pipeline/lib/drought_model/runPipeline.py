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
#import resource
import os
import logging
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
    try:
        for COUNTRY_CODE in COUNTRY_CODES.keys():
            logger.info(f'--------STARTING: {COUNTRY_CODE}' + '--------------------------')  
            for SEASON in COUNTRY_CODES[COUNTRY_CODE]['seasons']:
                logger.info(f'--------STARTING: {SEASON}' + '--------------------------')
                COUNTRY_SETTINGS = SETTINGS[COUNTRY_CODE]
                leadTimeLabels=COUNTRY_SETTINGS['lead_times'][SEASON][CURRENT_Month]
                TRIGGER_SCENARIO=COUNTRY_SETTINGS['TRIGGER_SCENARIO']
                for leadTimeLabel in leadTimeLabels:
                    leadTimeValue=int(leadTimeLabel.split('-')[0])
                    logger.info(f'--------STARTING: {leadTimeLabel}' + '--------------------------')
                    fc = Forecast(leadTimeLabel, leadTimeValue, COUNTRY_CODE,SEASON,TRIGGER_SCENARIO,COUNTRY_SETTINGS['admin_level'])
                    if COUNTRY_CODE=='KEN':
                        fc.getdata.callAllExposure()               
                    elif COUNTRY_CODE=='ETH':
                        fc.getdata_eth.callAllExposure()               
                    logger.info('--------Finished exposure data Processing')
                    #fc.getdata.callAllExposure()  
                    #logger.info('--------Finished exposure data Processing')
                    fc.db.upload()
                    logger.info('--------Finished upload')
                #fc.db.sendNotification()
                #logger.info('--------Finished notification')
            if COUNTRY_CODE=='ETH':
                fc.hotspot()
                fc.ipc()                
    except Exception as e:
        logger.error("Drought Data PIPELINE ERROR")
        logger.error(e)

    elapsedTime = str(time.time() - startTime)
    logger.info(str(elapsedTime))


if __name__ == "__main__":
    main()

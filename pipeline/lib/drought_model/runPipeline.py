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
        for COUNTRY_CODE in COUNTRY_CODES:
            logger.info(f'--------STARTING: {COUNTRY_CODE}' + '--------------------------')
            COUNTRY_SETTINGS = SETTINGS[COUNTRY_CODE]
            LEAD_TIMES = COUNTRY_SETTINGS['lead_times']
            leadTimeLabel=LEAD_TIMES[CURRENT_Month]
            leadTimeValue=int(leadTimeLabel.split('-')[0])
            logger.info(f'--------STARTING: {leadTimeLabel}' + '--------------------------')
            fc = Forecast(leadTimeLabel, leadTimeValue, COUNTRY_CODE,COUNTRY_SETTINGS['admin_level'])
            dynamic_draought_data =fc.getdata.processing()               
            logger.info('--------Finished data Processing')

            fc.getdata.callAllExposure()  
            logger.info('--------Finished exposure data Processing')
            fc.db.upload()
            logger.info('--------Finished upload')
            #fc.db.sendNotification()
            #logger.info('--------Finished notification')
    except Exception as e:
        logger.error("Drought Data PIPELINE ERROR")
        logger.error(e)

    elapsedTime = str(time.time() - startTime)
    logger.info(str(elapsedTime))


if __name__ == "__main__":
    main()

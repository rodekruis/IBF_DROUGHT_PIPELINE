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
from drought_model.ipcclass import IPCCLASS
#from drought_model.downloadecmwfforecast import downloadECMWFDATA
import drought_model.downloadforecast as EcmwfData
from drought_model.dynamicDataDb import DatabaseManager as dbm
from drought_model.googledrivedata import downloaddatalack 
import json
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
    
    # download ECMWF forecast    
    yearNow = CURRENT_DATE.year
    monthNow = CURRENT_DATE.strftime("%m")
    month_ = f'{monthNow}'
    year_ = f'{yearNow}'

    EcmwfData.downloadEcmwfForecast(ecmwfResoluation=0.5,ecmwfForecastPath=RASTER_INPUT,year_=year_,month_=month_)
    #EcmwfData.downloadEcmwfReForecast(ecmwfResoluation=0.5,ecmwfForecastPath=RASTER_INPUT)

    #downloadECMWFDATA.processing() 
    logger.info('finished data download')
    
    '''
    #dbm_ = dbm('1-month', 'ETH',3)
    dbm_ = dbm(leadTimeLabel='1-month', leadTimeValue=1,countryCodeISO3='ETH',admin_level=2)    
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
    '''    
    
    url_base='https://510ibfsystem.blob.core.windows.net/ibfdatapipelines/drought'
    
    if len(list(set(COUNTRY_CODES.keys())))==1:
        countryCode=list(set(COUNTRY_CODES.keys()))[0].lower()
        downloaddatalack(countryCode,url_base)
        filename=f'data_{countryCode}.zip'
        path_to_zip_file='./'+filename 
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall('./data')
    else:
        countryCode=None
        downloaddatalack(countryCode,url_base)
        filename='data.zip'
        path_to_zip_file='./'+filename 
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall('./data')  

    try:
        for COUNTRY_CODE in list(set(COUNTRY_CODES.keys())):
            logger.info(f'--------STARTING: {COUNTRY_CODE}' + '--------------------------')    
                   

            COUNTRY_SETTINGS = SETTINGS[COUNTRY_CODE]
            TRIGGER_SCENARIO=COUNTRY_SETTINGS['TRIGGER_LEVELS']['TRIGGER_SCENARIO']
            admin_level=COUNTRY_SETTINGS['admin_level']
            db = dbm(leadTimeLabel='1-month',
                     leadTimeValue=1,
                     countryCodeISO3=COUNTRY_CODE,
                     admin_level=admin_level,
                     SEASON=1)
            
            # get latest model run date from ibf portal 

            getRecentDate = db.apiGetRequestDate(COUNTRY_CODE,disasterType='drought')
            
            logger.info(f'----LAST MODEL RUN on : {getRecentDate.date().isoformat()}' + '-------') 
            ipcUploadChecker=0
            for SEASON in COUNTRY_CODES[COUNTRY_CODE]['seasons']:
                logger.info(f'--------STARTING: {SEASON}' + '--------------------------')               
                leadTimeLabels=COUNTRY_SETTINGS['lead_times'][SEASON][CURRENT_Month]
                #TRIGGER_SCENARIO=COUNTRY_SETTINGS['TRIGGER_SCENARIO']               
              
                for leadTimeLabel in leadTimeLabels:
                    leadTimeValue=int(leadTimeLabel.split('-')[0])
                    ecmwfLeadTimeValue=ecmwfLeadTimeValueList[leadTimeLabel]
                     
                    
                    logger.info(f'--------STARTING: {leadTimeLabel}' + '--------------------------')
                    fc = Forecast(leadTimeLabel,leadTimeValue,COUNTRY_CODE,SEASON,TRIGGER_SCENARIO,COUNTRY_SETTINGS['admin_level'],ecmwfLeadTimeValue)
                    
                    if leadTimeValue <4:
                        if COUNTRY_CODE=='KEN' and KMD_FORECAST:
                            logger.info(f'--------STARTING: {COUNTRY_CODE}' + '--------------------------')
                            #fc.getdata.callAllExposure()        
                            fc.getdata_eth.callAllExposure_kenya()      
                        elif COUNTRY_CODE in ['ETH','KEN','UGA']:#=='ETH':
                            logger.info(f'--------STARTING: {COUNTRY_CODE}' + '--------------------------')
                            #fc.getdata_eth.callAllExposure() 
                            fc.ecmwf.callAllExposure() 
                            fc.db.upload()
                            logger.info('--------Finished upload')
                            fc.db.sendNotification(leadTime=leadTimeValue)
                            logger.info('---Finished sending email') 
                            
                        logger.info('--------Finished exposure data Processing')

                    else:
                        df_total=fc.population_total_pcode
                        for indicator in ['population_affected','alert_threshold']:
                            df_total["amount"] = 0
                            statsdf = df_total[["placeCode", "amount"]]
                            stats = statsdf.to_dict(orient="records")  
                            exposure_data = {'countryCodeISO3': COUNTRY_CODE}
                            exposure_data['exposurePlaceCodes'] = stats
                            exposure_data["adminLevel"] = admin_level
                            exposure_data["leadTime"] = leadTimeLabel
                            exposure_data["dynamicIndicator"] = indicator
                            exposure_data["eventName"] = None
                            
                            statsPath = PIPELINE_OUTPUT + "calculated_affected/affected_" + str(leadTimeValue) + "_"  + COUNTRY_CODE + "_admin_" + f"{admin_level}_"+ indicator + f'_region{SEASON}'+ ".json"	
                            
                            with open(statsPath, 'w') as f:
                                json.dump(exposure_data, f)
                        
                        fc.db.upload()
                        logger.info('--------Finished upload')
                    if ipcUploadChecker <1:
                        fc.ipcclass.ipc_proccessing() 
                        logger.info('--------Finished processing ipc class data')
                        fc.db.uploadipc() 
                        fc.db.uploadRasterFile()
                        logger.info('--------Finished uploadingg ipc class data')   
                        ipcUploadChecker=ipcUploadChecker+1 
                    
            
            logger.info('--------Finished notification')
      

    except Exception as e:
        logger.error("Drought Data PIPELINE ERROR")
        logger.error(e)

    elapsedTime = str(time.time() - startTime)
    logger.info(str(elapsedTime))


if __name__ == "__main__":
    main()

import os
import geopandas as gpd
import numpy
import rioxarray
import rasterio as rio
import xarray as xr
import pandas as pd
import numpy as np
import math
from scipy.stats import gamma
from scipy.stats import norm
import glob
from geocube.api.core import make_geocube
from datetime import datetime
import re
import json
import logging

logger = logging.getLogger(__name__)

from ftplib import FTP, FTP_TLS
from drought_model.settings import *
from climate_indices import (
    compute,
    indices,
    utils,
)  # compute, eto, palmer, utils , lmoments

try:
    from drought_model.secrets import *
except ImportError:
    print("No secrets file found.")


class ICPACDATA:
    def __init__(
        self,
        leadTimeLabel,
        leadTimeValue,
        SEASON,
        TRIGGER_SCENARIO,
        admin_area_gdf,
        population_total,
        countryCodeISO3,
        admin_level,
    ):

        self.leadTimeLabel = leadTimeLabel
        self.leadTimeValue = leadTimeValue
        self.TRIGGER_SCENARIO = TRIGGER_SCENARIO
        self.SEASON = SEASON
        self.RASTER_OUTPUT = RASTER_OUTPUT
        self.countryCodeISO3 = countryCodeISO3
        self.population_df = population_total

        self.Icpac_Forecast_FtpPath = Icpac_Forecast_FtpPath
        self.Icpac_Forecast_FilePath = Icpac_Forecast_FilePath
        self.Icpac_Forecast_FtpPath_Rain = Icpac_Forecast_FtpPath_Rain
        self.Icpac_Forecast_FilePath_Rain = Icpac_Forecast_FilePath_Rain

        self.ICPAC_FTP_ADDRESS = ICPAC_FTP_ADDRESS
        self.ICPAC_FTP_USERNAME = ICPAC_FTP_USERNAME
        self.ICPAC_FTP_PASSWORD = ICPAC_FTP_PASSWORD

        self.TRIGGER_threshold = SETTINGS[countryCodeISO3]["TRIGGER_threshold"][0]
        self.EXPOSURE_DATA_SOURCES = SETTINGS[countryCodeISO3]["EXPOSURE_DATA_SOURCES"]
        self.DYNAMIC_INDICATORS = SETTINGS["DYNAMIC_INDICATORS"]
        self.TRIGGER_threshold_percentage = SETTINGS[countryCodeISO3][
            "TRIGGER_threshold"
        ][1]

        self.TRIGGER_rain_prob_threshold = SETTINGS[countryCodeISO3][
            "TRIGGER_rain_prob_threshold"
        ][0]
        self.TRIGGER_rain_prob_threshold_percentage = SETTINGS[countryCodeISO3][
            "TRIGGER_rain_prob_threshold"
        ][1]

        croping_zones_pcode = PIPELINE_INPUT + "croping_zones_pcode.csv"

        crop_df = pd.read_csv(croping_zones_pcode)
        admin_zones_eth = PIPELINE_INPUT + "admin2.geojson"
        admin_df = gpd.read_file(admin_zones_eth)  # fc.admin_area_gdf
        admin_area_gdf = admin_df

        admin_df = pd.merge(
            admin_area_gdf,
            crop_df[["ADM2_PCODE", "Crop_group"]],
            how="left",
            left_on="ADM2_PCODE",
            right_on="ADM2_PCODE",
        )

        self.min_lon = math.floor(admin_area_gdf.total_bounds[0])
        self.min_lat = math.floor(admin_area_gdf.total_bounds[1])
        self.max_lon = math.ceil(admin_area_gdf.total_bounds[2])
        self.max_lat = math.ceil(admin_area_gdf.total_bounds[3])

        admin_df = admin_df.query(f"Crop_group=={self.SEASON}")

        ### create a new unique identifier with type integer
        admin_df["ind"] = admin_df.apply(lambda row: row.ADM2_PCODE[-4:], axis=1)
        admin_df["pcode"] = admin_df.ind.astype(int)
        admin_df["cropzone"] = admin_df.Crop_group.astype(int)

        self.admin_df = admin_df

    def downloadforecast(self):
        try:
            self.retrieve_icpac_forecast_ftp()
        except:
            logger.error(" ICPAC FTP ERROR")
            pass

    def processing(self):

        spi_data = self.process_rain_total_eth()
        # drought_indicators=self.read_bulletin()
        df = self.population_df
        df_spi = pd.merge(df, spi_data, how="left", on="placeCode")
        df_spi["trigger"] = df_spi[self.TRIGGER_SCENARIO].fillna(0)
        df_spi["population_affected"] = df_spi.apply(
            lambda row: row.trigger * row.value, axis=1
        )
        return df_spi

    def callAllExposure(self):

        df_total = self.processing()
        # drought_indicators=self.read_bulletin()
        # for indicator, values in self.DYNAMIC_INDICATORS.items():
        # df_stats_levl=drought_indicators[indicator]
        # self.statsPath=PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
        # str(self.leadTimeValue) + '_' + self.countryCodeISO3 +'_admin_' +str(self.admin_level) + '_' + indicator + '.json'
        # result = {
        # 'countryCodeISO3': self.countryCodeISO3,
        # 'exposurePlaceCodes': df_stats_levl,
        # 'leadTime': self.leadTimeLabel,
        # 'dynamicIndicator': indicator,
        # 'adminLevel': self.admin_level
        # }

        # with open(self.statsPath, 'w') as fp:
        # json.dump(result, fp)

        for indicator, values in self.EXPOSURE_DATA_SOURCES.items():
            try:
                logger.info(f"indicator: {indicator}")
                df_total["amount"] = df_total[indicator]
                population_affected = df_total[["placeCode", "amount"]]
                stats = population_affected.to_dict(orient="records")
                df_stats = pd.DataFrame(stats)
                # stats_dff = pd.merge(df,self.pcode_df,  how='left',left_on='placeCode', right_on = f'placeCode_{self.admin_level}')
                for adm_level in SETTINGS[self.countryCodeISO3]["levels"]:
                    if adm_level == self.admin_level:
                        df_stats_levl = stats
                    else:
                        df_stats_levl = df_stats.groupby(f"placeCode_{adm_level}").agg(
                            {"amount": "sum"}
                        )
                        df_stats_levl.reset_index(inplace=True)
                        df_stats_levl["placeCode"] = df_stats_levl[
                            f"placeCode_{adm_level}"
                        ]
                        df_stats_levl = df_stats_levl[["amount", "placeCode"]].to_dict(
                            orient="records"
                        )

                    self.statsPath = (
                        PIPELINE_OUTPUT
                        + "calculated_affected/affected_"
                        + str(self.leadTimeValue)
                        + "_"
                        + self.countryCodeISO3
                        + "_admin_"
                        + str(adm_level)
                        + "_"
                        + indicator
                        + ".json"
                    )

                    result = {
                        "countryCodeISO3": self.countryCodeISO3,
                        "exposurePlaceCodes": df_stats_levl,
                        "leadTime": self.leadTimeLabel,
                        "dynamicIndicator": indicator,  # + '_affected',
                        "adminLevel": adm_level,
                    }

                    with open(self.statsPath, "w") as fp:
                        json.dump(result, fp)

                    if indicator == "population_affected":
                        alert_threshold = list(
                            map(self.get_alert_threshold, df_stats_levl)
                        )

                        alert_threshold_file_path = (
                            PIPELINE_OUTPUT
                            + "calculated_affected/affected_"
                            + str(self.leadTimeValue)
                            + "_"
                            + self.countryCodeISO3
                            + "_admin_"
                            + str(adm_level)
                            + "_"
                            + "alert_threshold"
                            + ".json"
                        )

                        alert_threshold_records = {
                            "countryCodeISO3": self.countryCodeISO3,
                            "exposurePlaceCodes": alert_threshold,
                            "leadTime": self.leadTimeLabel,
                            "dynamicIndicator": "alert_threshold",
                            "adminLevel": adm_level,
                        }

                        with open(alert_threshold_file_path, "w") as fp:
                            json.dump(alert_threshold_records, fp)
            except:
                logger.info(f"failed to output for indicator: {indicator}")
                pass

    def process_rain_total_eth(self):

        admin_df = self.admin_df
        prediction_data = CURRENT_DATE.strftime("%Y-%m-01")

        df_prediction_raw = (
            xr.open_dataset(self.Icpac_Forecast_FilePath_Rain, decode_times=False)
            .rename({"lat": "y", "lon": "x"})
            .rio.write_crs("epsg:4326", inplace=True)
        )
        df_prediction_ = df_prediction_raw.rio.clip_box(
            minx=self.min_lon, miny=self.min_lat, maxx=self.max_lon, maxy=self.max_lat
        )

        # save rainfall data to geotiff

        total_rain_forecast = (
            self.RASTER_OUTPUT
            + f"rainfall_extent_{self.leadTimeValue}_"
            + self.countryCodeISO3
            + ".tif"
        )
        df_prediction_raw["prec"].rio.to_raster(total_rain_forecast)

        precipitation = df_prediction_[
            "prec"
        ]  # df_prediction_raw['below'].rio.clip(admin_df.geometry.values, admin_df.crs, from_disk=True).sel(band=1).drop("band")
        precipitation.name = "precipitation"

        ####

        df_prediction_ = df_prediction_.rename({"y": "lat", "x": "lon"})

        # make sure we have the arrays with time as the inner-most dimension
        preferred_dims = ("lat", "lon")  # , "time")
        df_prediction_ = df_prediction_.transpose(*preferred_dims)

        historical_rain_data = RASTER_INPUT + "CHIRPS/chirps-*.tif"
        geotiff_list = glob.glob(historical_rain_data)

        time_var = xr.Variable(
            "time",
            [
                datetime.strptime(items.split("\\")[-1][12:19], "%Y.%m")
                for items in geotiff_list
            ],
        )

        geotiffs_da = xr.concat(
            [self.open_clip_tiff(i, df_prediction_) for i in geotiff_list], dim=time_var
        )
        geotiffs_ds = geotiffs_da.to_dataset(name="prec")

        df_prediction_1 = xr.concat(
            [
                df_prediction_["prec"] / 3,
                df_prediction_["prec"] / 3,
                df_prediction_["prec"] / 3,
            ],
            dim="time",
        )
        df_prediction_1["time"] = pd.date_range(
            start=prediction_data, periods=3, freq="M"
        )
        preferred_dims = ("lat", "lon", "time")

        observation = geotiffs_ds["prec"].transpose(*preferred_dims)

        # create one time sereies
        data3 = xr.concat([observation, df_prediction_1], dim="time")

        data_icpac = data3  # .rename({"y": "lat","x":"lon"})
        # make sure we have the arrays with time as the inner-most dimension

        data_icpac = data_icpac.transpose(*preferred_dims)
        data_icpac["time"] = pd.date_range(
            start=data_icpac["time"][0].values,
            periods=len(data_icpac["time"].values),
            freq="M",
        )

        data_arrays = {"rain": data_icpac}

        for label, da in data_arrays.items():
            if da["lat"][0] > da["lat"][1]:
                print(
                    f"The {label}-resolution DataArray's lats are descending -- flipping"
                )
                da["lat"] = np.flip(da["lat"])
            if da["lon"][0] > da["lon"][1]:
                print(
                    f"The {label}-resolution DataArray's lons are descending -- flipping"
                )
                da["lon"] = np.flip(da["lon"])

        da_precip_lo = data_arrays["rain"]

        initial_year = int(da_precip_lo["time"][0].dt.year)

        scale_months = 3

        icpac_spi_data = self.apply_spi_gamma_monthly(
            data_array=da_precip_lo,
            months=3,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2020,
        )

        icpac_spi = icpac_spi_data.transpose("lat", "lon", "time")
        icpac_spi = icpac_spi.to_dataset(name="spi3").rename({"lon": "x", "lat": "y"})
        # icpac_spi.rename({"lon": "x", "lat": "y"})

        # SPI for the

        spi_data = icpac_spi.isel(time=[-1])
        spi = spi_data["spi3"]

        # make your geo cube
        out_grid = make_geocube(
            vector_data=admin_df,
            measurements=["pcode", "cropzone"],
            like=spi,  # ensure the data are on the same grid
        )
        # spi.rename({'lon':'x','lat':'y'})

        #

        spi1 = spi.where(spi < self.TRIGGER_threshold)

        out_grid["spi"] = (spi1.dims, spi1.values, spi1.attrs, spi1.encoding)

        zonal_stats_df = (
            out_grid.groupby(out_grid.pcode).count().to_dataframe().reset_index()
        )

        zonal_stats_df["placeCode"] = zonal_stats_df.apply(
            lambda row: "ET" + str(int(row.pcode)).zfill(4), axis=1
        )
        zonal_stats_df["percentage"] = zonal_stats_df.apply(
            lambda row: 100 * (int(row.spi) / int(row.cropzone)), axis=1
        )
        zonal_stats_df.loc[
            zonal_stats_df["percentage"] >= self.TRIGGER_threshold_percentage,
            "Trigger_threshold_spi",
        ] = 1
        zonal_stats_df.loc[
            zonal_stats_df["percentage"] < self.TRIGGER_threshold_percentage,
            "Trigger_threshold_spi",
        ] = 0

        spi_data = icpac_spi.isel(time=[-4])
        spi = spi_data["spi3"]

        # merge the two together

        spi_obs = spi.where(spi < self.TRIGGER_threshold)

        out_grid["spi"] = (
            spi_obs.dims,
            spi_obs.values,
            spi_obs.attrs,
            spi_obs.encoding,
        )
        zonal_stats_df_obs = (
            out_grid.groupby(out_grid.pcode).count().to_dataframe().reset_index()
        )

        zonal_stats_df_obs["placeCode"] = zonal_stats_df_obs.apply(
            lambda row: "ET" + str(int(row.pcode)).zfill(4), axis=1
        )
        zonal_stats_df_obs["percentage"] = zonal_stats_df_obs.apply(
            lambda row: 100 * (int(row.spi) / int(row.cropzone)), axis=1
        )
        zonal_stats_df_obs.loc[
            zonal_stats_df_obs["percentage"] >= self.TRIGGER_threshold_percentage,
            "Trigger_threshold_spi_obs",
        ] = 1
        zonal_stats_df_obs.loc[
            zonal_stats_df_obs["percentage"] < self.TRIGGER_threshold_percentage,
            "Trigger_threshold_spi_obs",
        ] = 0
        spifile_name = PIPELINE_OUTPUT + "zonalstats_spi.csv"
        profile_name = PIPELINE_OUTPUT + "zonalstats_obs.csv"
        zonal_stats_df.to_csv(spifile_name)
        zonal_stats_df_obs.to_csv(profile_name)

        threshold_df_spi = pd.merge(
            zonal_stats_df[["placeCode", "Trigger_threshold_spi"]],
            zonal_stats_df_obs[["placeCode", "Trigger_threshold_spi_obs"]],
            how="left",
            left_on="placeCode",
            right_on="placeCode",
        )

        ########################################
        ############# probabilistic forecast ###
        ########################################

        df_prediction_prob = (
            xr.open_dataset(self.Icpac_Forecast_FilePath, decode_times=False)
            .rename({"lat": "y", "lon": "x"})
            .rio.write_crs("epsg:4326", inplace=True)
        )
        precipitation = df_prediction_prob["below"].rio.clip_box(
            minx=self.min_lon, miny=self.min_lat, maxx=self.max_lon, maxy=self.max_lat
        )

        below_rain_forecast = (
            self.RASTER_OUTPUT
            + f"rainfall_below_{self.leadTimeValue}_"
            + self.countryCodeISO3
            + ".tif"
        )
        normal_rain_forecast = (
            self.RASTER_OUTPUT
            + f"rainfall_normal_{self.leadTimeValue}_"
            + self.countryCodeISO3
            + ".tif"
        )
        above_rain_forecast = (
            self.RASTER_OUTPUT
            + f"rainfall_above_{self.leadTimeValue}_"
            + self.countryCodeISO3
            + ".tif"
        )

        df_prediction_prob["below"].rio.to_raster(below_rain_forecast)
        df_prediction_prob["normal"].rio.to_raster(normal_rain_forecast)
        df_prediction_prob["above"].rio.to_raster(above_rain_forecast)

        # make your geo cube
        out_grid_prob = make_geocube(
            vector_data=admin_df,
            measurements=["pcode", "cropzone"],
            like=precipitation,  # ensure the data are on the same grid
        )

        # merge the two together
        precipitation = precipitation.where(
            precipitation < self.TRIGGER_rain_prob_threshold
        )
        out_grid_prob["below"] = (
            precipitation.dims,
            precipitation.values,
            precipitation.attrs,
            precipitation.encoding,
        )

        # merge the two together

        zonal_stats_rain_prob_df = (
            out_grid_prob.groupby(out_grid_prob.pcode)
            .count()
            .to_dataframe()
            .reset_index()
        )

        zonal_stats_rain_prob_df["placeCode"] = zonal_stats_rain_prob_df.apply(
            lambda row: "ET" + str(int(row.pcode)).zfill(4), axis=1
        )
        zonal_stats_rain_prob_df["percentage"] = zonal_stats_rain_prob_df.apply(
            lambda row: 100 * (int(row.below) / int(row.cropzone)), axis=1
        )
        zonal_stats_rain_prob_df.loc[
            zonal_stats_rain_prob_df["percentage"]
            >= self.TRIGGER_rain_prob_threshold_percentage,
            "Trigger_threshold_below",
        ] = 1

        zonal_stats_rain_prob_df.loc[
            zonal_stats_rain_prob_df["percentage"]
            < self.TRIGGER_rain_prob_threshold_percentage,
            "Trigger_threshold_below",
        ] = 0
        profile_name = PIPELINE_OUTPUT + "zonalstats_prob.csv"
        zonal_stats_rain_prob_df.to_csv(profile_name)

        threshold_df = pd.merge(
            threshold_df_spi[
                ["placeCode", "Trigger_threshold_spi", "Trigger_threshold_spi_obs"]
            ],
            zonal_stats_rain_prob_df[["placeCode", "Trigger_threshold_below"]],
            how="left",
            left_on="placeCode",
            right_on="placeCode",
        )
        threshold_df["trigger_treshold_both"] = threshold_df.apply(
            lambda row: (row.Trigger_threshold_below) * row.Trigger_threshold_spi,
            axis=1,
        )
        threshold_df["trigger_treshold_one"] = threshold_df.apply(
            lambda row: math.ceil(
                0.5 * row.Trigger_threshold_below + 0.5 * row.Trigger_threshold_spi
            ),
            axis=1,
        )

        return threshold_df

    def retrieve_icpac_forecast_ftp(
        self,
    ):
        """
        Download and save a file from ICPAC's ftp server.
        Parameters
        ----------
        ftp_filepath : str
            path on the server where the file is located
        output_filepath : str
            path to save the file to
        Examples
        --------
        >>> retrieve_file_ftp(ftp_filepath=
        ... '/SharedData/gcm/seasonal/202101/' \
        ... 'PredictedRainfallProbbability-FMA2021_Jan2021.nc',
        ... output_filepath='example.nc')
        """
        ftp_address = self.ICPAC_FTP_ADDRESS
        ftp_username = self.ICPAC_FTP_USERNAME
        ftp_password = self.ICPAC_FTP_PASSWORD
        # question: is this the best method to set the vars and raise the error?
        if None in (ftp_address, ftp_username, ftp_password):
            raise RuntimeError("One of the ftp variables is not set")
        # TODO: ugly, is there a better method? if not doing, mypy complains
        assert ftp_address is not None, ftp_address
        assert ftp_username is not None, ftp_username
        assert ftp_password is not None, ftp_password
        ftps = self.connect_icpac_ftp(
            ftp_address=ftp_address,
            ftp_username=ftp_username,
            ftp_password=ftp_password,
        )
        with open(self.Icpac_Forecast_FilePath, "wb") as f:
            ftps.retrbinary("RETR " + self.Icpac_Forecast_FtpPath, f.write)

        with open(self.Icpac_Forecast_FilePath_Rain, "wb") as f:
            ftps.retrbinary("RETR " + self.Icpac_Forecast_FtpPath_Rain, f.write)

    def connect_icpac_ftp(
        self,
        ftp_address: str,
        ftp_username: str,
        ftp_password: str,
    ) -> FTP_TLS:
        """
        @author: https://github.com/Tinkaa
        https://github.com/OCHA-DAP/pa-aa-toolbox/blob/083fd436e5a4ae530aaf69f0b290b104042faa9c/src/aatoolbox/datasources/icpac/icpac.py

        Connect to ICPAC's ftp.
        To connect you need credentials.
        Parameters
        ----------
        ftp_address : str
            IP address of the ftp server
        ftp_username : str
            username of the ftp server
        ftp_password : str
            password of the ftp server
        Returns
        -------
        ftps: FTP
            a FTP object
        """
        # for some reason it was impossible to access the FTP
        # this class fixes it, which is copied from this SO
        # https://stackoverflow.com/questions/14659154/ftps-with-python-ftplib-session-reuse-required
        class MyFTP_TLS(FTP_TLS):
            """Explicit FTPS, with shared TLS session."""

            def ntransfercmd(self, cmd, rest=None):
                conn, size = FTP.ntransfercmd(self=self, cmd=cmd, rest=rest)
                if self._prot_p:
                    conn = self.context.wrap_socket(
                        sock=conn,
                        server_hostname=self.host,
                        session=self.sock.session,
                    )  # this is the fix
                return conn, size

        ftps = MyFTP_TLS(host=ftp_address, user=ftp_username, passwd=ftp_password)
        ftps.prot_p()

        return ftps

    def open_clip_tiff(self, filename, precipitation_forecast):
        """
        clip tiff files to forecasted rainfall extent
        """
        preferred_dims = ("lat", "lon")

        t = (
            rioxarray.open_rasterio(filename)
            .rio.write_crs("epsg:4326", inplace=True)
            .rio.clip_box(
                minx=self.min_lon,
                miny=self.min_lat,
                maxx=self.max_lon,
                maxy=self.max_lat,
            )
            .sel(band=1)
            .drop("band")
        )
        t = t.where(t > 0)
        t = t.rename({"x": "lon", "y": "lat"})
        t = t.transpose(*preferred_dims)

        # interpolate into the higher resolution grid from IMERG
        interp_t = t.interp(
            lat=precipitation_forecast["lat"], lon=precipitation_forecast["lon"]
        )

        return interp_t

    def apply_spi_gamma_monthly(
        self,
        data_array: xr.DataArray,
        months: int,
        data_start_year: int = 2000,
        calibration_year_initial: int = 2000,
        calibration_year_final: int = 2020,
    ) -> xr.DataArray:

        # stack the lat and lon dimensions into a new dimension named point, so at each lat/lon
        # we'll have a time series for the geospatial point, and group by these points
        da_precip_groupby = data_array.stack(point=("lat", "lon")).groupby("point")

        spi_args = {
            "scale": months,
            "distribution": indices.Distribution.gamma,
            "data_start_year": data_start_year,
            "calibration_year_initial": calibration_year_initial,
            "calibration_year_final": calibration_year_final,
            "periodicity": compute.Periodicity.monthly,
        }

        # apply the SPI function to the data array
        da_spi = xr.apply_ufunc(indices.spi, da_precip_groupby, kwargs=spi_args)

        # unstack the array back into original dimensions
        da_spi = da_spi.unstack("point")

        return da_spi

    def process_rain_probability_eth(self):

        admin_df = self.admin_area_gdf

        df_prediction_prob = (
            xr.open_dataset(self.Icpac_Forecast_FilePath, decode_times=False)
            .rename({"lat": "y", "lon": "x"})
            .rio.write_crs("epsg:4326", inplace=True)
        )
        precipitation = df_prediction_prob["below"].rio.clip_box(
            minx=self.min_lon, miny=self.min_lat, maxx=self.max_lon, maxy=self.max_lat
        )

        below_rain_forecast = (
            self.RASTER_OUTPUT
            + f"rainfall_below_{self.leadTimeValue}_"
            + self.countryCodeISO3
            + ".tif"
        )
        normal_rain_forecast = (
            self.RASTER_OUTPUT
            + f"rainfall_normal_{self.leadTimeValue}_"
            + self.countryCodeISO3
            + ".tif"
        )
        above_rain_forecast = (
            self.RASTER_OUTPUT
            + f"rainfall_above_{self.leadTimeValue}_"
            + self.countryCodeISO3
            + ".tif"
        )

        df_prediction_prob["below"].rio.to_raster(below_rain_forecast)
        df_prediction_prob["normal"].rio.to_raster(normal_rain_forecast)
        df_prediction_prob["above"].rio.to_raster(above_rain_forecast)

        # make your geo cube
        out_grid_prob = make_geocube(
            vector_data=admin_df,
            measurements=["pcode", "cropzone"],
            like=precipitation,  # ensure the data are on the same grid
        )
        # merge the two together
        precipitation = precipitation.where(
            precipitation < self.TRIGGER_rain_prob_threshold
        )
        out_grid_prob["below"] = (
            precipitation.dims,
            precipitation.values,
            precipitation.attrs,
            precipitation.encoding,
        )

        # merge the two together

        zonal_stats_rain_prob_df = (
            out_grid_prob.groupby(out_grid_prob.pcode)
            .count()
            .to_dataframe()
            .reset_index()
        )

        zonal_stats_rain_prob_df["placeCode"] = zonal_stats_rain_prob_df.apply(
            lambda row: "ET" + str(int(row.pcode)).zfill(4), axis=1
        )
        zonal_stats_rain_prob_df["percentage"] = zonal_stats_rain_prob_df.apply(
            lambda row: 100 * (int(row.below) / int(row.cropzone)), axis=1
        )
        zonal_stats_rain_prob_df.loc[
            zonal_stats_rain_prob_df["percentage"]
            >= self.TRIGGER_rain_prob_threshold_percentage,
            "Trigger_threshold_prob",
        ] = 1
        zonal_stats_rain_prob_df.loc[
            zonal_stats_rain_prob_df["percentage"]
            < self.TRIGGER_rain_prob_threshold_percentage,
            "Trigger_threshold_prob",
        ] = 0

        return zonal_stats_rain_prob_df

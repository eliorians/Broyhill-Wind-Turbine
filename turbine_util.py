
import os
import time
import traceback
import warnings
import pandas as pd
import logging
import pytz

import forecast_util

logger = logging.getLogger('turbine_util')

#all columns that exist in frames.csv
#see "./turbine-data/turbine_data_info.cfg" for column info
column_names = [
    "idx", "timestamp", "sample_cnt", 
    "WTG1_R_filt_genWinding_temp_calc_degC", "WTG1_R_filt_genWinding_temp_calc_degC_MAX", "WTG1_R_filt_genWinding_temp_calc_degC_MIN", 
    "WTG1_R_DSP_GridStateEventStatus", "WTG1_R_DSP_GridStateEventStatus_MAX", "WTG1_R_DSP_GridStateEventStatus_MIN", 
    "WTG1_R_DSP_Ctrl_Opt", "WTG1_R_DSP_SW_SVN_Rev", "WTG1_R_ctrl_inv_Q_cmd_kVAR", 
    "WTG1_R_ctrl_rec_torque_meas_kNm", "WTG1_R_ctrl_rec_torque_cmd_kNm", "WTG1_R_EncoderPositionBkup_cnt", 
    "WTG1_R_InvOffTime_sec", "WTG1_R_InvStandbyTime_sec", "WTG1_R_InvActiveTime_sec", 
    "WTG1_R_RecOffTime_sec", "WTG1_R_RecMotorTime_sec", "WTG1_R_RecStandbyTime_sec", 
    "WTG1_R_RecActiveTime_sec", "WTG1_R_ctrl_rec_speed_limit_RPM", "WTG1_R_GridOutageEvent_cnt", 
    "WTG1_R_GridSagEvent_cnt", "WTG1_R_BlowerFanTime_sec", "WTG1_R_CabFanTime_sec", 
    "WTG1_R_GenFanTime_sec", "WTG1_R_Rec_Id_cmd_A", "WTG1_R_SigDeltaPhiCheck_deg", 
    "WTG1_R_FanGenCycle_cnt", "WTG1_R_FanCabCycle_cnt", "WTG1_R_FanBlowerCycle_cnt", 
    "WTG1_R_TempUnused1_degC", "WTG1_R_ctrl_rec_spd_hyst_out_RPM", "WTG1_R_filt_temp_amb_degC", 
    "WTG1_R_filt_temp_gen1_degC", "WTG1_R_filt_temp_gen2_degC", "WTG1_R_filt_temp_frame_degC", 
    "WTG1_R_ctrl_rec_airDensity_kg_m3", "WTG1_R_ctrl_rec_Nr_RPM", "WTG1_R_ctrl_rec_Nr_RPM_MAX", 
    "WTG1_R_ctrl_rec_Nr_RPM_MIN", "WTG1_R_ctrl_rec_N4_RPM", "WTG1_R_ctrl_rec_N4_RPM_MAX", 
    "WTG1_R_ctrl_rec_N4_RPM_MIN", "WTG1_R_SigRotDeltaPhi_deg", "WTG1_R_Rec_pst_k", 
    "WTG1_R_Rec_pst_k_MAX", "WTG1_R_Rec_pst_k_MIN", "WTG1_W_Set_ctrl_P_set_kW",
    "WTG1_R_Rec_comp_k", "WTG1_R_Rec_comp_k_MAX", "WTG1_R_Rec_comp_k_MIN", "WTG1_R_Rec_comp_k_STDDEV", 
    "WTG1_R_Rec_pctl_spd_limit_RPM", "WTG1_R_Rec_pctl_spd_limit_RPM_MAX", "WTG1_R_Rec_pctl_spd_limit_RPM_MIN", 
    "WTG1_R_Rec_pctl_spd_limit_RPM_STDDEV", "WTG1_R_Accel_X_ABS_G", "WTG1_R_Accel_X_ABS_G_MAX", 
    "WTG1_R_Accel_X_ABS_G_MIN", "WTG1_R_Accel_X_ABS_G_STDDEV", "WTG1_R_Accel_Y_ABS_G", 
    "WTG1_R_Accel_Y_ABS_G_MAX", "WTG1_R_Accel_Y_ABS_G_MIN", "WTG1_R_Accel_Y_ABS_G_STDDEV", 
    "WTG1_R_TotalkVARPower_kVAR", "WTG1_R_TotalkVARPower_kVAR_MAX", "WTG1_R_TotalkVARPower_kVAR_MIN", 
    "WTG1_R_TotalkVARPower_kVAR_STDDEV", "WTG1_R_DBLActuation_cnt", "WTG1_R_DBLActuation_cnt_MIN", 
    "WTG1_R_VibDWThrsh_cnt", "WTG1_R_VibXWThrsh_cnt", "WTG1_R_VibDWFlt_cnt", 
    "WTG1_R_VibXWFlt_cnt", "WTG1_R_VibDWWrn_cnt", "WTG1_R_VibXWWrn_cnt", 
    "WTG1_R_GridQualityFlag", "WTG1_R_GridQualityFlag_MAX", "WTG1_R_GridQualityFlag_MIN", "WTG1_R_DSPBootRunTime_Secs", 
    "WTG1_R_InvPwr_kW", "WTG1_R_InvPwr_kW_MAX", "WTG1_R_InvPwr_kW_MIN", "WTG1_R_InvPwr_kW_STDDEV",  #INVERTER POWER COLUMNS
    "WTG1_R_InvVltLNPhsA_Vrms", "WTG1_R_InvVltLNPhsB_Vrms", "WTG1_R_InvVltLNPhsC_Vrms",      
    "WTG1_R_InvCurPhsA_Arms", "WTG1_R_InvCurPhsB_Arms", "WTG1_R_InvCurPhsC_Arms", 
    "WTG1_R_InvFreq_Hz", 
    "WTG1_R_TempCab_degC",  #INSIDE TURBINE TEMPERATURE
    "WTG1_R_TempAmb_degC",  #OUTSIDE TEMPERATURE
    "WTG1_R_RecPLLSpd_RPM", "WTG1_R_RecPLLSpd_RPM_MAX", "WTG1_R_RecPLLSpd_RPM_MIN", 
    "WTG1_R_RecPLLSpd_RPM_STDDEV", "WTG1_R_InvEnergyTot_kWh", "WTG1_R_AccelDW_G",
    "WTG1_R_AccelDX_G", "WTG1_R_TempL12_degC", "WTG1_R_TempBusCap_degC", 
    "WTG1_R_TempGen1_degC", "WTG1_R_TempGen2_degC", 
    "WTG1_R_TempIGBTinv_degC", "WTG1_R_TempIGBTrec_degC", "WTG1_R_TempMIB_degC", 
    "WTG1_R_RecCurPhsA_Arms", "WTG1_R_RecCurPhsB_Arms", "WTG1_R_RecCurPhsC_Arms",
    "WTG1_R_RecPwr_kW", 
    "WTG1_R_DBLPwr_kW", "WTG1_R_DBLPwr_kW_MAX", "WTG1_R_DBLPwr_kW_MIN", "WTG1_R_DBLPwr_kW_STDDEV", 
    "WTG1_R_TempFrame_degC", 
    "WTG1_R_YawPosition_deg", 
    "WTG1_R_RotorSpeed_RPM", "WTG1_R_RotorSpeed_RPM_MAX", "WTG1_R_RotorSpeed_RPM_MIN", #ROTOR SPEED
    "WTG1_R_RotorSpeed_RPM_STDDEV", 
    "WTG1_R_WindSpeed_mps", "WTG1_R_WindSpeed_mps_MAX", "WTG1_R_WindSpeed_mps_MIN", "WTG1_R_WindSpeed_mps_STDDEV", #INSTANTANEOUS WINDSPEED
    "WTG1_R_WindSpeed1m_mps", "WTG1_R_WindSpeed10m_mps", #WINDSPEED 1M AND 10M AVG
    "WTG1_R_YawVaneAvg_deg", "WTG1_R_YawVaneAvg_deg_MAX", "WTG1_R_YawVaneAvg_deg_MIN", "WTG1_R_YawVaneAvg_deg_STDDEV",  #yaw position
    "WTG1_R_TurbineState", "WTG1_R_TurbineState_MAX", "WTG1_R_TurbineState_MIN", #TURBINE STATE
    "WTG1_R_BrakeState", "WTG1_R_BrakeState_MIN", 
    "WTG1_R_TimeOnline_hr", "WTG1_R_TimeAvailable_hr", 
    "WTG1_R_AnyFltCond", "WTG1_R_AnyFltCond_MAX",       #WARNING FLAGS
    "WTG1_R_AnyEnvCond", "WTG1_R_AnyEnvCond_MAX", 
    "WTG1_R_AnyExtCond", "WTG1_R_AnyExtCond_MAX", 
    "WTG1_R_AnyWrnCond", "WTG1_R_AnyWrnCond_MAX", 
    "WTG1_R_StateOffTime_sec", "WTG1_R_StateOffTime_sec_MIN", #CUMULATIVE OFF TIME
    "WTG1_R_StateWaitTime_sec", "WTG1_R_StateWaitTime_sec_MIN", 
    "WTG1_R_StateMotorTime_sec", "WTG1_R_StateMotorTime_sec_MIN", 
    "WTG1_R_StateStandbyTime_sec", "WTG1_R_StateStandbyTime_sec_MIN", 
    "WTG1_R_StateServiceTime_sec", "WTG1_R_StateServiceTime_sec_MIN", 
    "WTG1_R_RunTime_sec", "WTG1_R_RunTime_sec_MIN", "WTG1_R_RunTime_sec_AVG", "WTG1_R_RunTime_sec_STDDEV", 
    "WTG1_R_ResetCount", "WTG1_R_ResetCount_MIN", 
    "WTG1_R_MCUBootRunTime_sec", "WTG1_R_MCUBootRunTime_sec_MIN", 
    "WTG1_R_StateActiveTime_sec", "WTG1_R_StateActiveTime_sec_MIN", 
    "WTG1_R_AvailableTime_sec", "WTG1_R_AvailableTime_sec_MIN", 
    "WTG1_R_EnvCondTime_sec", "WTG1_R_EnvCondTime_sec_MIN", 
    "WTG1_R_ExtCondTime_sec", "WTG1_R_ExtCondTime_sec_MIN", 
    "WTG1_R_FaultTime_sec", "WTG1_R_FaultTime_sec_MIN", 
    "WTG1_W_DispatchWDtimeout_sec", "WTG1_R_PowerProductionState", "WTG1_R_PowerProductionState_MIN", 
    "WTG1_R_PowerProductionTimer", "WTG1_R_PowerProductionTimer_MIN", 
    "WTG1_R_DispatchableState", "WTG1_R_DispatchableState_MIN", 
    "WTG1_R_MCUSnaplogRunning", "WTG1_R_MCUSnaplogRunning_MIN", 
    "WTG1_R_WindSpeed1s_mps", "WTG1_R_WindSpeed1s_mps_MAX", "WTG1_R_WindSpeed1s_mps_MIN", "WTG1_R_WindSpeed1s_mps_STDDEV", #WINDSPEED 1s AVG 
    "WTG1_R_RawWindSpeedInst_mps", "WTG1_R_RawWindSpeedInst_mps_MAX", "WTG1_R_RawWindSpeedInst_mps_MIN", "WTG1_R_RawWindSpeedInst_mps_STDDEV", #RAW WIND SPEEDS
    "WTG1_R_RawWindSpeed1s_mps", "WTG1_R_RawWindSpeed1s_mps_MAX", "WTG1_R_RawWindSpeed1s_mps_MIN", "WTG1_R_RawWindSpeed1s_mps_STDDEV", 
    "WTG1_R_RawWindSpeed1m_mps", "WTG1_R_RawWindSpeed1m_mps_MAX", "WTG1_R_RawWindSpeed1m_mps_MIN", "WTG1_R_RawWindSpeed1m_mps_STDDEV", 
    "WTG1_R_RawWindSpeed10m_mps", 
    "WTG1_R_YawLeftTime_sec", "WTG1_R_YawRightTime_sec",    #yaw wind up
    "WTG1_R_BrkANormApply_cnt", "WTG1_R_BrkAFastApply_cnt", "WTG1_R_BrkARelease_cnt", 
    "WTG1_R_BrkBNormApply_cnt", "WTG1_R_BrkBFastApply_cnt", "WTG1_R_BrkBRelease_cnt", 
    "WTG1_R_YawUnwindLeft_sec", "WTG1_R_YawUnwindLeft_sec_MIN", 
    "WTG1_R_YawUnwindRight_sec", "WTG1_R_YawUnwindRight_sec_MIN", #yaw unwind
    "WTG1_R_TimeNonOp_sec", "WTG1_R_TimeNonOp_sec_MIN", 
    "WTG1_R_TimeExclude_sec", "WTG1_R_TimeExclude_sec_MIN", 
    "WTG1_R_NonAlmConditions", "WTG1_R_NonAlmConditions_MIN", 
    "WTG1_R_YawErrorShortTermAvg_deg", "WTG1_R_YawErrorShortTermAvg_deg_MAX", "WTG1_R_YawErrorShortTermAvg_deg_MIN", #yaw errors
    "WTG1_R_YawErrorLongTermAvg_deg", "WTG1_R_YawErrorLongTermAvg_deg_MAX", "WTG1_R_YawErrorLongTermAvg_deg_MIN",    #yaw errirs
    "WTG1_R_YawChatterRMS_deg", "WTG1_R_YawChatterRMS_deg_MAX", "WTG1_R_YawChatterRMS_deg_MIN", "WTG1_R_YawChatterTime_sec", #yaw chatter
    "WTG1_R_YawMotorStart_cnt", 
    "WTG1_R_YawRate_deg_per_sec", "WTG1_R_YawRate_deg_per_sec_MAX", "WTG1_R_YawRate_deg_per_sec_MIN", #yaw rate
    "WTG1_R_TSW_Version", "WTG1_R_Turbine_SN", "WTG1_R_DispatchDisableTimer_sec", 
    "WTG1_R_HydraulicBrakeCaliperPressure1_PSI", "WTG1_R_HydraulicBrakeCaliperPressure1_PSI_MAX", "WTG1_R_HydraulicBrakeCaliperPressure1_PSI_MIN", 
    "WTG1_R_HydraulicBrakeCaliperPressure2_PSI", "WTG1_R_HydraulicBrakeCaliperPressure2_PSI_MAX", "WTG1_R_HydraulicBrakeCaliperPressure2_PSI_MIN", 
    "WTG1_R_HydraulicBrakeCaliperPressure3_PSI", "WTG1_R_HydraulicBrakeCaliperPressure3_PSI_MAX", "WTG1_R_HydraulicBrakeCaliperPressure3_PSI_MIN", 
    "WTG1_R_HydraulicSystemPressure_PSI", "WTG1_R_HydraulicSystemPressure_PSI_MAX", "WTG1_R_HydraulicSystemPressure_PSI_MIN", 
    "WTG1_R_MCU_SW_SVN_Rev", "WTG1_R_HydraulicBrakeFastApplyTime_sec", 
    "WTG1_R_MCU_GridEventStatus", "WTG1_R_MCU_GridEventStatus_MAX", "WTG1_R_MCU_GridEventStatus_MIN"
]

#columns to be used
use_columns = [
    'timestamp',                                                                                                           #current time
    'WTG1_R_InvPwr_kW', 'WTG1_R_InvPwr_kW_MAX', 'WTG1_R_InvPwr_kW_MIN', 'WTG1_R_InvPwr_kW_STDDEV',                         #power produced
    "WTG1_R_InvFreq_Hz",                                                                                                   #inverter frequency 
    'WTG1_R_WindSpeed_mps', 'WTG1_R_WindSpeed_mps_MAX', 'WTG1_R_WindSpeed_mps_MIN', 'WTG1_R_WindSpeed_mps_STDDEV',         #windspeed instantaneous
    'WTG1_R_WindSpeed1m_mps', 'WTG1_R_WindSpeed10m_mps',                                                                   #windspeed 1m and 10m
    "WTG1_R_WindSpeed1s_mps", "WTG1_R_WindSpeed1s_mps_MAX", "WTG1_R_WindSpeed1s_mps_MIN", "WTG1_R_WindSpeed1s_mps_STDDEV", #windspeed 1s
    'WTG1_R_TempAmb_degC',                                                                                                 #ambient temperature
    'WTG1_R_YawLeftTime_sec', 'WTG1_R_YawRightTime_sec', 'WTG1_R_YawUnwindRight_sec', 'WTG1_R_YawUnwindLeft_sec',          #yaw wind/unwind
    "WTG1_R_YawVaneAvg_deg", "WTG1_R_YawVaneAvg_deg_MAX", "WTG1_R_YawVaneAvg_deg_MIN", "WTG1_R_YawVaneAvg_deg_STDDEV",     #yaw position
    "WTG1_R_RotorSpeed_RPM", "WTG1_R_RotorSpeed_RPM_MAX", "WTG1_R_RotorSpeed_RPM_MIN",                                     #rotor speed
    'WTG1_R_AnyWrnCond', "WTG1_R_AnyFltCond", "WTG1_R_AnyEnvCond", "WTG1_R_AnyExtCond", "WTG1_R_DSP_GridStateEventStatus"  #any warning flags
]

#column types for each used column
turbine_column_types = {
    'WTG1_R_InvPwr_kW'               : float,               # Power produced
    'WTG1_R_InvPwr_kW_MAX'           : float,
    'WTG1_R_InvPwr_kW_MIN'           : float,
    'WTG1_R_InvPwr_kW_STDDEV'        : float,
    'WTG1_R_InvFreq_Hz'              : float,               # Inverter frequency
    'WTG1_R_WindSpeed_mps'           : float,               # Wind speed instantaneous
    'WTG1_R_WindSpeed_mps_MAX'       : float,
    'WTG1_R_WindSpeed_mps_MIN'       : float,
    'WTG1_R_WindSpeed_mps_STDDEV'    : float,
    'WTG1_R_WindSpeed1m_mps'         : float,               # Wind speed at 1m
    'WTG1_R_WindSpeed10m_mps'        : float,               # Wind speed at 10m
    'WTG1_R_WindSpeed1s_mps'         : float,               # Wind speed at 1s
    'WTG1_R_WindSpeed1s_mps_MAX'     : float,
    'WTG1_R_WindSpeed1s_mps_MIN'     : float,
    'WTG1_R_WindSpeed1s_mps_STDDEV'  : float,
    'WTG1_R_TempAmb_degC'            : float,               # Ambient temperature
    'WTG1_R_YawLeftTime_sec'         : int,                 # Yaw wind/unwind
    'WTG1_R_YawRightTime_sec'        : int,
    'WTG1_R_YawUnwindRight_sec'      : int,
    'WTG1_R_YawUnwindLeft_sec'       : int,
    'WTG1_R_YawVaneAvg_deg'          : float,               # Yaw position
    'WTG1_R_YawVaneAvg_deg_MAX'      : float,
    'WTG1_R_YawVaneAvg_deg_MIN'      : float,
    'WTG1_R_YawVaneAvg_deg_STDDEV'   : float,
    'WTG1_R_RotorSpeed_RPM'          : float,               # Rotor speed
    'WTG1_R_RotorSpeed_RPM_MAX'      : float,
    'WTG1_R_RotorSpeed_RPM_MIN'      : float,
    'WTG1_R_AnyWrnCond'              : int,                 # Any warning flags
    'WTG1_R_AnyFltCond'              : int,                 # Any fault conditions
    'WTG1_R_AnyEnvCond'              : int,                 # Any environmental conditions
    'WTG1_R_AnyExtCond'              : int,                 # Any external conditions
    'WTG1_R_DSP_GridStateEventStatus': int                  # DSP Grid State Event Status
}

forecast_column_types = {
    'temperature_F'                     : int,
    'windSpeed_mph'                     : int,
    'windDirection'                     : str,
    'shortForecast'                     : str,
    'probabilityOfPrecipitation_percent': int,
    'dewpoint_degC'                     : float,
    'relativeHumidity_percent'          : int
}

#setup for program logging
def logging_setup():
    # Create a "logs" directory if it doesn't exist
    logs_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_directory, exist_ok=True)
    # Set up logging to a file in the "logs" directory
    log_file = os.path.join(logs_directory, 'turbine_util.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Send log messages to the console
            logging.FileHandler(log_file)  # Save log messages to a file in the "logs" directory
        ])

#read the main frames.csv SQL dump file
def readSQLDump():
    logger.info("in readSQLDump")

    #path to data and the row the data starts
    dataPath = "./turbine-data/frames.csv"
    skipTop = 17
    skipBottom = 8

    #read tubrine data into dataframe
    df = pd.read_csv(dataPath, sep='\t', skiprows=skipTop, header=None, names=column_names, usecols=use_columns)
    
    #drop end of dump file 
    df.drop(df.tail(skipBottom).index, inplace=True)
    
    return df

#finds the forecast filename in ./forecast-data-processed and returns true if it exists
def findForecastFile(filename):
    filepath = './forecast-data-processed/' + filename
    return os.path.isfile(filepath)

#clean turbine data, see various steps throughout
def cleanTurbineData(df):
    logger.info("in cleanTurbineData")

    #error catching
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)
    try:

        #set timestamp type, reading mixed format
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc= True)
        #make format consistent
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #set back to datetime with consistent format
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)

        #set all other column types
        df = df.astype(turbine_column_types)

        #aggregate hourly
        df.set_index('timestamp', inplace=True)
        df = df.resample('H').agg({
            'WTG1_R_InvPwr_kW'               : 'mean',
            'WTG1_R_InvPwr_kW_MAX'           : 'max',
            'WTG1_R_InvPwr_kW_MIN'           : 'min',
            'WTG1_R_InvPwr_kW_STDDEV'        : 'mean',
            'WTG1_R_InvFreq_Hz'              : 'mean',
            'WTG1_R_WindSpeed_mps'           : 'mean',
            'WTG1_R_WindSpeed_mps_MAX'       : 'max',
            'WTG1_R_WindSpeed_mps_MIN'       : 'min',
            'WTG1_R_WindSpeed_mps_STDDEV'    : 'mean',
            'WTG1_R_WindSpeed1m_mps'         : 'mean',
            'WTG1_R_WindSpeed10m_mps'        : 'mean',
            'WTG1_R_WindSpeed1s_mps'         : 'mean',
            'WTG1_R_WindSpeed1s_mps_MAX'     : 'max',
            'WTG1_R_WindSpeed1s_mps_MIN'     : 'min',
            'WTG1_R_WindSpeed1s_mps_STDDEV'  : 'mean',
            'WTG1_R_TempAmb_degC'            : 'mean',
            'WTG1_R_YawLeftTime_sec'         : 'mean',
            'WTG1_R_YawRightTime_sec'        : 'mean',
            'WTG1_R_YawUnwindRight_sec'      : 'mean',
            'WTG1_R_YawUnwindLeft_sec'       : 'mean',
            'WTG1_R_YawVaneAvg_deg'          : 'mean',
            'WTG1_R_YawVaneAvg_deg_MAX'      : 'max',
            'WTG1_R_YawVaneAvg_deg_MIN'      : 'min',
            'WTG1_R_YawVaneAvg_deg_STDDEV'   : 'mean',
            'WTG1_R_RotorSpeed_RPM'          : 'mean',
            'WTG1_R_RotorSpeed_RPM_MAX'      : 'max',
            'WTG1_R_RotorSpeed_RPM_MIN'      : 'min',
            'WTG1_R_AnyWrnCond'              : 'last',
            'WTG1_R_AnyFltCond'              : 'last',
            'WTG1_R_AnyEnvCond'              : 'last',
            'WTG1_R_AnyExtCond'              : 'last',
            'WTG1_R_DSP_GridStateEventStatus': 'last'
        }).reset_index()

    except FutureWarning as warning:
        logger.warning(warning)
    except Exception as error:
        logger.error(error)

    #df.to_csv('./turbine-data-processed/cleanedFrames.csv')
    #logger.info('"./turbine-data-processed/cleanedFrames.csv" saved successfully')

    return df

def combineTurbineForecast(df, hours_to_forecast):
    logger.info("in combineTurbineForecast")

    #process forecast data
    forecast_util.main()

    #error catching
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)
        try:

            #create timestamp_est since forecast files are in EST
            df['timestamp_est'] = df['timestamp'].dt.tz_convert(pytz.timezone('US/Eastern'))
            #subtract HOURS_TO_FORECAST to get the time of forecast that we are looking for
            df['timestamp_est_forecast'] = df['timestamp_est'] - pd.Timedelta(hours=hours_to_forecast)
            #create corresponding forecast filename
            df['forecast_file'] = ('forecast_' + df['timestamp_est_forecast'].dt.strftime('%m-%d-%Y_%H-%M') + '.csv').astype(str)
            #create a column that will be true if the forecast data exists
            df['forecast_file_exists'] = df['forecast_file'].apply(findForecastFile).astype(bool)
            #drop calculation columns
            df = df.drop(['timestamp_est', 'timestamp_est_forecast'], axis=1)

            #list to hold compacted forecast dataframes (each forecast_df -> 1 row)
            forecast_dfs = []

            #NOTE: forecast dfs dont always start at the time of their filetime. This may be due to the time of collected
            #      ex: collected at 2:00pm -> first forecast is for 2:00pm-3:00pm
            #          but if collected at 2:01pm -> first forecast is for 3:00-4:00pm
            #          this isnt confirmed why but this is occuring, must select rows based around the timestamp not the index

            #iterate turbine rows
            for _, turbine_row in df.iterrows():
                turbine_row['timestamp'] = pd.to_datetime(turbine_row['timestamp'], utc=True)

                #if forecast file exists for the current turbine_row
                if turbine_row['forecast_file_exists'] == True:
                    
                    #create filepath, and turn to df
                    file_path = os.path.join('./forecast-data-processed/', turbine_row['forecast_file'])
                    forecast_df = pd.read_csv(file_path)
                    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'], utc=True)
                    #locate index of matching timestamp
                    forecast_index = forecast_df[forecast_df['timestamp'] == turbine_row['timestamp']].index
                    #drop future dates
                    forecast_df = forecast_df.iloc[:forecast_index[0] + 1]
                    #reverse order so matching time is at the top
                    forecast_df = forecast_df.sort_index(ascending=False)
                    #clean up index (make index 0 to n-1 starting with matching forecast)
                    forecast_df.reset_index(drop=True, inplace=True)
                    forecast_df.drop(columns=['Unnamed: 0', 'timestamp'], inplace=True)
                    #drop most of the older ones
                    forecast_df = forecast_df.head(hours_to_forecast)
                    #stack to create multi index
                    forecast_df = forecast_df.stack()
                    #get new column names in the form of column_0 for row 0 and so on
                    new_columns = [f'{column}_{index}' for index, column in forecast_df.index]
                    #set columns and transform back to frame
                    forecast_df.index = new_columns
                    forecast_df = forecast_df.to_frame().T
                    #add timestamp back in for merging
                    forecast_df['timestamp'] = turbine_row['timestamp']
                    #add to list to be merged later
                    forecast_dfs.append(forecast_df)
                    #forecast_df.to_csv('./turbine-data-processed/forecast_test.csv')

            #concat list of forecast dfs and clean it
            forecast_df = pd.concat(forecast_dfs)
            forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'], utc=True)
            #forecast_df.to_csv('./turbine-data-processed/forecasts_test.csv')
            
            #final merge
            df = df.merge(forecast_df, on='timestamp', how='outer')

        except FutureWarning as warning:
            logger.warning(warning)
        except Exception as error:
            logger.exception(error)
            traceback.print_exc()

    #df.to_csv('./turbine-data-processed/combinedFrames.csv')
    #logger.info('"./turbine-data-processed/combinedFrames.csv" saved successfully')
    return df

def trimData(df):
    logger.info("in trimData")
 
    #error catching
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)
        try:

            #trim down to the first forecast value and beyond
            firstForecast_index = df['forecast_file_exists'].idxmax()
            df = df.loc[firstForecast_index:]
            df.reset_index(drop=True, inplace=True)

            #drop rows with missing data
            df = df.dropna()

        except FutureWarning as warning:
            logger.warning(warning)
        except Exception as error:
            logger.exception(error)
            traceback.print_exc()

    #df.to_csv('./turbine-data-processed/trimmedFrames.csv')
    #logger.info('"./turbine-data-processed/trimmedFrames.csv" saved successfully')
    return df

def main(hours_to_forecast):

    logging_setup()
    logger.info("Starting turbine_util")

    #read in data
    df = readSQLDump()

    #clean data
    df = cleanTurbineData(df)

    #combine with forecast data
    df = combineTurbineForecast(df, hours_to_forecast)

    #trim dataframe down to the data that will be used to train with
    df = trimData(df)

    df.to_csv('./turbine-data-processed/finalFrames.csv')
    logger.info('"./turbine-data-processed/finalFrames.csv" saved successfully')
    return df

if __name__ == "__main__":
    main()

import os
import pandas as pd
import logging
import pytz

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
    "WTG1_R_TurbineState", "WTG1_R_TurbineState_MAX", "WTG1_R_TurbineState_MIN", 
    "WTG1_R_BrakeState", "WTG1_R_BrakeState_MIN", 
    "WTG1_R_TimeOnline_hr", "WTG1_R_TimeAvailable_hr", 
    "WTG1_R_AnyFltCond", "WTG1_R_AnyFltCond_MAX", 
    "WTG1_R_AnyEnvCond", "WTG1_R_AnyEnvCond_MAX", 
    "WTG1_R_AnyExtCond", "WTG1_R_AnyExtCond_MAX", 
    "WTG1_R_AnyWrnCond", "WTG1_R_AnyWrnCond_MAX", 
    "WTG1_R_StateOffTime_sec", "WTG1_R_StateOffTime_sec_MIN", 
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
column_types = {
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

    #set timestamp and make consistent
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    #set all other column types
    df = df.astype(column_types)

    #aggregate hourly
    df = df.groupby(df['timestamp'].dt.floor('H')).agg({
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

    #change to eastern time zone, this must be done after grouping to effectively handle daylight saving time
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC', ambiguous='infer')
    df['timestamp'] = df['timestamp'].dt.tz_convert(pytz.timezone('US/Eastern'))

    #add column for the associated forecast with the hour, 
    df['forecast_file'] = 'forecast_' + df['timestamp'].dt.strftime('%m-%d-%Y_%H-%M') + '.csv'
    df['forecast_file'] = df['forecast_file'].astype(str)

    #add a column that will be true if the forecast data exists
    df['forecast_file_exists'] = df['forecast_file'].apply(findForecastFile)
    df['forecast_file_exists'] = df['forecast_file_exists'].astype(bool)

    return df

def main():

    logging_setup()
    logger.info("Running turbine-util...")

    df = readSQLDump()

    df = cleanTurbineData(df)

    df.to_csv('./turbine-data-processed/cleanedFrames.csv')
    logger.info('Turbine data cleaned and saved to "./turbine-data-processed/cleanedFrames.csv"')

if __name__ == "__main__":
    main()
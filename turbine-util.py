
import os
import pandas as pd
import logging

logger = logging.getLogger('turbine_util')

column_names = [
    "idx", "timestamp", "sample_cnt", 
    "WTG1_R_filt_genWinding_temp_calc_degC", "WTG1_R_filt_genWinding_temp_calc_degC_MAX", "WTG1_R_filt_genWinding_temp_calc_degC_MIN",  
    "WTG1_R_DSP_GridStateEventStatus", "WTG1_R_DSP_GridStateEventStatus_MAX", "WTG1_R_DSP_GridStateEventStatus_MIN", 
    "WTG1_R_DSP_Ctrl_Opt", "WTG1_R_DSP_SW_SVN_Rev", "WTG1_R_ctrl_inv_Q_cmd_kVAR", "WTG1_R_ctrl_rec_torque_meas_kNm", "WTG1_R_ctrl_rec_torque_cmd_kNm", "WTG1_R_EncoderPositionBkup_cnt", 
    "WTG1_R_InvOffTime_sec", "WTG1_R_InvStandbyTime_sec", "WTG1_R_InvActiveTime_sec", 
    "WTG1_R_RecOffTime_sec", "WTG1_R_RecMotorTime_sec", "WTG1_R_RecStandbyTime_sec", "WTG1_R_RecActiveTime_sec", "WTG1_R_ctrl_rec_speed_limit_RPM", 
    "WTG1_R_GridOutageEvent_cnt", "WTG1_R_GridSagEvent_cnt", "WTG1_R_BlowerFanTime_sec", "WTG1_R_CabFanTime_sec", 
    "WTG1_R_GenFanTime_sec", "WTG1_R_Rec_Id_cmd_A", "WTG1_R_SigDeltaPhiCheck_deg", "WTG1_R_FanGenCycle_cnt", "WTG1_R_FanCabCycle_cnt", "WTG1_R_FanBlowerCycle_cnt", "WTG1_R_TempUnused1_degC", 
    "WTG1_R_ctrl_rec_spd_hyst_out_RPM", "WTG1_R_filt_temp_amb_degC", "WTG1_R_filt_temp_gen1_degC", "WTG1_R_filt_temp_gen2_degC", "WTG1_R_filt_temp_frame_degC", "WTG1_R_ctrl_rec_airDensity_kg_m3", 
    "WTG1_R_ctrl_rec_Nr_RPM", "WTG1_R_ctrl_rec_Nr_RPM_MAX", "WTG1_R_ctrl_rec_Nr_RPM_MIN", "WTG1_R_ctrl_rec_N4_RPM", "WTG1_R_ctrl_rec_N4_RPM_MAX", "WTG1_R_ctrl_rec_N4_RPM_MIN", "WTG1_R_SigRotDeltaPhi_deg", 
    "WTG1_R_Rec_pst_k", "WTG1_R_Rec_pst_k_MAX", "WTG1_R_Rec_pst_k_MIN", "WTG1_W_Set_ctrl_P_set_kW", "WTG1_R_Rec_comp_k", "WTG1_R_Rec_comp_k_MAX", "WTG1_R_Rec_comp_k_MIN", "WTG1_R_Rec_comp_k_STDDEV", 
    "WTG1_R_Rec_pctl_spd_limit_RPM", "WTG1_R_Rec_pctl_spd_limit_RPM_MAX", "WTG1_R_Rec_pctl_spd_limit_RPM_MIN", "WTG1_R_Rec_pctl_spd_limit_RPM_STDDEV", "WTG1_R_Accel_X_ABS_G", "WTG1_R_Accel_X_ABS_G_MAX", "WTG1_R_Accel_X_ABS_G_MIN", 
    "WTG1_R_Accel_X_ABS_G_STDDEV", "WTG1_R_Accel_Y_ABS_G", "WTG1_R_Accel_Y_ABS_G_MAX", "WTG1_R_Accel_Y_ABS_G_MIN", "WTG1_R_Accel_Y_ABS_G_STDDEV", "WTG1_R_TotalkVARPower_kVAR", "WTG1_R_TotalkVARPower_kVAR_MAX", "WTG1_R_TotalkVARPower_kVAR_MIN", 
    "WTG1_R_TotalkVARPower_kVAR_STDDEV", "WTG1_R_DBLActuation_cnt", "WTG1_R_DBLActuation_cnt_MIN", "WTG1_R_VibDWThrsh_cnt", "WTG1_R_VibXWThrsh_cnt", "WTG1_R_VibDWFlt_cnt", "WTG1_R_VibXWFlt_cnt", "WTG1_R_VibDWWrn_cnt", "WTG1_R_VibXWWrn_cnt", "WTG1_R_GridQualityFlag", 
    "WTG1_R_GridQualityFlag_MAX", "WTG1_R_GridQualityFlag_MIN", "WTG1_R_DSPBootRunTime_Secs", "WTG1_R_InvPwr_kW", "WTG1_R_InvPwr_kW_MAX", "WTG1_R_InvPwr_kW_MIN", "WTG1_R_InvPwr_kW_STDDEV", "WTG1_R_InvVltLNPhsA_Vrms", "WTG1_R_InvVltLNPhsB_Vrms", "WTG1_R_InvVltLNPhsC_Vrms", 
    "WTG1_R_InvCurPhsA_Arms", "WTG1_R_InvCurPhsB_Arms", "WTG1_R_InvCurPhsC_Arms", "WTG1_R_InvFreq_Hz", "WTG1_R_TempCab_degC", "WTG1_R_TempAmb_degC", "WTG1_R_RecPLLSpd_RPM", "WTG1_R_RecPLLSpd_RPM_MAX", "WTG1_R_RecPLLSpd_RPM_MIN", "WTG1_R_RecPLLSpd_RPM_STDDEV", "WTG1_R_InvEnergyTot_kWh", 
    "WTG1_R_AccelDW_G", "WTG1_R_AccelDX_G", "WTG1_R_TempL12_degC", "WTG1_R_TempBusCap_degC", "WTG1_R_TempGen1_degC", "WTG1_R_TempGen2_degC", "WTG1_R_TempIGBTinv_degC", "WTG1_R_TempIGBTrec_degC", "WTG1_R_TempMIB_degC", "WTG1_R_RecCurPhsA_Arms", "WTG1_R_RecCurPhsB_Arms", "WTG1_R_RecCurPhsC_Arms",
    "WTG1_R_RecPwr_kW", "WTG1_R_DBLPwr_kW", "WTG1_R_DBLPwr_kW_MAX", "WTG1_R_DBLPwr_kW_MIN", "WTG1_R_DBLPwr_kW_STDDEV", "WTG1_R_TempFrame_degC", "WTG1_R_YawPosition_deg", "WTG1_R_RotorSpeed_RPM", "WTG1_R_RotorSpeed_RPM_MAX", "WTG1_R_RotorSpeed_RPM_MIN", "WTG1_R_RotorSpeed_RPM_STDDEV", 
    "WTG1_R_WindSpeed_mps", "WTG1_R_WindSpeed_mps_MAX", "WTG1_R_WindSpeed_mps_MIN", "WTG1_R_WindSpeed_mps_STDDEV", "WTG1_R_WindSpeed1m_mps", "WTG1_R_WindSpeed10m_mps", "WTG1_R_YawVaneAvg_deg", "WTG1_R_YawVaneAvg_deg_MAX", "WTG1_R_YawVaneAvg_deg_MIN", "WTG1_R_YawVaneAvg_deg_STDDEV", 
    "WTG1_R_TurbineState", "WTG1_R_TurbineState_MAX", "WTG1_R_TurbineState_MIN", "WTG1_R_BrakeState", "WTG1_R_BrakeState_MIN", "WTG1_R_TimeOnline_hr", "WTG1_R_TimeAvailable_hr", "WTG1_R_AnyFltCond", "WTG1_R_AnyFltCond_MAX", "WTG1_R_AnyEnvCond", "WTG1_R_AnyEnvCond_MAX", "WTG1_R_AnyExtCond", 
    "WTG1_R_AnyExtCond_MAX", "WTG1_R_AnyWrnCond", "WTG1_R_AnyWrnCond_MAX", "WTG1_R_StateOffTime_sec", "WTG1_R_StateOffTime_sec_MIN", "WTG1_R_StateWaitTime_sec", "WTG1_R_StateWaitTime_sec_MIN", "WTG1_R_StateMotorTime_sec", "WTG1_R_StateMotorTime_sec_MIN", "WTG1_R_StateStandbyTime_sec", 
    "WTG1_R_StateStandbyTime_sec_MIN", "WTG1_R_StateServiceTime_sec", "WTG1_R_StateServiceTime_sec_MIN", "WTG1_R_RunTime_sec", "WTG1_R_RunTime_sec_MIN", "WTG1_R_RunTime_sec_AVG", "WTG1_R_RunTime_sec_STDDEV", "WTG1_R_ResetCount", "WTG1_R_ResetCount_MIN", "WTG1_R_MCUBootRunTime_sec", 
    "WTG1_R_MCUBootRunTime_sec_MIN", "WTG1_R_StateActiveTime_sec", "WTG1_R_StateActiveTime_sec_MIN", "WTG1_R_AvailableTime_sec", "WTG1_R_AvailableTime_sec_MIN", "WTG1_R_EnvCondTime_sec", "WTG1_R_EnvCondTime_sec_MIN", "WTG1_R_ExtCondTime_sec", "WTG1_R_ExtCondTime_sec_MIN", "WTG1_R_FaultTime_sec", 
    "WTG1_R_FaultTime_sec_MIN", "WTG1_W_DispatchWDtimeout_sec", "WTG1_R_PowerProductionState", "WTG1_R_PowerProductionState_MIN", "WTG1_R_PowerProductionTimer", "WTG1_R_PowerProductionTimer_MIN", "WTG1_R_DispatchableState", "WTG1_R_DispatchableState_MIN", "WTG1_R_MCUSnaplogRunning", 
    "WTG1_R_MCUSnaplogRunning_MIN", "WTG1_R_WindSpeed1s_mps", "WTG1_R_WindSpeed1s_mps_MAX", "WTG1_R_WindSpeed1s_mps_MIN", "WTG1_R_WindSpeed1s_mps_STDDEV", "WTG1_R_RawWindSpeedInst_mps", "WTG1_R_RawWindSpeedInst_mps_MAX", "WTG1_R_RawWindSpeedInst_mps_MIN", "WTG1_R_RawWindSpeedInst_mps_STDDEV", 
    "WTG1_R_RawWindSpeed1s_mps", "WTG1_R_RawWindSpeed1s_mps_MAX", "WTG1_R_RawWindSpeed1s_mps_MIN", "WTG1_R_RawWindSpeed1s_mps_STDDEV", "WTG1_R_RawWindSpeed1m_mps", "WTG1_R_RawWindSpeed1m_mps_MAX", "WTG1_R_RawWindSpeed1m_mps_MIN", "WTG1_R_RawWindSpeed1m_mps_STDDEV", "WTG1_R_RawWindSpeed10m_mps", 
    "WTG1_R_YawLeftTime_sec", "WTG1_R_YawRightTime_sec", "WTG1_R_BrkANormApply_cnt", "WTG1_R_BrkAFastApply_cnt", "WTG1_R_BrkARelease_cnt", "WTG1_R_BrkBNormApply_cnt", "WTG1_R_BrkBFastApply_cnt", "WTG1_R_BrkBRelease_cnt", "WTG1_R_YawUnwindLeft_sec", "WTG1_R_YawUnwindLeft_sec_MIN", 
    "WTG1_R_YawUnwindRight_sec", "WTG1_R_YawUnwindRight_sec_MIN", "WTG1_R_TimeNonOp_sec", "WTG1_R_TimeNonOp_sec_MIN", "WTG1_R_TimeExclude_sec", "WTG1_R_TimeExclude_sec_MIN", "WTG1_R_NonAlmConditions", "WTG1_R_NonAlmConditions_MIN", "WTG1_R_YawErrorShortTermAvg_deg", 
    "WTG1_R_YawErrorShortTermAvg_deg_MAX", "WTG1_R_YawErrorShortTermAvg_deg_MIN", "WTG1_R_YawErrorLongTermAvg_deg", "WTG1_R_YawErrorLongTermAvg_deg_MAX", "WTG1_R_YawErrorLongTermAvg_deg_MIN", "WTG1_R_YawChatterRMS_deg", "WTG1_R_YawChatterRMS_deg_MAX", "WTG1_R_YawChatterRMS_deg_MIN", 
    "WTG1_R_YawChatterTime_sec", "WTG1_R_YawMotorStart_cnt", "WTG1_R_YawRate_deg_per_sec", "WTG1_R_YawRate_deg_per_sec_MAX", "WTG1_R_YawRate_deg_per_sec_MIN", "WTG1_R_TSW_Version", "WTG1_R_Turbine_SN", "WTG1_R_DispatchDisableTimer_sec", "WTG1_R_HydraulicBrakeCaliperPressure1_PSI", 
    "WTG1_R_HydraulicBrakeCaliperPressure1_PSI_MAX", "WTG1_R_HydraulicBrakeCaliperPressure1_PSI_MIN", "WTG1_R_HydraulicBrakeCaliperPressure2_PSI", "WTG1_R_HydraulicBrakeCaliperPressure2_PSI_MAX", "WTG1_R_HydraulicBrakeCaliperPressure2_PSI_MIN", "WTG1_R_HydraulicBrakeCaliperPressure3_PSI", 
    "WTG1_R_HydraulicBrakeCaliperPressure3_PSI_MAX", "WTG1_R_HydraulicBrakeCaliperPressure3_PSI_MIN", "WTG1_R_HydraulicSystemPressure_PSI", "WTG1_R_HydraulicSystemPressure_PSI_MAX", "WTG1_R_HydraulicSystemPressure_PSI_MIN", "WTG1_R_MCU_SW_SVN_Rev", "WTG1_R_HydraulicBrakeFastApplyTime_sec", 
    "WTG1_R_MCU_GridEventStatus", "WTG1_R_MCU_GridEventStatus_MAX", "WTG1_R_MCU_GridEventStatus_MIN"
]

useColumns = [
    'timestamp', 'WTG1_R_InvPwr_kW'
]

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

def readSQLDump():
    #path to data and the row the data starts
    dataPath = "./turbine-data/frames.csv"
    skipRows = 17

    #read tubrine data into dataframe
    df = pd.read_csv(dataPath, sep='\t', skiprows=skipRows, header=None, names=column_names, usecols=useColumns)
    
    #todo stop reading frames when you reach the "\."
    
    return df


def cleanTurbineData(df):

    #todo aggregate hourly

    return df

def main():

    logging_setup()
    logger.info("Running turbine-util...")

    df = readSQLDump()
    df = cleanTurbineData(df)

    

if __name__ == "__main__":
    main()
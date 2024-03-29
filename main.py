
import os
import logging
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import turbine_util
import plots

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


logger = logging.getLogger('main')

def generate_features(hours_to_forecast, allFeats, feats_list):
    features = []
    if allFeats:
        feats_list=['windDirection', 'probabilityOfPrecipitation_percent', 'dewpoint_degC', 'relativeHumidity_percent', 'temperature_F', 'windSpeed_mph']
    for number in range(hours_to_forecast-1):
        for feature in feats_list:
            features.append(f"{feature}_{number}")
    return features

#! ------- CONFIG ------- !#

#The hour that will be forecast from
#set threshold minutes to 0 if changed to allow data to reset
hoursToForecast=12

#How often the data should be reprocessed
threshold_minutes=120

#Size of split in train/test data
split=.2

#Wether to train and evaluate the model. Set the model type from the model list, as well as the target and list of features to train with.
toTrain= True
modelType='polynomial_regression'
targetToTrain = 'WTG1_R_InvPwr_kW'
featuresToTrain = generate_features(hours_to_forecast=hoursToForecast, allFeats=False, feats_list=['windSpeed_mph'])
featuresToTrain = ['windSpeed_mph_0', 'windSpeed_mph_1', 'windSpeed_mph_2', 'windSpeed_mph_3']

#Wether to train and evaluate all models in the model list
toTrainAll = False

modelList = {
    'linear_regression'     : LinearRegression(),
    'random_forest'         : RandomForestRegressor(),
    'polynomial_regression' : make_pipeline(PolynomialFeatures(3), LinearRegression())
}

#Wether to plot stuff (not for turning off prediction outcomes)
toPlot=False

#! ------- END CONFIG ------- !#

def logging_setup():
    # Create a "logs" directory if it doesn't exist
    logs_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_directory, exist_ok=True)
    # Set up logging to a file in the "logs" directory
    log_file = os.path.join(logs_directory, 'main.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Send log messages to the console
            logging.FileHandler(log_file)  # Save log messages to a file in the "logs" directory
        ])

def dataProcessed(filepath, threshold_minutes):
    """
    Returns true if [filepath] has been updated in the last [threshold minutes].
    """
    # get the creation time of the file and current time
    file_creation_time = os.path.getmtime(filepath)
    current_time = time.time()
    # calculate the difference in seconds between the current time and the file creation time and convert to minutes
    time_difference_seconds = current_time - file_creation_time
    time_difference_minutes = time_difference_seconds / 60
    # check if the file has been created in the last threshold_minutes
    if time_difference_minutes <= threshold_minutes:
        return True
    else:
        return False

def train_test_split(df, split):
    """
    Sequentially splits a [df] based on the [split]
    """
    logger.info("in train_test_split")

    #find the split index
    split_index = int((1 - split) * len(df))
    #split
    train_df, test_df = df.iloc[:split_index], df.iloc[split_index:]

    train_df.to_csv('./model-data/train_df.csv')
    test_df.to_csv('./model-data/test_df.csv')
    return train_df, test_df


def train_eval_model(train_df, test_df, target, features, model_list, model_name):
    try:
        logger.info("in train_eval_model")

        model = model_list.get(model_name)

        # split train and test data into features and target
        x_train, y_train = train_df[features], train_df[target]
        x_test, y_test = test_df[features], test_df[target]

        # initialize and train the linear regression model
        model.fit(x_train, y_train)

        # predict on the test set
        y_pred = model.predict(x_test)

        # evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        #plot the predictions
        plots.plotPrediction(test_df['timestamp'], y_test, y_pred, model_name)

        #log evaluation metrics
        logger.info(f"Model: {model}")
        logger.info(f"Root Mean Squared Error: {rmse}")

        with open('./model-data/eval.txt', "a") as f:
            f.write(f"Model: {model}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"Features: {features}\n")
            f.write("\n")

    except Exception as e:
        logger.error(f"Error: {str(e)}")

def main():
    logging_setup()
    logger.info("Starting main")    
    
    #process turbine data and get dataframe
    #if the data has been processed recently then dont do it again, saves time
    data_path='./turbine-data-processed/finalFrames.csv'
    if (dataProcessed(data_path, threshold_minutes) == True):
        df = pd.read_csv(data_path)
    else:
        df = turbine_util.main(hoursToForecast)

    #plotting stuff
    if toPlot == True:
        plots.plot_PowerVSActualWind(df, 'WTG1_R_InvPwr_kW', 'WTG1_R_WindSpeed_mps')
        plots.plot_PowerVSForecastWind(df, 'WTG1_R_InvPwr_kW', 'windSpeed_mph_0')

    #train & evaluate the model, training all based on the config
    if toTrain == True:

        #train/test split
        train_df, test_df = train_test_split(df, split)

        if toTrainAll == True:
                for modelName, model in modelList.items():
                    train_eval_model(train_df, test_df, targetToTrain, featuresToTrain, modelList, modelName)
        else:
            modelName=modelType
            train_eval_model(train_df, test_df, targetToTrain, featuresToTrain, modelList, modelName)

if __name__ == "__main__":
    main()
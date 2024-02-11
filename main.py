
import csv
import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import turbine_util

logger = logging.getLogger('main')

def generate_features(hours_to_forecast, allFeats, feats_list):
    features = []
    if allFeats:
        feats_list=['windDirection', 'probabilityOfPrecipitation_percent', 'dewpoint_degC', 'relativeHumidity_percent', 'temperature_F', 'windSpeed_mph']
    for number in range(hours_to_forecast):
        for feature in feats_list:
            features.append(f"{feature}_{number}")

    return features


#! CONFIG

#the hour that will be forecast from
#set threshold minutes to 0 if changed to allow data to reset
hours_to_forecast=12

#how often the data should be reprocessed
threshold_minutes=60

#size of split in train/test data
split=.2

#target to train and plot
target = 'WTG1_R_InvPwr_kW'

#list of features to train
features_to_train = ['windDirection_0', 'probabilityOfPrecipitation_percent_0', 'dewpoint_degC_0', 'relativeHumidity_percent_0', 'temperature_F_0', 'windSpeed_mph_0']

#or use generate using this method
genFeatures= True
if(genFeatures):
    features_to_train = generate_features(hours_to_forecast=hours_to_forecast, allFeats=False, feats_list=['probabilityOfPrecipitation_percent', 'dewpoint_degC', 'relativeHumidity_percent', 'temperature_F', 'windSpeed_mph'])

#the model that will be used
model_type='linear_regression'

#select the model type from model_list
model_list = {
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(),
    'gradient_boosting': GradientBoostingRegressor(),
    'svr': SVR(),
    'k_neighbors': KNeighborsRegressor(),
    'decision_tree': DecisionTreeRegressor()
}

#weather or not to train and evaluate all models in the model list
testAllModels = False

#weather plots should appear or not
plot=True

#features to be plotted
features_to_plot= ['windSpeed_mph_0']

#or plot all features
genFeatures= False
if(genFeatures):
    features_to_plot= generate_features(hours_to_forecast=hours_to_forecast, feats_list=['probabilityOfPrecipitation_percent', 'dewpoint_degC', 'relativeHumidity_percent_0', 'temperature_F_0', 'windSpeed_mph_0'])

#! END CONFIG


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

def dataProcessed(file_path, threshold_minutes=20):
    # get the creation time of the file and current time
    file_creation_time = os.path.getmtime(file_path)
    current_time = time.time()
    # calculate the difference in seconds between the current time and the file creation time and convert to minutes
    time_difference_seconds = current_time - file_creation_time
    time_difference_minutes = time_difference_seconds / 60
    # check if the file has been created in the last threshold_minutes
    if time_difference_minutes <= threshold_minutes:
        return True
    else:
        return False

def plotFeatures(df, target, features):
    logger.info("in plotFeatures")
    
    #get x axis
    xAxis = df['timestamp']
    #plot the target
    plt.plot(xAxis, df[target], label=target, color='red')
    #plot features
    for feature in features:
        plt.plot(xAxis, df[feature], label=feature)
    #show plot
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title('Plot of Features')
    plt.legend()
    plt.savefig('./plots/features_plot.png')
    plt.show()
    return


def train_test_split(df, split):
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
        cur_time = time.time()
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        #log evaluation metrics
        logger.info(f"Model: {model}")
        logger.info(f"Mean Squared Error: {mse}")
        logger.info(f"Root Mean Squared Error: {rmse}")
        logger.info(f"Mean Absolute Error: {mae}")    
        logger.info(f"Mean Absolute Percentage Error: {mape}")
        logger.info(f"R^2 Score: {r2}")

        with open('./model-data/eval.txt', "a") as f:
            f.write(f"Model: {model}\n")
            f.write(f"Date Trained: {cur_time}\n")
            f.write(f"Mean Squared Error: {mse}\n")
            f.write(f"Root Mean Squared Error: {rmse}\n")
            f.write(f"Mean Absolute Error: {mae}\n")
            f.write(f"Mean Absolute Percentage Error: {mape}\n")
            f.write(f"R^2 Error: {r2}\n")
            f.write(f"Features: {features}\n")
            f.write("\n")

    except Exception as e:
        logger.error(f"Error: {str(e)}")

def main():
    logging_setup()
    logger.info("Starting main")    
    
    #process turbine data and get dataframe
    #if the data has been processed recently then dont do it again
    data_path='./turbine-data-processed/finalFrames.csv'
    if (dataProcessed(data_path, threshold_minutes) == True):
        df = pd.read_csv(data_path)
    else:
        df = turbine_util.main(hours_to_forecast)
        
    #train/test split
    train_df, test_df = train_test_split(df, split)

    #plot various features against the target
    if (plot == True):
        plotFeatures(df, target, features_to_plot)

    #train & evaluate the model, training all based on the config
    if(testAllModels == True):
            for model_name, model in model_list.items():
                train_eval_model(train_df, test_df, target, features_to_train, model_list, model_name)
    else:
        model_name=model_type
        train_eval_model(train_df, test_df, target, features_to_train, model_list, model_name)


    #todo: visualize the predictions


if __name__ == "__main__":
    main()
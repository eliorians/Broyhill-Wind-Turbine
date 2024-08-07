
import os
import logging
import time
import traceback
import numpy as np
import pandas as pd

import turbine_util
import plots

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from params import paramList 
from params import modelList
from params import featsList

logger = logging.getLogger('main')

def generate_features(allFeats, hoursOut, feats_list):
    '''
    Generates list of features to be used by the model given the number of hours out to use
    and the list of base features.
    
    ARGS
    allFeats: True to use all features in the base list from params.py
    hoursOut: the number columns to use from each base feature. Use hoursToForcast for all features or 1 for just the _0's
    '''
    features = []
    if allFeats:
        feats_list=featsList
    for number in range(hoursOut):
        for feature in feats_list:
            features.append(f"{feature}_{number}")
    return features

#! ------- CONFIG ------- !#

#The hour that will be forecasted
#NOTE: set threshold minutes to 0 if changed to allow data to reset
hoursToForecast=12

#How often the data should be reprocessed
threshold_minutes=0

#Wether to train and evaluate the model
toTrain=True

#Set the model type from the model list (see params.py for model list)
modelType='linear_regression'

#Column from finalFrames.csv to predict
targetToTrain = 'WTG1_R_InvPwr_kW'

#Columns from finalFrames.csv to be used in training (allFeates=True for all possible features, see base feats list in )
featuresToTrain = generate_features(allFeats=False, hoursOut=1, feats_list=['windSpeed_mph'])

#Size of split in train/test data
split=.2

#Wether to plot stuff
toPlot=True

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

    ARGS
    filepath: relative filepath to the file you want to check
    threshold_minutes: # minutes the file must have been updated ago to return True
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

    ARGS
    df: the dataframe to split
    split: the split proportion
    """
    logger.info("in train_test_split")

    #find the split index
    split_index = int((1 - split) * len(df))
    #split
    train_df, test_df = df.iloc[:split_index], df.iloc[split_index:]

    train_df.to_csv('./model-data/train_df.csv')
    test_df.to_csv('./model-data/test_df.csv')
    return train_df, test_df

def train_eval_model(df, split, target, features, model_name):
    '''
    Grid search to optimize hyperparamerters
    cross validation for fairness
    train model
    test model
    evaluate

    ARGS
    df: the dataframe to pull test and train data from
    split: the split proportion of the train/test data
    target: column to predict
    features: list of columns to use as features
    model_name: model to be trained and evaluated
    '''
    try:
        logger.info("in train_eval_model")

        model = modelList.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in modelList. See options in modelList from params.py")

        #split train and test data into features and target
        train_df, test_df = train_test_split(df, split)
        x_train, y_train = train_df[features], train_df[target]
        x_test, y_test = test_df[features], test_df[target]

        #process for baseline model, returns average target for all points
        if model == 'baseline':
            #get the mean of the target
            target_mean = train_df[target].mean()

            #predict the average value for all instances in the test set
            y_pred = np.full_like(test_df[target], fill_value=target_mean)

        #process for any other selected model. 
        #uses grid search to optimize parameters -> see settings in params.py
        else:
            #get the parameter list from params.py
            param_grid = paramList.get(model_name)
            if param_grid is None:
                raise ValueError(f"Parameter grid for '{model_name}' not found in paramList. See paramList in params.py")
            
            #perform grid search with 5 fold cross validation
            grid_search = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=3, n_jobs=-1)
            
            #fit the grid search to the data
            grid_search.fit(x_train, y_train)

            #get the best model from grid search
            best_model = grid_search.best_estimator_
            
            #cross validation on the best model found to evaluate performance
            scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

            #predict on the test set using the best model
            y_pred = best_model.predict(x_test)

            #training without gridsearch
            # model.fit(x_train, y_train)
            # y_pred = model.predict(x_test)

        #evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_pred)

        #plot the predictions
        plots.plotPrediction(test_df['timestamp'], y_test, y_pred, model_name)

        #log evaluation metrics
        logger.info(f"Model: {model}")
        logger.info(f"RMSE: {rmse}")
        logger.info(f"R^2: {r_squared}")
        if model != 'baseline':
            logger.info(f"Cross-Validation Scores: {scores}")
        logger.info(f"Average Score: {scores.mean()}")
        logger.info(f"Features used: {features}")


        with open('./model-data/eval.txt', "a") as f:
            f.write(f"Model: {model}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"R^2: {r_squared}\n")
            f.write(f"Cross-Validation Scores: {scores}\n")
            f.write(f"Average Score: {scores.mean()}\n")
            f.write(f"Features: {features}\n")
            f.write(f"Hours to Forecast: {hoursToForecast}\n")
            f.write("\n")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()

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
        plots.plotQuantities(df, 'WTG1_R_TurbineState')
        plots.plot_TargetVSActual(df, 'WTG1_R_InvPwr_kW', 'WTG1_R_WindSpeed_mps')
        plots.plot_TargetVSFeature(df, 'WTG1_R_InvPwr_kW', 'windSpeed_mph_0', 'scatter')
        print("target min: "+ str(df[targetToTrain].min()))
        print("target max: "+ str(df[targetToTrain].max()))
        print("target mean: "+ str(df[targetToTrain].mean()))

    #train & evaluate the model
    if toTrain == True:
        train_eval_model(df, split, targetToTrain, featuresToTrain, modelType)

if __name__ == "__main__":
    main()
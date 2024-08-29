
import os
import logging
import time
import traceback
import numpy as np
import pandas as pd

import turbine_util
import plots

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, cross_val_score

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

#The data to use (date is the day collected)
#dataPath = "./turbine-data/frames_11-16-23.csv"
dataPath = "./turbine-data/frames_6-17-24.csv"

#The hour that will be forecasted
#NOTE: set threshold minutes to 0 if changed to allow data to reset
hoursToForecast=12

#How often the data should be reprocessed
threshold_minutes=60

#Wether to train and evaluate the model
toTrain=True

#Set the model type from the model list (see params.py for model list)
modelType= 'polynomial_regression'

#Column from finalFrames.csv to predict
targetToTrain = 'WTG1_R_InvPwr_kW'

#Columns from finalFrames.csv to be used in training (allFeates=True for all possible features. See 'featsList' in params.py for the base features being used)
featuresToTrain = generate_features(allFeats=False, hoursOut=1, feats_list=['windSpeed_mph'])

#Percentage of data that goes to testing (ex: .2 = 80/20 training/testing)
split=.2

#General Plots
toPlot= False
#Prediction Plots (one per fold for nested gridsearch)
toPlotPredictions= True

#The type of validation technique to use. Select from: ['basic', 'gridsearch', 'nested_gridsearch']
validation='gridsearch'

#The # of folds to use for either gridsearch or nested gridsearch
gridsearch_splits = 3
nested_gridsearch_outerfolds = 3
nested_gridsearch_innerfolds = 3

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
    so that the order is preserved and test_df comes after training_df.
    Also, holdout hoursToForecast rows between the training and test sets so that the test set is truly unseen.

    ARGS
    df: the dataframe to split
    split: the split proportion
    """
    logger.info("in train_test_split")

    #find the split index
    split_index = int((1 - split) * len(df))

    #calculate the indices for the holdout and test set
    #have a 'hoursToForecast' size gap between the test and train set to ensure test data is unseen. 
    holdout_start_index = split_index
    test_start_index = split_index + hoursToForecast

    #split the data (training: everything before the holdout; test: everything after holdout)
    train_df = df.iloc[:holdout_start_index]
    test_df = df.iloc[test_start_index:]

    #save files for testing
    holdout_df = df.iloc[holdout_start_index:test_start_index]
    holdout_df.to_csv('./model-data/holdout_df.csv')
    train_df.to_csv('./model-data/train_df.csv')
    test_df.to_csv('./model-data/test_df.csv')

    return train_df, test_df

def train_eval_model(df, split, target, features, model_name):
    '''
    Train and evaluate the mode. Use the config section of this file to select the validation type, model, and many other things.

    ARGS
    df: the dataframe to pull test and train data from
    split: the split proportion of the train/test data
    target: column to predict
    features: list of columns to use as features
    model_name: model to be trained and evaluated
    '''
    try:
        logger.info("in train_eval_model")

        #get the model info from params.py
        model = modelList.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in modelList. See options in modelList from params.py")

        #process for baseline model - returns average target for all points
        if model == 'baseline':
            train_df, test_df = train_test_split(df, split)
            x_train, y_train = train_df[features], train_df[target]
            x_test, y_test = test_df[features], test_df[target]
            
            #get the mean of the target
            target_mean = train_df[target].mean()

            #predict the average value for all instances in the test set
            y_pred = np.full_like(test_df[target], fill_value=target_mean)

            #eval
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r_squared = r2_score(y_test, y_pred)

            #logging
            logger.info(f"RMSE: {rmse}")
            logger.info(f"R^2: {r_squared}")
            with open('./model-data/eval.txt', "a") as f:
                f.write(f"Model: {model}\n")
                f.write(f"RMSE: {rmse}\n")
                f.write(f"R^2: {r_squared}\n")
                f.write(f"\n")
    
            return
        
        #begin timing
        start_time = time.time()

        #process for all other models - varies based on the validation technique selected
        if validation == 'basic':
            #split train and test data into features and target
            train_df, test_df = train_test_split(df, split)
            x_train, y_train = train_df[features], train_df[target]
            x_test, y_test = test_df[features], test_df[target]

            #fit the data
            model.fit(x_train, y_train)
            #predict
            y_pred = model.predict(x_test)

            #evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r_squared = r2_score(y_test, y_pred)

            #plot the predictions
            if toPlotPredictions:
                plots.plotPrediction(test_df['timestamp'], y_test, y_pred, model_name)

        elif validation == 'gridsearch':
            #split full dataframe into train and test data based on the split ratio
            train_df, test_df = train_test_split(df, split)
            #section these out to the feature columns (x) and the target column (y)
            x_train, y_train = train_df[features], train_df[target]
            x_test, y_test = test_df[features], test_df[target]

            #get the parameter list for the specified model from params.py
            param_grid = paramList.get(model_name)
            if param_grid is None:
                raise ValueError(f"Parameter grid for '{model_name}' not found in paramList. See paramList in params.py")
            
            #initialize GridSearchCV with TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=gridsearch_splits)
            grid_search = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=tscv, verbose=0, n_jobs=-1, refit=True)
            
            #perform the gridsearch to find the best model parameters
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters found for {model}: ", grid_search.best_params_)

            #evaluate the model found using gridsearch with a similar technique (essentially testing on seen data) using TimeSeriesSplit again
            cross_scores = cross_val_score(best_model, x_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')

            #predict on the test set (unseen data, not used in gridsearch) using the best model from the gridsearch
            y_pred = best_model.predict(x_test)

            #evaluate the model predictions against the actual
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r_squared = r2_score(y_test, y_pred)

            #plot the predicted y values against the actual y values
            if toPlotPredictions:
                plots.plotPrediction(test_df['timestamp'], y_test, y_pred, model_name)
        
        elif validation == 'nested_gridsearch':
            #TODO nested gridsearch is not correct, need to get a better understanding of this

            #split data into n folds
            outer_cv = TimeSeriesSplit(n_splits=nested_gridsearch_outerfolds)
            outer_scores = []

            #for each fold
            for train_index, test_index in outer_cv.split(df):
                #split train and test data into features and target
                train_df, test_df = df.iloc[train_index], df.iloc[test_index]
                x_train, y_train = train_df[features], train_df[target]
                x_test, y_test = test_df[features], test_df[target]

                #get the parameter list from params.py
                param_grid = paramList.get(model_name)
                if param_grid is None:
                    raise ValueError(f"Parameter grid for '{model_name}' not found in paramList. See paramList in params.py")

                #perform grid search with cross validation on training set for hyperparameter tuning
                inner_cv = TimeSeriesSplit(n_splits=nested_gridsearch_innerfolds)
                grid_search = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=inner_cv, n_jobs=-1, verbose=0)
                grid_search.fit(x_train, y_train)
                
                #get the best model from grid search
                best_model = grid_search.best_estimator_

                #preict on the test set using the best model
                y_pred = best_model.predict(x_test)

                #evaluate the model and append to scores
                mse = mean_squared_error(y_test, y_pred)    
                rmse = np.sqrt(mse)
                r_squared = r2_score(y_test, y_pred)
                outer_scores.append((rmse, r_squared))
                
                logger.info(f"Outer Fold RMSE: {rmse}")
                logger.info(f"Outer Fold R^2: {r_squared}")

                #plot the predictions for each outer fold
                if toPlotPredictions:
                    plots.plotPrediction(test_df['timestamp'], y_test, y_pred, model_name)

            #average the scores across all outer folds
            avg_rmse = np.mean([score[0] for score in outer_scores])
            avg_r_squared = np.mean([score[1] for score in outer_scores])

        else:
            raise ValueError(f"Validation '{validation}' not valid. Set validation in the config to either 'basic', 'gridsearch', or 'nested_gridsearch'.")
        
        #end timing
        end_time = time.time()
        final_time = end_time - start_time

        #log evaluation metrics
        logger.info(f"Model: {modelType}")
        logger.info(f"Validation Technique: {validation}")
        logger.info(f"Training Time (seconds): {final_time}")
        if validation == 'basic':
            logger.info(f"RMSE: {rmse}")
            logger.info(f"R^2: {r_squared}")
        if validation == 'gridsearch':
            logger.info(f"RMSE: {rmse}")
            logger.info(f"R^2: {r_squared}")
            logger.info(f"Average Cross-Validation Score: {cross_scores.mean()}")
        if validation == 'nested_gridsearch':
            logger.info(f"Average RMSE: {avg_rmse}")
            logger.info(f"Average R^2: {avg_r_squared}")
        logger.info(f"Features used: {features}")

        with open('./model-data/eval.txt', "a") as f:
            f.write(f"Model: {modelType}\n")
            f.write(f"Validation Technique: {validation}\n")
            f.write(f"Training Time (seconds): {final_time}\n")
            if validation == 'basic':
                f.write(f"RMSE: {rmse}\n")
                f.write(f"R^2: {r_squared}\n")
            if validation == 'gridsearch':
                f.write(f"RMSE: {rmse}\n")
                f.write(f"R^2: {r_squared}\n")
                f.write(f"Average Cross-Validation Score: {cross_scores.mean()}\n")
                f.write(f"N-Splits: {gridsearch_splits}\n")
            if validation == 'nested_gridsearch':
                f.write(f"Average RMSE: {avg_rmse}\n")
                f.write(f"Average R^2: {avg_r_squared}\n")
                f.write(f"Outer Folds: {nested_gridsearch_outerfolds}\n")
                f.write(f"Inner Folds: {nested_gridsearch_innerfolds}\n")
            f.write(f"Features: {features}\n")
            f.write(f"Hours to Forecast: {hoursToForecast}\n")
            f.write(f"Data used: {dataPath}\n")
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
        df = turbine_util.main(dataPath, hoursToForecast)

    #plotting stuff
    if toPlot == True:
        #plot quantities of turbine states
        plots.plotQuantities(df, 'WTG1_R_TurbineState')
        #plot the power output against the windspeed measured at the turbine
        plots.plot_TargetVSActual(df, 'WTG1_R_InvPwr_kW', 'WTG1_R_WindSpeed_mps')
        #plot the power output against the windspeed measured by the forecast
        plots.plot_TargetVSFeature(df, 'WTG1_R_InvPwr_kW', 'windSpeed_mph_0', 'scatter')
        print("target min: "+ str(df[targetToTrain].min()))
        print("target max: "+ str(df[targetToTrain].max()))
        print("target mean: "+ str(df[targetToTrain].mean()))

    #train & evaluate the model
    if toTrain == True:
        train_eval_model(df, split, targetToTrain, featuresToTrain, modelType)

if __name__ == "__main__":
    main()
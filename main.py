
import os
import logging
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, f_regression
from sklearn.pipeline import Pipeline

import turbine_util
import plots

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate

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
    hoursOut: How many hours out of forecast to use for the model. Use hoursToForcast for all features or 1 for just the _0's (aka the forecast for that hour).
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
#max = 154
hoursToForecast=6

#How often the data should be reprocessed
threshold_minutes=0

#Wether to train and evaluate the model
toTrain=False

#Percentage of data that goes to testing (ex: .2 = 80/20 training/testing)
split=.2

#Set the model type from the model list: (more details in params.py)
# ['baseline', 'linear_regression','random_forest', 'bagging 'polynomial_regression', 'decision_tree', 'gradient_boosted_reg', 'ridge_cv', 'lasso_cv', 'elastic_net_cv', 'svr', 'kernal_ridge', 'ada_boost', 'mlp_regressor', 'gaussian']
modelType= 'polynomial_regression'

#Column from finalFrames.csv to predict
targetToTrain = 'WTG1_R_InvPwr_kW'

#Columns from finalFrames.csv to be used in training (allFeates=True for all possible features. See 'featsList' in params.py for the base features being used)
#use hoursOut= 1 for only the forecast for that hour
#use hourOut= hoursToForecast to use all forecasted values from hoursToForecast hours before.
featuresToTrain = generate_features(allFeats=True, hoursOut=1, feats_list=['windSpeed_knots'])

#number of inner and outer splits for nested cross validation
nested_outersplits = 5
nested_innersplits = 4

#Use feature selection (give all features, or as many to test. no feature selection for basic validation)
feature_selection = True

#Type of feature selection to use, options are ['sfs', 'kbest']
#sfs = Sequential Feature Selection / kbest = Select K Best
feature_type = 'sfs'
#Feature selection Types (for k best, see within train_eval where it is called)
feature_selection_splits = TimeSeriesSplit(n_splits=5)

#General Plots
toPlot= True
#Prediction Plots (one per fold for nested gridsearch)
toPlotPredictions= True


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
        logger.info(f"in train_eval_model with {modelType}, with feature selection(T/F): {feature_selection}")

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
                    
        #get the parameters for gridsearch
        param_grid = paramList.get(model_name)
        if param_grid is None:
            raise ValueError(f"Parameter grid for '{model_name}' not found in paramList. See paramList in params.py")
        
        #specify columns from the dataframe to use
        x, y = df[features], df[target]

        # Feature Selection
        if (feature_selection == True):
            logger.info(f"Setting up feature selection of type {feature_type}...")

            # Determine type of feature selection (SFS or KBest)
            if (feature_type == 'sfs'):

                #create feature selector to occur in inner loop 
                feature_selector = SequentialFeatureSelector(
                        n_features_to_select='auto',        # Automatically selects the best number of features
                        direction='forward',                # Use 'forward' for SFS or 'backward' for SBS
                        cv=feature_selection_splits,        # Time series split
                        scoring='neg_mean_squared_error',   # Choose scoring metric (e.g., MSE)
                        estimator=model,
                        n_jobs=-1)
                
                # Create a pipeline - first the feature selection, then the model with grid search
                pipeline = Pipeline([
                    ('feature_selection', feature_selector),
                    ('model', model)
                ])

                # Update the param grid to include a prefix of model_ so it knows which step in the pipeline to be used at
                model_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
                feature_selection_param_grid = {f'feature_selection__{key}': value for key, value in param_grid.items() if key in ['n_features_to_select', 'direction', 'cv', 'scoring', 'estimator', 'n_jobs']}
                param_grid = {**feature_selection_param_grid, **model_param_grid}
            
            elif (feature_type == 'kbest'):
                

                # Create feature selector using SelectKBest
                feature_selector = SelectKBest() 
                feature_selection_param_grid = {'feature_selection__k': [1, 3, 5, 7], 'feature_selection__score_func' : [f_regression]}

                # Create a pipeline - first the feature selection, then the model with grid search
                pipeline = Pipeline([
                    ('feature_selection', feature_selector),
                    ('model', model)
                ])

                # Update the param grid to include a prefix of model_ and feature_selection_
                model_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
                param_grid = {**feature_selection_param_grid, **model_param_grid}
            else:
                raise ValueError(f"Feature type '{feature_type}' not valid. Set feature type in the config to either 'sfs' or 'kbest'.")

        else:
            # Create pipline - no feature selection, so make pipeline of JUST the model
            pipeline = Pipeline([
                ('model', model)
            ])
             # Update the param grid to include a prefix of model_
            model_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
            param_grid = {**model_param_grid}

        # Define outer and inner cross validation techniques (TimeSeriesSplit technique to uphold temporal order)
        outer_tscv = TimeSeriesSplit(n_splits=nested_outersplits)
        inner_tscv = TimeSeriesSplit(n_splits=nested_innersplits)

        # Initialize GridSearchCV for the inner loop
        logger.info(f"Setting up gridsearch for inner loop...")
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_tscv, scoring='neg_root_mean_squared_error', n_jobs=-1, refit=True)

        # Scoring metrics to use
        scoring = {'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}

        # Perform nested cross-validation
        # - estimator=grid_search: Pass the GridSearchCV object as the estimator, which will make the 'inner splits' use GridSearch to tune hyperparameters
        # - cv=outer_tscv: The outer cross-validation strategy to split data into training and test sets
        # - return_train_score=False: Only return the test scores, not the training scores
        logger.info(f"Running nested cross validation...")
        nested_scores = cross_validate(estimator=grid_search, X=x, y=y, cv=outer_tscv, scoring=scoring, return_train_score=False)

        #get the test scores from testing with outer loop
        avg_rmse = -nested_scores['test_rmse'].mean()
        avg_r_squared = nested_scores['test_r2'].mean()
        std_rmse = nested_scores['test_rmse'].std()
        std_r_squared = nested_scores['test_r2'].std()

        # Perform GridSearchCV on the entire dataset to get the best parameters
        logger.info(f"Running gridsearch on the full dataset to grab best parameters and features...")

        # Split full dataframe into train and test data based on the split ratio
        train_df, test_df = train_test_split(df, split)
        #section these out to the feature columns (x) and the target column (y)
        x_train, y_train = train_df[features], train_df[target]
        x_test, y_test = test_df[features], test_df[target]

        # Set up the gridsearch and train
        grid_search_full = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_tscv, scoring='neg_root_mean_squared_error', n_jobs=-1, refit=True)
        grid_search_full.fit(x_train, y_train)

        # Save the results from training (model, parameters, features)
        best_model  = grid_search_full.best_estimator_
        best_params = grid_search_full.best_params_
        if 'feature_selection' in best_model.named_steps:
            feature_selector = best_model.named_steps['feature_selection']
            selected_features_mask = feature_selector.get_support()
            selected_feature_indices = [i for i, selected in enumerate(selected_features_mask) if selected]
            selected_features = [features[i] for i in selected_feature_indices]

        # Predict on the test set (unseen data, not used in gridsearch) using the best model from the gridsearch
        logger.info(f"Predicting...")
        y_pred = best_model.predict(x_test)

        # Evaluate the model predictions against the actual
        logger.info(f"Evaluating Predictions...")
        prediction_mse = mean_squared_error(y_test, y_pred)
        prediction_rmse = np.sqrt(prediction_mse)
        prediction_r_squared = r2_score(y_test, y_pred)

        # Plot the predicted y values against the actual y values
        if toPlotPredictions:
            logger.info(f"Plotting...")
            plots.plotPrediction(test_df['timestamp'], y_test, y_pred, model_name, hoursToForecast)
        
        #end timing
        end_time = time.time()
        final_time = end_time - start_time

        #log evaluation metrics (in console and eval.txt)
        logger.info(f"Model: {modelType}")
        logger.info(f'Hours Out: {hoursToForecast}')
        logger.info(f"Average RMSE: {avg_rmse}")
        logger.info(f"Average R^2: {avg_r_squared}")
        # logger.info(f"Std RMSE: {std_rmse}")
        # logger.info(f"Std R^2: {std_r_squared}")
        logger.info(f"Predicted RMSE: {prediction_rmse}")
        logger.info(f"Predicted R^2: {prediction_r_squared}")
        logger.info(f"Features given: {features}")
        if (feature_selection == True):
                logger.info(f'Feature Selection Type: {feature_type}')
                logger.info(f"Selected Features: {selected_features}")
        logger.info(f"Best Parameters: {best_params}")
        logger.info(f"Training Time (seconds): {final_time}")
        logger.info(f"Outer N-Splits: {nested_outersplits}")
        logger.info(f"Inner N-Splits: {nested_innersplits}")
        
        with open('./model-data/eval.txt', "a") as f:
            f.write(f"Model: {modelType}\n")
            f.write(f'Hours Out: {hoursToForecast}\n')
            f.write(f"Average RMSE: {avg_rmse}\n")
            f.write(f"Average R^2: {avg_r_squared}\n")
            # f.write(f"Std RMSE: {std_rmse}\n")
            # f.write(f"Std R^2: {std_r_squared}\n")
            f.write(f"Predicted RMSE: {prediction_rmse}\n")
            f.write(f"Predicted R^2: {prediction_r_squared}\n")
            f.write(f"Features given: {features}\n")
            if (feature_selection == True):
                f.write(f'Feature Selection Type: {feature_type}\n')
                f.write(f"Selected Features: {selected_features}\n")
            f.write(f"Best Parameters: {best_params}\n")
            f.write(f"Training Time (seconds): {final_time}\n")
            f.write(f"Outer N-Splits: {nested_outersplits}\n")
            f.write(f"Inner N-Splits: {nested_innersplits}\n")
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

        #plot quanitity of turbine states being used
        plots.plotQuantities(df, 'WTG1_R_TurbineState')

        #plot the distribution of windspeed (knots converted or original mph)
        #columnName = 'windSpeed_mph'
        #columnName = 'windSpeed_knots'
        #plots.plot_windspeed_distribution(columnName)

        #plot target against feature collected from the turbine data
        #plots.plot_TargetVSActual(df, 'WTG1_R_InvPwr_kW', 'WTG1_R_WindSpeed_mps')

        #plot target against  feature collected from forecast data
        #plots.plot_TargetVSForecasted(df, 'WTG1_R_InvPwr_kW', 'windSpeed_knots_0')

        #plot other features
        #plots.plot_TargetVSFeature(df, 'WTG1_R_InvPwr_kW', 'windSpeed_knots', plotType='reg')

        #print target distibution
        #print("target min: "+ str(df[targetToTrain].min()))
        #print("target max: "+ str(df[targetToTrain].max()))
        #print("target mean: "+ str(df[targetToTrain].mean()))

    #train & evaluate the model
    if toTrain == True:
        train_eval_model(df, split, targetToTrain, featuresToTrain, modelType)

if __name__ == "__main__":
    main()
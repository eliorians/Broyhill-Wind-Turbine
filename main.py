
import os
import logging
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
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
    hoursOut: How many hours out of forecast to use for the model. Use hoursToForcast for all features or 1 for just the _0's (aka the forecast for that hour). Max is hoursToForecast-1
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
toTrain=True

#Set the model type from the model list: (more details in params.py)
# ['baseline', 'linear_regression','random_forest', 'polynomial_regression', 'decision_tree', 'gradient_boosted_reg', 'ridge_cv', 'lasso_cv', 'elastic_net_cv', 'svr', 'kernal_ridge', 'ada_booster]
modelType= 'linear_regression'

#Column from finalFrames.csv to predict
targetToTrain = 'WTG1_R_InvPwr_kW'

#Columns from finalFrames.csv to be used in training (allFeates=True for all possible features. See 'featsList' in params.py for the base features being used)
#use hoursOut= 1 for only the forecast for that hour
#use hourOut= hoursToForecast-1 to use all forecasted vallues from hoursToForecast hours before.
featuresToTrain = generate_features(allFeats=True, hoursOut=1, feats_list=['windSpeed_knots'])

#Use feature selection (give all features, or as many to test)
feature_selection = True
feature_selection_splits = TimeSeriesSplit(n_splits=5)

#Percentage of data that goes to testing (ex: .2 = 80/20 training/testing)
split=.2

#General Plots
toPlot= False
#Prediction Plots (one per fold for nested gridsearch)
toPlotPredictions= True

#The type of validation technique to use.
# ['basic', 'gridsearch', 'nested_crossval']
validation='nested_crossval'

#Wether to run a full gridsearch within nested crossvalidation to get the best parameters and features used. Increases runtime.
getParameters = False

#number of splits for grisearch
gridsearch_splits = 5
#number of inner and outer splits for nested cross validation
nested_outersplits = 5
nested_innersplits = 4

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
        logger.info(f"in train_eval_model using: {validation} with {modelType}, with feature selection(T/F): {feature_selection}")

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

            # Apply feature selection
            if (feature_selection == True):
                logger.info(f"Beginning feature selection...")
                sfs = SequentialFeatureSelector(
                    n_features_to_select='auto',        # Automatically selects the best number of features
                    direction='forward',                # Use 'forward' for SFS or 'backward' for SBS
                    cv=feature_selection_splits,        # Time series split
                    scoring='neg_mean_squared_error',
                    estimator=model,
                    n_jobs=-1) 
                
                # Fit the feature selector on the training data
                sfs.fit(x_train, y_train)

                # Get the selected features and trim training data to these features
                selected_features = [f for f, support in zip(features, sfs.get_support()) if support]
                x_train = x_train[selected_features]
                x_test = x_test[selected_features]

            #fit the data
            logger.info(f"Fitting data...")
            model.fit(x_train, y_train)

            #predict
            logger.info(f"Predicting...")
            y_pred = model.predict(x_test)

            #evaluate the model
            logger.info(f"Evaluating...")
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r_squared = r2_score(y_test, y_pred)

            #plot the predictions
            if toPlotPredictions:
                logger.info(f"Plotting...")
                plots.plotPrediction(test_df['timestamp'], y_test, y_pred, model_name)

        elif validation == 'gridsearch':
            #split full dataframe into train and test data based on the split ratio
            train_df, test_df = train_test_split(df, split)
            #section these out to the feature columns (x) and the target column (y)
            x_train, y_train = train_df[features], train_df[target]
            x_test, y_test = test_df[features], test_df[target]

            # Apply feature selection
            if (feature_selection == True):
                logger.info(f"Beginning feature selection...")
                sfs = SequentialFeatureSelector(
                    n_features_to_select='auto',        # Automatically selects the best number of features
                    direction='forward',                # Use 'forward' for SFS or 'backward' for SBS
                    cv=feature_selection_splits,        # Time series split
                    scoring='neg_mean_squared_error',
                    estimator=model,
                    n_jobs=-1) 
                
                # Fit the feature selector on the training data
                sfs.fit(x_train, y_train)

                # Get the selected features and trim training data to these features
                selected_features = [f for f, support in zip(features, sfs.get_support()) if support]
                x_train = x_train[selected_features]
                x_test = x_test[selected_features]

            #get the parameter list for the specified model from params.py
            param_grid = paramList.get(model_name)
            if param_grid is None:
                raise ValueError(f"Parameter grid for '{model_name}' not found in paramList. See paramList in params.py")
            
            #initialize GridSearchCV with TimeSeriesSplit for cross-validation
            logger.info(f"Starting GridSearch...")
            tscv = TimeSeriesSplit(n_splits=gridsearch_splits)
            grid_search = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=tscv, verbose=0, n_jobs=-1, refit=True)
            
            #perform the gridsearch to find the best model parameters
            grid_search.fit(x_train, y_train)

            #save the results from training
            best_model  = grid_search.best_estimator_
            best_params = grid_search.best_params_

            #predict on the test set (unseen data, not used in gridsearch) using the best model from the gridsearch
            logger.info(f"Predicintg...")
            y_pred = best_model.predict(x_test)

            #evaluate the model predictions against the actual
            logger.info(f"Evaluating...")
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r_squared = r2_score(y_test, y_pred)

            #plot the predicted y values against the actual y values
            if toPlotPredictions:
                logger.info(f"Plotting...")
                plots.plotPrediction(test_df['timestamp'], y_test, y_pred, model_name)
        
        elif validation == 'nested_crossval':
            
            #get the parameters for gridsearch
            param_grid = paramList.get(model_name)
            if param_grid is None:
                raise ValueError(f"Parameter grid for '{model_name}' not found in paramList. See paramList in params.py")
            
            #specify columns from the dataframe to use
            x, y = df[features], df[target]

            if (feature_selection == True):
                logger.info(f"Setting up feature selection and inner loop pipeline...")
                # Create feature selector to occur in inner loop 
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

                #update the param grid to include a prefix of model_ so it knows which step in the pipeline to be used at
                model_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
                feature_selection_param_grid = {f'feature_selection__{key}': value for key, value in param_grid.items() if key in ['n_features_to_select', 'direction', 'cv', 'scoring', 'estimator', 'n_jobs']}
                param_grid = {**feature_selection_param_grid, **model_param_grid}
            else:
                # Create pipline - no feature selection, so make pipeline of JUST the model
                pipeline = Pipeline([
                    ('model', model)
                ])

            # Define outer and inner cross validation techniques (TimeSeriesSplit technique to uphold temporal order)
            outer_tscv = TimeSeriesSplit(n_splits=nested_outersplits)
            inner_tscv = TimeSeriesSplit(n_splits=nested_innersplits)

            # Initialize GridSearchCV for the inner loop
            logger.info(f"Setting up gridsearch for inner loop...")
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_tscv, scoring='neg_root_mean_squared_error', n_jobs=-1, refit=True)

            #scoring metrics to use
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

            if (getParameters == True):
                # Perform GridSearchCV on the entire dataset to get the best parameters
                logger.info(f"Running gridsearch on the full dataset to grab best parameters and features...")
                grid_search_full = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_tscv, scoring='neg_root_mean_squared_error', n_jobs=-1, refit=True)
                grid_search_full.fit(x, y)
                best_params = grid_search_full.best_params_

                # Pull out the features selected
                best_pipeline = grid_search_full.best_estimator_
                if 'feature_selection' in best_pipeline.named_steps:
                    feature_selector = best_pipeline.named_steps['feature_selection']
                    selected_features_mask = feature_selector.get_support()
                    selected_feature_indices = [i for i, selected in enumerate(selected_features_mask) if selected]
                    selected_features = [features[i] for i in selected_feature_indices]
                        
        else:
            raise ValueError(f"Validation '{validation}' not valid. Set validation in the config to either 'basic', 'gridsearch', or 'nested_crossval'.")
        
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
            if (feature_selection == True):
                    logger.info(f"Selected Features: {selected_features}")
        if validation == 'gridsearch':
            logger.info(f"RMSE: {rmse}")
            logger.info(f"R^2: {r_squared}")
            logger.info(f"N-Splits: {gridsearch_splits}")
            logger.info(f"Best Parameters: {best_params}")
            if (feature_selection == True):
                    logger.info(f"Selected Features: {selected_features}")
        if validation == 'nested_crossval':
            logger.info(f"Average RMSE: {avg_rmse}")
            logger.info(f"Average R^2: {avg_r_squared}")
            logger.info(f"Std RMSE: {std_rmse}")
            logger.info(f"Std R^2: {std_r_squared}")
            logger.info(f"Outer N-Splits: {nested_outersplits}")
            logger.info(f"Inner N-Splits: {nested_innersplits}")
            if (getParameters):
                logger.info(f"Best Parameters: {best_params}")
                if (feature_selection == True):
                    logger.info(f"Selected Features: {selected_features}")
        logger.info(f"Features given: {features}")
        
        with open('./model-data/eval.txt', "a") as f:
            f.write(f"Model: {modelType}\n")
            f.write(f"Validation Technique: {validation}\n")
            f.write(f"Training Time (seconds): {final_time}\n")
            if validation == 'basic':
                f.write(f"RMSE: {rmse}\n")
                f.write(f"R^2: {r_squared}\n")
                if (feature_selection == True):
                        f.write(f"Selected Features: {selected_features}\n")
            if validation == 'gridsearch':
                f.write(f"RMSE: {rmse}\n")
                f.write(f"R^2: {r_squared}\n")
                f.write(f"N-Splits: {gridsearch_splits}\n")
                f.write(f"Best Parameters: {best_params}\n")
                if (feature_selection == True):
                        f.write(f"Selected Features: {selected_features}\n")
            if validation == 'nested_crossval':
                f.write(f"Average RMSE: {avg_rmse}\n")
                f.write(f"Average R^2: {avg_r_squared}\n")
                f.write(f"Std RMSE: {std_rmse}\n")
                f.write(f"Std R^2: {std_r_squared}\n")
                f.write(f"Outer N-Splits: {nested_outersplits}\n")
                f.write(f"Inner N-Splits: {nested_innersplits}\n")
                if (getParameters):
                    f.write(f"Best Parameters: {best_params}\n")
                    if (feature_selection == True):
                        f.write(f"Selected Features: {selected_features}\n")
            f.write(f"Features given: {features}\n")
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

        #plot quanitity of turbine states being used
        #plots.plotQuantities(df, 'WTG1_R_TurbineState')

        #plot the distribution of windspeed (knots converted or original mph)
        columnName = 'windSpeed_mph'
        #columnName = 'windSpeed_mph'
        plots.plot_windspeed_distribution(columnName)

        #plot target against feature collected from the turbine data
        plots.plot_TargetVSActual(df, 'WTG1_R_InvPwr_kW', 'WTG1_R_WindSpeed_mps')

        #plot target against  feature collected from forecast data
        plots.plot_TargetVSForecasted(df, 'WTG1_R_InvPwr_kW', 'windSpeed_knots_0')

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
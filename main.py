
import os
import logging
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import turbine_util


logger = logging.getLogger('main')

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

def plotFeatures(df):
    logger.info("in plotFeatures")
    warnings.filterwarnings("ignore")
    
    xAxis = df['timestamp']
    featuresToPlot=['WTG1_R_InvPwr_kW', 'windSpeed_mph_0', 'windSpeed_mph_1', 'windSpeed_mph_2']

    for feature in featuresToPlot:
        plt.plot(xAxis, df[feature], label=feature)

    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title('Plot of Features')
    plt.legend()
    plt.savefig('./plots/features_plot.png')
    #plt.show()
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


def train_linear_regression(train_df, test_df):
    logger.info("in train_linear_regression")

    # select features for training and target variable
    features = ['windSpeed_mph_0']
    target = 'WTG1_R_InvPwr_kW'

    # split train and test data into features and target
    x_train, y_train = train_df[features], train_df[target]
    x_test, y_test = test_df[features], test_df[target]

    # initialize and train the linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # predict on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Model: {model}")
    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R^2 Score: {r2}")

def main():
    logging_setup()
    logger.info("Starting main")
    
    #process turbine data and get dataframe
    #if the data has been processed recently then dont do it again
    data_path='./turbine-data-processed/finalFrames.csv'
    if (dataProcessed(data_path, threshold_minutes=20) == True):
        df = pd.read_csv(data_path)
    else:
        df = turbine_util.main()
        

    #plot various features against the target (WTG1_R_InvPwr_kW)
    plotFeatures(df)

    #train/test split
    train_df, test_df = train_test_split(df, split=.2)

    #train & evaluate the model
    train_linear_regression(train_df, test_df)


if __name__ == "__main__":
    main()
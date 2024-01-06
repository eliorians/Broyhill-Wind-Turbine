
import os
import pandas as pd
import logging

import forecast_util
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

def main():
    
    #process forecast data
    forecast_util.main

    #process turbine data
    turbine_util.main
    df = pd.read_csv('./turbine-data-processed/cleanedFrames.csv')
    
    #todo setup tensorflow time series model


if __name__ == "__main__":
    main()
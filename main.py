
import os
import pandas as pd
import logging

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
    
    #process turbine data and get dataframe
    df = turbine_util.main

    #todo split off, saving some to be used as validation
    # train_size = int(len(df) * 0.8)
    # train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

    



if __name__ == "__main__":
    main()

import os
import pandas as pd
import logging

logger = logging.getLogger('turbine_util')

dataPath = ""

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

def main():

    logging_setup()
    logger.info("Running turbine-util...")

    #read tubrine data into dataframe
    df = pd.read_csv('/turbine-data/frames.csv')
    print(df)

if __name__ == "__main__":
    main()
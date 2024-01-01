
import os
import pandas as pd
import logging

logger = logging.getLogger('weather_util')

def logging_setup():
    # Create a "logs" directory if it doesn't exist
    logs_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_directory, exist_ok=True)
    # Set up logging to a file in the "logs" directory
    log_file = os.path.join(logs_directory, 'weather_util.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Send log messages to the console
            logging.FileHandler(log_file)  # Save log messages to a file in the "logs" directory
        ])

def cleanWeatherData(df):

    return df

def main():

    logging_setup()
    logger.info("Running weather-util...")

    df = cleanWeatherData(df)

    

if __name__ == "__main__":
    main()
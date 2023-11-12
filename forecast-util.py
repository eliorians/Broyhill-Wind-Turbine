
import pandas as pd
import logging
import json
import os
import re

json_folder = './forecast-data/'
logger = logging.getLogger('forecast_util')
logger.info("Converting forecast data...")

def logging_setup():
    # Create a "logs" directory if it doesn't exist
    logs_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_directory, exist_ok=True)
    # Set up logging to a file in the "logs" directory
    log_file = os.path.join(logs_directory, 'forecast_util.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Send log messages to the console
            logging.FileHandler(log_file)  # Save log messages to a file in the "logs" directory
        ])

def convert_filepath(filepath):
    # extract the directory path and filename from the original filepath
    directory, filename = os.path.split(filepath)
    # remove the seconds, and change the extension to .csv
    base_name = re.sub(r'_\d+\.json$', '.csv', filename)
    # Construct the new directory path with "-processed" and the new filename
    new_directory = os.path.join(directory, 'forecast-data-processed')
    new_filepath = os.path.join(new_directory, base_name)
    return new_filepath

def process_json_file(filepath):

    #open json file
    with open(filepath, 'r') as json_file:
        forecast_dict = json.load(json_file)

    #turn to dictionary, then dataframe
    periods = forecast_dict.get("properties", {}).get("periods", [])
    df = pd.DataFrame(periods)

    #save to csv
    filepath = convert_filepath(filepath)
    df.to_csv(filepath)
    logger.info(f"Data successfully saved to {filepath}")

              
def main():

    logging_setup()

    #iterate through all files in the forecast-data folder
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            # Construct the full path to the JSON file
            filepath = os.path.join(json_folder, filename)
            # Process the JSON file
            process_json_file(filepath)

if __name__ == '__main__':
    main()
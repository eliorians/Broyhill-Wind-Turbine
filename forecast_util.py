
import pandas as pd
import logging
import json
import os
import re
import time

json_folder = './forecast-data/'
logger = logging.getLogger('forecast_util')

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

#converts all forecast json files to csv files
def jsonToCSV():
    start_time = time.time()
    file_count = 0

    #iterate through all files in the forecast-data folder
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            # Construct the full path to the JSON file
            filepath = os.path.join(json_folder, filename)
            # Process the JSON file
            process_json_file(filepath)
            file_count = file_count + 1

    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Program finished. Processed {file_count} in {runtime:.2f} seconds")

#process individual json file
def process_json_file(filepath):

    #open json file
    with open(filepath, 'r') as json_file:
        forecast_dict = json.load(json_file)

    #turn to dictionary, then dataframe
    periods = forecast_dict.get("properties", {}).get("periods", [])
    df = pd.DataFrame(periods)

    #save to csv
    filepath = process_filepath(filepath)
    df.to_csv(filepath)
    logger.info(f"Data successfully saved to {filepath}")

#converts forecast path to csv path
def process_filepath(filepath):
    # extract the directory path and filename from the original filepath
    directory, filename = os.path.split(filepath)
    # remove the seconds, and change the extension to .csv
    base_name = re.sub(r'(_|\d+)\.json$', '.csv', filename)
    base_name = base_name.rsplit('-', 1)[0] + '.csv'
    # Construct the new directory path with "-processed" and the new filename
    new_directory = os.path.join(os.path.dirname(directory), 'forecast-data-processed')
    new_filepath = os.path.join(new_directory, base_name)
    return new_filepath

def main():

    logging_setup()
    logger.info("Running forecast_util...")
    
    #convert all forecast files to json
    jsonToCSV()

    

if __name__ == '__main__':
    main()
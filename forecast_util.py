
import pandas as pd
import logging
import json
import os
import re
import time

logger = logging.getLogger('forecast_util')

column_types = {
    'temperature_F'                     : int,
    'windSpeed_mph'                     : int,
    'windDirection'                     : str,
    'shortForecast'                     : str,
    'probabilityOfPrecipitation_percent': int,
    'dewpoint_degC'                     : float,
    'relativeHumidity_percent'          : int
}

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

#extracts json data
def extractJson(data):
    try:
        return data.get('value')
    except (json.JSONDecodeError, AttributeError):
        return None


#converts forecast path to csv path
def getNewFilepath(filepath):
    # extract the directory path and filename from the original filepath
    directory, filename = os.path.split(filepath)
    # remove the seconds, and change the extension to .csv
    base_name = re.sub(r'(_|\d+)\.json$', '.csv', filename)
    base_name = base_name.rsplit('-', 1)[0] + '.csv'
    # Construct the new directory path with "-processed" and the new filename
    new_directory = os.path.join(os.path.dirname(directory), 'forecast-data-processed')
    new_filepath = os.path.join(new_directory, base_name)
    return new_filepath

#process each forecast data file
def cleanForecastData(filepath):

    #open json file
    with open(filepath, 'r') as json_file:
        forecast_dict = json.load(json_file)

    #turn to dictionary, then dataframe
    periods = forecast_dict.get("properties", {}).get("periods", [])
    df = pd.DataFrame(periods)

    #extract json values and add unit to column headers
    df['probabilityOfPrecipitation_percent'] = df['probabilityOfPrecipitation'].apply(extractJson)
    df['dewpoint_degC'] = df['dewpoint'].apply(extractJson)
    df['relativeHumidity_percent'] = df['relativeHumidity'].apply(extractJson)
    df['temperature_F'] = df['temperature']
    df['windSpeed_mph'] = df['windSpeed'].str.replace(' mph', '')

    #deal with null values
    df['relativeHumidity_percent'] = df['relativeHumidity_percent'].interpolate()

    #set column types, only keep one and make it the timestampt
    df['timestamp'] = pd.to_datetime(df['startTime'])
    df = df.astype(column_types)

        #drop uneeded columns
    columns_to_drop = ['number', 'name', 'isDaytime', 'temperatureUnit', 'temperature', 'temperatureTrend', 'icon', 'detailedForecast', 'probabilityOfPrecipitation', 'dewpoint', 'relativeHumidity', 'windSpeed', 'startTime', 'endTime', 'shortForecast', ]
    df.drop(columns=columns_to_drop, inplace=True)
    
    #save to csv
    filepath = getNewFilepath(filepath)
    df.to_csv(filepath)
    logger.info(f"Data successfully saved to {filepath}")

def main():

    logging_setup()
    logger.info("Running forecast_util...")

    #tracking time to process all files    
    start_time = time.time()
    file_count = 0

    #iterate through all files in the forecast-data folder and clean each
    json_folder = './forecast-data/'
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            # Construct the path to the JSON file
            filepath = os.path.join(json_folder, filename)
            # Do the work
            cleanForecastData(filepath)
            file_count = file_count + 1

    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Program finished. Processed {file_count} in {runtime:.2f} seconds")

if __name__ == '__main__':
    main()
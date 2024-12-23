
import math
import traceback
import warnings
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
    'windDirection_x'                   : float,
    'windDirection_y'                   : float,
    'probabilityOfPrecipitation_percent': int,
    'dewpoint_degC'                     : float,
    'relativeHumidity_percent'          : int
}

#map the wind direction to vectors around a circle
direction_mapping = {
    'N'     : (0, 1),
    'NE'    : (math.sqrt(2)/2, math.sqrt(2)/2),
    'E'     : (1, 0),
    'SE'    : (math.sqrt(2)/2, -math.sqrt(2)/2),
    'S'     : (0, -1),
    'SW'    : (-math.sqrt(2)/2, -math.sqrt(2)/2),
    'W'     : (-1, 0),
    'NW'    : (-math.sqrt(2)/2, math.sqrt(2)/2),
    ''      : (0,0) # wind speed is 0, so wind direction is ''
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
    '''
    given a json value, extract it

    ARGS
    data: json data

    RETURNS
    the value within the json
    '''
    try:
        return data.get('value')
    except (json.JSONDecodeError, AttributeError):
        return None

def getNewFilepath(filepath):
    '''
    Converts json filepath to the processed csv filepath. Drops seconds to ensure consistent hourly naming.

    ARGS
    filepath: the current name of the forecast file

    RETURNS
    the updated name for the .csv file.
    '''
    # extract the directory path and filename from the original filepath
    directory, filename = os.path.split(filepath)
    # remove the seconds, and change the extension to .csv
    base_name = re.sub(r'(_|\d+)\.json$', '.csv', filename)
    base_name = base_name.rsplit('-', 1)[0] + '.csv'
    # Construct the new directory path with "-processed" and the new filename
    new_directory = os.path.join(os.path.dirname(directory), 'forecast-data-processed')
    new_filepath = os.path.join(new_directory, base_name)
    return new_filepath

def cleanForecastData(filepath):
    '''
    Process each forecast data file.

    ARGS
    filepath: the path to the forecast file to process
    '''

    #open json file
    with open(filepath, 'r') as json_file:
        forecast_dict = json.load(json_file)

    #error catching
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)
        try:
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
            #currently only found in relativeHumidity_percent column
            df['relativeHumidity_percent'] = df['relativeHumidity_percent'].interpolate()
            
            #convert timestamp to UTC timezone, this makes for easier manioulation by elimating daylight savings
            #use the endtime as the timestamp since this lines up with the turbine data
            df['timestamp'] = pd.to_datetime(df['endTime'], utc=True)

            #make wind direction numeric
            df['windDirection'] = [direction_mapping[direction] for direction in df['windDirection']]
            #split into two columns
            df[['windDirection_x', 'windDirection_y']] = pd.DataFrame(df['windDirection'].tolist(), index=df.index)

            #create new column 'windspeed_knots' by converting 'windSpeed_mph'
            df = convert_windspeed_to_knots(df)

            #set column types
            df = df.astype(column_types)
            
            #drop uneeded columns
            columns_to_drop = ['number', 'name', 'isDaytime', 'temperatureUnit', 'temperature', 'temperatureTrend', 
                               'icon', 'detailedForecast', 'probabilityOfPrecipitation', 'dewpoint', 'relativeHumidity', 
                               'windSpeed', 'shortForecast', 'endTime', 'startTime', 'windDirection']
            df.drop(columns=columns_to_drop, inplace=True)

        except FutureWarning as warning:
            print(f"Warning while processing {filepath}: " + str(warning))
        except Exception as error:
            print(f"Error while processing {filepath}: " + str(error))
            traceback.print_exc()

    #save to csv
    filepath = getNewFilepath(filepath)
    df.to_csv(filepath)
    #logger.info(f"Data successfully saved to {filepath}")

def convert_windspeed_to_knots(df):
    """
    Converts the 'windspeed_mph' column to knots and stores the result in a new column 'windspeed_knots'.

    Parameters:
    df (pd.DataFrame): DataFrame containing a column 'windspeed_mph'.

    Returns:
    pd.DataFrame: DataFrame with the new 'windspeed_knots' column.
    """
    # Conversion factor from mph to knots
    mph_to_knots = 0.868976
    
    # Ensure 'windSpeed_mph' is numeric, coercing errors to NaN
    df['windSpeed_mph'] = pd.to_numeric(df['windSpeed_mph'], errors='coerce')
    
    # Create new 'windSpeed_knots' column and round to the nearest integer
    df['windSpeed_knots'] = (df['windSpeed_mph'] * mph_to_knots).round().astype('Int64')  # Using Int64 to allow for NaN
    
    return df

def main():

    logging_setup()
    logger.info("Starting forecast_util")

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
    logger.info(f"forecast_util.py finished successfully, processing {file_count} files in {runtime:.2f} seconds")

if __name__ == '__main__':
    main()
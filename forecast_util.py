
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
    'windDirection'                     : str,
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

            #set column types
            df = df.astype(column_types)
            
            #drop uneeded columns
            columns_to_drop = ['number', 'name', 'isDaytime', 'temperatureUnit', 'temperature', 'temperatureTrend', 'icon', 'detailedForecast', 'probabilityOfPrecipitation', 'dewpoint', 'relativeHumidity', 'windSpeed', 'startTime', 'endTime', 'shortForecast', ]
            df.drop(columns=columns_to_drop, inplace=True)
            
            #keep only next X hours of forecast, since this is what we will be using
            hours_to_forecast = 12
            df = df.head(hours_to_forecast)

            #reshape data to make for easier merging into main dataset
            df_result = pd.DataFrame()
            columns_to_reshape = ["windDirection","probabilityOfPrecipitation_percent","dewpoint_degC","relativeHumidity_percent","temperature_F","windSpeed_mph"]
            for column in columns_to_reshape:
                # Create a temporary DataFrame for the current column
                temp_df = df[['timestamp', column]].copy()
                # Create a new column 'n' representing the index
                temp_df['n'] = temp_df.index
                # Pivot the DataFrame
                temp_df = temp_df.pivot_table(index='timestamp', columns='n', values=column, aggfunc='first')
                # Rename columns with the desired format
                temp_df.columns = [f'{column}_{col}h' for col in temp_df.columns]
                # Reset index to make 'timestamp' a regular column
                temp_df = temp_df.reset_index()
                # Concatenate the result to the final DataFrame
                df_result = pd.concat([df_result, temp_df], axis=1)

            #drop duplicate tiemstamp columns and save result
            df_result = df_result.loc[:, ~df_result.columns.duplicated()]
            df = df_result

        except FutureWarning as warning:
            print(f"Warning while processing {filepath}: " + str(warning))
        except Exception as error:
            print(f"Error while processing {filepath}: " + str(error))

    #save to csv
    filepath = getNewFilepath(filepath)
    df.to_csv(filepath)
    #logger.info(f"Data successfully saved to {filepath}")

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

import pandas as pd
import json
import os
import re

json_folder = './forecast-data/'

def convert_filepath(filepath):
    # extract the directory path and filename from the original filepath
    directory, filename = os.path.split(filepath)
    # remove the seconds, and change the extension to .csv
    base_name = re.sub(r'_\d+\.json$', '.csv', filename)
    # Construct the new directory path with "-processed" and the new filename
    new_directory = directory.rstrip('/') + '-processed'
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
              
def main():

    #iterate through all files in the forecast-data folder
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            # Construct the full path to the JSON file
            filepath = os.path.join(json_folder, filename)
            # Process the JSON file
            process_json_file(filepath)

if __name__ == '__main__':
    main()
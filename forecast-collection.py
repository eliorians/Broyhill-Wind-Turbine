import os
import time
import requests
import json

testing = False

def main():
    #metadata url is used to get the actual url below used for requests
    metadata_url = "https://api.weather.gov/points/36.2135,-81.6923"
    url = "https://api.weather.gov/gridpoints/RNK/17,16/forecast/hourly"
    headers = {"User-Agent": "ElisForecastCollection/1.0 (eli.orians@gmail.com)", "Accept": "application/geo+json"}

    while True:
        # Calculate the seconds remaining until the next hour and wait that long
        current_time = time.localtime()
        seconds_until_next_hour = 3600 - current_time.tm_min * 60 - current_time.tm_sec
        if (not testing):
            time.sleep(seconds_until_next_hour)
        
        # Make an API request to get the hourly forecast
        timestamp = time.strftime("%m-%d-%Y_%H-%M-%S")
        response = requests.get(url, headers=headers)
        
        #if the request is good, get the get the data and save the json
        if response.status_code == 200:
            data = response.json()
            file_name = os.path.join("forecast-data", f"forecast_{timestamp}.json")

            with open(file_name, "w") as json_file:
                json.dump(data, json_file)
                print(f"Data saved to {file_name}")
        else:
            print(f"Error: {response.status_code} at {timestamp}")
            if(testing):
                print(response.content)
        
        if (testing):
            exit()

if __name__ == '__main__':
    main()
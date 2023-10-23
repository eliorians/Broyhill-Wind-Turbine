import os
import time
import requests
import json

def main():
    #metadata url is used to get the actual url below used for requests
    metadata_url = "https://api.weather.gov/points/36.2135,-81.6923"
    url = "https://api.weather.gov/gridpoints/RNK/17,16/forecast/hourly"
    headers = {"User-Agent": "ElisForecastCollection/1.0 (eli.orians@gmail.com)", "Accept": "application/geo+json"}

    print("Running forecast collection, start time: " + time.strftime("%m/%d/%Y %H:%M:%S"))

    while True:
        # Calculate the seconds remaining until the next hour and wait that long
        current_time = time.localtime()
        seconds_until_next_hour = 3600 - current_time.tm_min * 60 - current_time.tm_sec
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

        #error 500 is interal error, attempt to recquest 6 more times as to only go one minute away from the hour at most
        elif response.status_code == 500:
            errors = 1
            while errors < 7:
                time.sleep(10)
                timestamp = time.strftime("%m-%d-%Y_%H-%M-%S")
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    file_name = os.path.join("forecast-data", f"forecast_{timestamp}.json")

                    with open(file_name, "w") as json_file:
                        json.dump(data, json_file)
                        print(f"Data saved to {file_name}")
                    
                    break
                else:
                    errors = errors + 1

            if response.status_code != 200:
                print(f"Error: {response.status_code} at {timestamp}")
                print(response.content)
        
        else:
            print(f"Error: {response.status_code} at {timestamp}")
            print(response.content)
    

if __name__ == '__main__':
    main()
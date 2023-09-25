import pandas as pd

def main():

    #put data into dataframes
    windData = pd.read_csv('./data/wind-data.csv')
    print(windData)
    weatherData = pd.read_csv('./data/weather-data.csv')
    print(weatherData)

    #sort wind data by date

    #remove duplicates from wind data

    #consolidate wind data from 15 min intervals to hourly
        #take average of power production


if __name__ == "__main__":
    main()
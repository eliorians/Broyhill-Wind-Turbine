import pandas as pd

#change to same time zone
# wind data = GMT -> find more data from Jim Dees
# weather data = UTC ?

def main():

    #data into dataframes
    windData = pd.read_csv('./data/raw/wind-data.csv')
    weatherData = pd.read_csv('./data/raw/weather-data.csv')

    #data cleaning
    windData = cleanWind(windData)
    weatherData = cleanWeather(weatherData)

    #test
    windData.to_csv('./data/processed/wind-data-cleaned.csv')
    weatherData.to_csv('./data/processed/weather-data-cleaned.csv')

    #combine wind and weather data on date/time


def cleanWind(windData):
    #sort by date/time
    windData = windData.sort_values(by=['datadatetime']) 
    #drop duplicates
    windData = windData.drop_duplicates()
    #convert datadatetime to a timestamp and aggregate to hourly by taking the average production
    windData['datadatetime'] = pd.to_datetime(windData['datadatetime'])
    windData = windData.groupby(pd.Grouper(key='datadatetime', freq='H')).mean()

    return windData

def cleanWeather(weatherData):
    #sort by date/time
    weatherData = weatherData.sort_values(by=['DATE'])
    #convert datadatetime to a timestamp
    weatherData['DATE'] = pd.to_datetime(weatherData['DATE'])
    #aggregate hourly, mean of windspeed/temp/etc/ -> makes errors
    #weatherData = weatherData.groupby(pd.Grouper(key='DATE', freq='H')).mean()

    return weatherData


if __name__ == "__main__":
    main()
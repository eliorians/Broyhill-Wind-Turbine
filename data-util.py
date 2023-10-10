import pandas as pd

def main():

    #data into dataframes
    windData = pd.read_csv('./data/raw/wind-data.csv')
    weatherData = pd.read_csv('./data/raw/weather-data.csv')

    #data cleaning
    windData = cleanWind(windData)
    weatherData = cleanWeather(weatherData)

    #cleaned data output
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
    #find and drop empty columns
    empty_cols = [col for col in weatherData.columns if weatherData[col].isnull().all()]
    weatherData.drop(empty_cols, axis=1,inplace=True)

    #aggregate hourly
    #weatherData = weatherData.groupby(pd.Grouper(key='DATE', freq='H')).mean()

    return weatherData


if __name__ == "__main__":
    main()
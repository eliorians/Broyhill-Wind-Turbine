import pandas as pd

#columns and their data type from weather data, other columns are not selected
#makes for faster reading and makes aggregating easier later on
weathertypes = {
    'DATE'                      : str,
    'DailyWeather'              : str,
    'HourlyAltimeterSetting'    : str,
    'HourlyDewPointTemperature' : str,
    'HourlyDryBulbTemperature'  : str,
    'HourlyPrecipitation'       : float,
    'HourlyPresentWeatherType'  : str,
    'HourlyRelativeHumidity'    : float,
    'HourlySkyConditions'       : str,
    'HourlyStationPressure'     : float,
    'HourlyVisibility'          : float,
    'HourlyWetBulbTemperature'  : float,
    'HourlyWindDirection'       : str,
    'HourlyWindGustSpeed'       : int,
    'HourlyWindSpeed'           : int
}

windtypes = {
    'datadatetime'              : str,
    'powerproduction'           : float,
}

def main():

    #data into dataframes
    windData = pd.read_csv('./data/raw/wind-data.csv', low_memory=False)
    weatherData = pd.read_csv('./data/raw/weather-data.csv', low_memory=False)

    #data cleaning
    windData = cleanWind(windData)
    weatherData = cleanWeather(weatherData)

    #cleaned data output
    windData.to_csv('./data/processed/wind-data-cleaned.csv')
    weatherData.to_csv('./data/processed/weather-data-cleaned.csv')

    #test data types
    print('wind data types:')
    print(windData.dtypes)
    print()
    print('weather data types:')
    print(weatherData.dtypes)


    #todo combine wind and weather data on date/time


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
    #keep only useful columns
    columns_to_remove = [col for col in weatherData.columns if col not in weathertypes]
    weatherData = weatherData.drop(columns=columns_to_remove)

    #todo convert datatypes?
    #weatherData = weatherData[weathertypes.keys()].astype(weathertypes)
    
    #convert data column to datetime
    weatherData['DATE'] = pd.to_datetime(weatherData['DATE'])
    
    #sort by date/time
    weatherData = weatherData.sort_values(by=['DATE'])
    
    #todo aggregate hourly
    
    return weatherData


if __name__ == "__main__":
    main()
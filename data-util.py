import pandas as pd

#columns and their data type from each source, as well as how they will be aggregated
weathertypes = {
    'DATE'                      : str,       #key
    'DailyWeather'              : str,      #last
    'HourlyAltimeterSetting'    : str,      #mean
    'HourlyDewPointTemperature' : str,      #mean
    'HourlyDryBulbTemperature'  : str,      #mean
    'HourlyPrecipitation'       : float,    #mean
    'HourlyPresentWeatherType'  : str,      #last
    'HourlyRelativeHumidity'    : float,    #mean
    'HourlySkyConditions'       : str,      #last
    'HourlyStationPressure'     : float,    #mean
    'HourlyVisibility'          : float,    #mean
    'HourlyWetBulbTemperature'  : float,    #mean
    'HourlyWindDirection'       : str,      #?
    'HourlyWindGustSpeed'       : int,      #?max
    'HourlyWindSpeed'           : int       #mean
}

windtypes = {
    'datadatetime'              : str,      #key
    'powerproduction'           : float,    #mean
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
    # print('wind data types:')
    # print(windData.dtypes)
    # print()
    # print('weather data types:')
    # print(weatherData.dtypes)

    #!combine wind and weather data on date/time


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

    #convert datatypes?
    #weatherData = weatherData[weathertypes.keys()].astype(weathertypes)
    
    #convert data column to datetime
    weatherData['DATE'] = pd.to_datetime(weatherData['DATE'])
    
    #sort by date/time
    weatherData = weatherData.sort_values(by=['DATE'])
    
    #aggregate hourly
    #weatherData = weatherData.groupby(pd.Grouper(key='DATE', freq='H')).last()
    
    return weatherData


if __name__ == "__main__":
    main()
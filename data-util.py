import pandas as pd

def main():

    #data into dataframes
    windData = pd.read_csv('./data/raw/wind-data.csv')
    weatherData = pd.read_csv('./data/raw/weather-data.csv')

    #data cleaning
    windData = cleanWind(windData)

    #test
    windData.to_csv('./data/processed/weather-data-cleaned.csv')

def cleanWind(windData):
    #sort by date/time
    windData = windData.sort_values(by=['datadatetime']) 
    #drop duplicates
    windData = windData.drop_duplicates()
    #convert datadatetime to a timestamp and aggregate to hourly by taking the average production
    windData['datadatetime'] = pd.to_datetime(windData['datadatetime'])
    windData = windData.groupby(pd.Grouper(key='datadatetime', freq='H')).mean()

    return windData


if __name__ == "__main__":
    main()
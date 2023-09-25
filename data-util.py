import pandas as pd

def main():

    #data into dataframes
    windData = pd.read_csv('./data/wind-data.csv')
    weatherData = pd.read_csv('./data/weather-data.csv')

    #data cleaning
    windData = cleanWind(windData)

    print(windData)


def cleanWind(windData):
    #sort by date/time
    windData = windData.sort_values(by=['datadatetime']) 
    #drop duplicates
    windData = windData.drop_duplicates()

    return windData


if __name__ == "__main__":
    main()
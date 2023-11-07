
import pandas as pd
import json

def main():
    json_path = './forecast-data/forecast_10-20-2023_23-00-00.json'
    df = pd.read_json(json_path)
    print(df)

if __name__ == '__main__':
    main()
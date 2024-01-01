# Broyhill-Wind-Turbine
Using machine learning to predict the yield of the Broyhill Wind Turbine. 

Folders:

    logs                    - location for all log files

    forecast-data           - raw weather forecast data collected from National Weather Service API. Collected every hour.

    forecast-data-processed - cleaned forecast data to be used by the model

    turbine-data            - data collected from turbine since 2009, not stored on GitHub

    turbine-data-processed  - cleaned turbine data to be used by the model

    weater-data             - historical weather data for Boone, not forecasts but actual.

    weater-data-processed   - cleaned weather data to be used by the model 

Files:

    forecast-collection.py  - collecting forecast data from the National Weather Service API

    forecast-util.py        - cleans collected forecast data to be used by the model

    turbine-util.py         - clean turbine data to be used by the model

    weather-util.py         - cleans historical weather data to be used by the model

    main.py                 - where the model is trained and evaluated




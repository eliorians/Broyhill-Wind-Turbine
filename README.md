# Broyhill-Wind-Turbine
Using machine learning to predict the yield of the Broyhill Wind Turbine. 

Folders:

    logs                    - location of all log files

    forecast-data           - raw weather forecast data collected from National Weather Service API. Collected every hour.

    forecast-data-processed - cleaned forecast data to be used by the model

    turbine-data            - data collected from turbine since 2009 (frames.csv)

    turbine-data-processed  - cleaned turbine data from thoughout tubrine_util

    plots                   - saved plots made throughout the project

    model-data              - saved outputs from the training process


Files:

    forecast_collection.py  - collecting forecast data from the National Weather Service API

    forecast_util.py        - cleans collected forecast data to be combined with turbine data.

    turbine_util.py         - clean turbine data and combines with forecast data to be used by the model. 

    main.py                 - where the model is trained and evaluated. see #!CONFIG section at top of file for settings.

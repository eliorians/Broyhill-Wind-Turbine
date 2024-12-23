# Broyhill-Wind-Turbine
Using machine learning to predict the yield of the Broyhill Wind Turbine. 

Folders:

    logs                    - location of all log files

    forecast-data           - raw weather forecast data collected from National Weather Service API. Collected every hour.

    forecast-data-processed - cleaned forecast data to be used by the model (run to get processed files)

    turbine-data            - data collected from turbine since 2009. (LFS used to store the turbine data)

    turbine-data-processed  - cleaned turbine data from thoughout tubrine_util (run to get processed files)

    plots                   - saved plots made throughout the project

    model-data              - saved outputs from the training process, eval.txt contains historical runs


Files:

    forecast_collection.py  - collecting forecast data from the National Weather Service API

    forecast_util.py        - cleans collected forecast data to be combined with turbine data.

    turbine_util.py         - clean turbine data and combines with forecast data to be used by the model. 

    main.py                 - where the model is trained and evaluated. see #!CONFIG section at top of file for settings as well as params.py

    params.py               - holds the base features, list of implemented models and their grid search parameters 

    plots.py                - plotting functions used across the project

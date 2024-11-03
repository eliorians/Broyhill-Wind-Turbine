
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sea
from matplotlib import pyplot as plt

def logging_setup():
    logger = logging.getLogger('plots')
    logger.setLevel(logging.INFO)

    # Create a "logs" directory if it doesn't exist
    logs_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_directory, exist_ok=True)

    # Set up logging to a file in the "logs" directory
    log_file = os.path.join(logs_directory, 'plots.log')
    logging.basicConfig(
        level=logging.INFO,  # Set the root logger level to DEBUG
        format='%(levelname)s - %(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Send log messages to the console
            logging.FileHandler(log_file)  # Save log messages to a file in the "logs" directory
        ])

def plotPrediction(timestamp, actual, prediction, model, hoursOut):
    '''
    Plot the prediction against the actual. 
    Creates a scatter plot and a line plot and saves both in ./plots/prediction_plots

    ARGS
    timestamp: the x values for the plot
    actual: actual y values
    prediction: predicted y values
    model: name of model used to create predictions
    '''
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plotPrediction")

    #SCATTER PLOT
    sea.set_theme('paper', style='whitegrid')
    sea.jointplot(x=actual, y=prediction,
                height=14,ratio=6,
                marginal_ticks=True, color='blue',
                kind='reg',
                marginal_kws=dict(color='green'),
                joint_kws={'line_kws': {'color': 'red'}},
                scatter_kws={'alpha': 0.6},
                label='Predicted vs Actual',
    )
    plt.yticks(np.arange(-10, 56, step=1))
    plt.xlabel('Actual (kW)')
    plt.ylabel('Predicted (kW)')
    date_start = timestamp.min()
    date_stop = timestamp.max()
    plt.title(f'{model} Scatter Plot of Actual vs Predicted - Date Range: {date_start} to {date_stop} predicting {hoursOut} hours out')
    plt.legend()
    plt.savefig(f'./plots/prediction_plots/{model}_{hoursOut}_scatter.png')
    plt.show()

    #LINE PLOT
    plt.figure(figsize=(14, 6))
    sea.lineplot(x=timestamp, y=actual, label='Actual', color='blue')
    sea.lineplot(x=timestamp, y=prediction, label='Predicted', color='red')
    plt.xlabel('Timestamp')
    plt.gca().set_xticks(timestamp[::24])
    plt.xticks(rotation=45)
    plt.ylabel('Value (kW)')
    plt.yticks(np.arange(-10, 56, step=1))
    plt.title(f'{model} Line Plot of Actual vs Predicted\nDate Range: {date_start} to {date_stop} predicting {hoursOut} hours out')
    plt.legend()
    plt.savefig(f'./plots/prediction_plots/{model}_{hoursOut}_lineplot.png')
    plt.show()

def plot_TargetVSActual(df, target, actual):
    '''
    Used for plotting the target against some actual data.

    ARGS
    df: the dataframe to pull data from
    target: column to plot as target
    actual: column to plot as actual
    '''
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_TargetVSActual")

    #create plot
    sea.set_theme('paper', style='whitegrid')
    sea.jointplot(data=df, x=actual, y=target, 
                height=10, ratio=5,
                marginal_ticks=True,
                kind='reg',
                joint_kws={'line_kws': {'color': 'red'}},
                marginal_kws=dict(color='green'),
                scatter_kws={'alpha': 0.5},
    )
        
    #output
    plt.savefig(f'./plots/target_vs_actual/{target}_VS_{actual}.png')
    plt.show()

def plot_TargetVSForecasted(df, target, forecast):
    '''
    Used for plotting the target against some actual data, tuned to the forecast data

    ARGS
    df: the dataframe to pull data from
    target: column to plot as target
    wind: column to plot as forecasted windspeed
    '''
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_TargetVSForecasted")

    #create plot
    sea.set_theme('paper', style='whitegrid')
    g = sea.jointplot(data=df, x=forecast, y=target, 
                height=10, ratio=5,
                marginal_ticks=True,
                kind='reg',
                joint_kws={'line_kws': {'color': 'green'}},
                marginal_kws=dict(color='green'),
                scatter_kws={'alpha': 0.5},
    )

    #Polynomial regression line
    degree = 3
    x = df[forecast].values
    y = df[target].values
    p = np.polyfit(x, y, degree)  # Fit polynomial of the specified degree
    poly = np.poly1d(p)

    # Generate x values for plotting the polynomial fit line
    x_range = np.linspace(x.min(), x.max(), 500)
    y_poly = poly(x_range)

    g.ax_joint.plot(x_range, y_poly, color='red', lw=2, label=f'Poly degree {degree}')

    #zoom in on the lower values of forecasted windspeed
    #g.ax_joint.set_xlim(0, 20)

    g.set_axis_labels("Wind Speed (mph)", "Power Output (kW)")
    g.ax_joint.legend()
        
    #output
    plt.savefig(f'./plots/target_vs_forecasted/{target}_VS_{forecast}_poly_{degree}.png')
    plt.show()

def plot_TargetVSFeature(df, target, feature, plotType):
    '''
    Used for plotting the target against a feature.

    ARGS
    df: dataframe to pull columns from
    target: target column from the df to plot against
    feature: column to get from df to plot against target
    plotType: type of plot to use [hex, hist, kde, reg, resid, scatter]
    '''
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_TargetVSFeature")

    #hex plot
    if (plotType == 'hex'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=feature, y=target,
                    height=10, ratio=5,
                    marginal_ticks=True, color='red',
                    kind='hex',
                    marginal_kws=dict(color='green'),
        )

    #hist plot
    if (plotType == 'hist'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=feature, y=target,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='hist',
                    marginal_kws=dict(color='green'),
        )

    #kde plot
    if (plotType == 'kde'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=feature, y=target,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='kde',
                    alpha=.9,
                    marginal_kws=dict(color='green'),
        )

    #reg plot
    if (plotType == 'reg'):
        sea.set_theme('paper', style='whitegrid')
        bins = 30
        order=1

        sea.jointplot(data=df, x=feature, y=target,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='reg',
                    order=order,
                    marginal_kws=dict(color='green'),
                    joint_kws={'line_kws': {'color': 'red'}},
                    scatter_kws={'alpha': 0.5},         
        )

    #resid plot
    if (plotType == 'resid'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=feature, y=target,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='resid',
                    scatter_kws={'alpha': 0.5},
                    marginal_kws=dict(color='green'),
        )

    #scatter plot
    if (plotType == 'scatter'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=feature, y=target,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='scatter',
                    marginal_kws=dict(color='green'),
                    alpha= 0.5,
        )
    
    plt.savefig(f'./plots/target_vs_feature/{target}_VS_{feature}.png')
    plt.show()

def plotQuantities(df, column):
    '''
    Creates a bar plot with the count of each entry for a given column.
    Useful columns to plot: 'WTG1_R_TurbineState' and 'windSpeed_0'
    '''
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plotQuantities")

    #get counts of each unique value in the specified column
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    counts = counts.sort_values(by=column)

    #plotting
    sea.set_theme('paper', style='whitegrid')
    sea.barplot(data=counts, x=column, y='count', color='blue')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Count of Each Entry in {column}')

    #output
    plt.savefig(f'./plots/counts/{column}_count.png')
    plt.show()

    #write counts to file
    #counts.to_csv(f"{column}_counts.csv", index=False)

def plot_windspeed_distribution(columnName):
    '''
    Plot the distribution of windspeed_mph across all CSV files in the given folder as a bar plot.
    Label all wind speeds from 0 to 40 on the x-axis.
    '''
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_windspeed_distribution")

    wind_speeds = []
    csv_folder = 'forecast-data-processed'

    # Iterate over all CSV files in the processed folder
    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            # Load the CSV file
            filepath = os.path.join(csv_folder, filename)
            try:
                df = pd.read_csv(filepath)
                # Collect all windspeed values
                wind_speeds.extend(df[columnName].dropna().tolist())
            except Exception as error:
                print(f"Error reading {filename}: {error}")

    # Convert wind speeds to integers for binning
    wind_speeds = [int(ws) for ws in wind_speeds]

    # Create a frequency count of wind speeds from 0 to 40
    wind_speed_counts = pd.Series(wind_speeds).value_counts().sort_index()
    wind_speed_counts = wind_speed_counts.reindex(range(0, 41), fill_value=0)  # Fill missing wind speeds with 0 count

    # Plotting the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(wind_speed_counts.index, wind_speed_counts.values, color='skyblue')

    # Set the x-axis to show all values from 0 to 40
    plt.xticks(range(0, 41, 1))  # Show every wind speed from 0 to 40
    max_freq = wind_speed_counts.max()  # Find the maximum frequency
    plt.yticks(range(0, max_freq + 5000, 5000))  # Set y-ticks to be every 500


    plt.title(f'Distribution of {columnName}', fontsize=14)
    plt.xlabel(columnName, fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)

    # Save the plot
    plt.savefig(f'./plots/counts/all_{columnName}.png')
    plt.show()

def plotHoursOutAccuracy():
    #values collected
    linearRegression_hours = [6, 12, 24, 48, 72, 96, 120, 144, 154]
    linearRegression_rmse = [9.810125117039744, 10.286906840714082, 10.74599004815966, 11.626232167435283, 12.495538478919908, 13.354669656794997, 13.610518715669519, 14.884202462577573, 16.003219059547543]

    #set the values to be used
    model = 'linear_regression'

    if (model == 'linear_regression'):
        rmse = linearRegression_rmse
        hours = linearRegression_hours

    #plot
    plt.figure(figsize=(12, 10))
    plt.bar(hours, rmse, color='seagreen', width=5)
    
    plt.xlabel('Hours Out (Days out)')
    plt.ylabel('RMSE')
    plt.title(f'Prediction Accuracy Over Time (RMSE) for {model}')

    hour_day_labels = [f'{hours}h\n({hours / 24:.1f} days)' for hours in linearRegression_hours]
    plt.xticks(ticks=linearRegression_hours, labels=hour_day_labels, rotation=65, ha='center')
    
    plt.savefig(f'plots\hoursOutAccuracy\{model}')
    plt.show()

if __name__ == "__main__":
    plotHoursOutAccuracy()
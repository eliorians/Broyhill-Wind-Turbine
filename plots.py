
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
    plt.savefig(f'./plots/prediction_plots/{model}_{hoursOut}_scatter.png', bbox_inches='tight', pad_inches=0.5)
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
    plt.savefig(f'./plots/prediction_plots/{model}_{hoursOut}_lineplot.png', bbox_inches='tight', pad_inches=0.5)
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

    total_count = counts['count'].sum()
    plt.text(0.95, 0.95, f'Total Count: {total_count}', ha='right', va='top', transform=plt.gca().transAxes, 
             fontsize=12, color='black', fontweight='bold')

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
    baseline_rmse = 11.44744822
    hours          = [1,    2,    3,    4,    5,    6,    12,    24,    48,    72,    96,    120,   144]
    hours_12       = [1,    2,    3,    4,    5,    6,    12]
    linear         = [9.70, 9.68, 9.73, 9.83, 9.85, 9.81, 10.28, 10.74, 11.62, 12.49, 13.35, 13.61, 14.88]
    poly           = [9.67, 9.51, 9.52, 9.51, 9.44, 9.43, 9.95]
    poly_scaling   = [9.60, 9.40, 9.44, 9.47, 9.46, 9.37, 9.99,  9.88,  11.48, 13.02, 12.75, 13.66, 15.64]
    svr            = [9.83, 9.81, 9.82, 9.82, 9.84, 9.55, 9.79, 10.20,  12.04, 12.25, 12.70, 12.87, 15.02]


    #plot
    plt.figure(figsize=(16, 12))
    plt.plot(hours, linear, color='red', label='Linear Regression', marker='o', alpha=.5)
    #plt.plot(hours_12, poly, color='blue', label='Polynomial Regression (No Convergence after 12 hours)', marker='o', alpha=.5)
    plt.plot(hours, poly_scaling, color='green', label='Polynomial Regression w/ Scaling', marker='o', alpha=.5)
    plt.plot(hours, svr, color='purple', label='SVR', marker='o', alpha=.5)

    
    plt.axhline(baseline_rmse, color='black', linestyle='--', linewidth=1.5, label=f'Baseline RMSE ({baseline_rmse:.2f})')

    # axis
    plt.yticks(np.arange(0, 16, 0.5))
    plt.xticks(hours)
    plt.ylim(5, 16)
    # labels
    plt.xlabel('Hours Out (Days out)')
    plt.ylabel('RMSE (Lower Values Better)')
    plt.title(f'Prediction Accuracy Over Time (Avg RMSE)')
    plt.legend()
    plt.grid(True)
    
    #show hours converted to days
    # hour_day_labels = [f'{hour}h\n({hour / 24:.1f} days)' for hour in hours]
    # plt.xticks(ticks=hours, labels=hour_day_labels, rotation=65, ha='center')
    
    plt.savefig(f'plots\hoursOutAccuracy\multiple-models')
    plt.show()

def plotForecastAccuracy(df, hoursOut):
    """
    Plots a comparison of forecasted wind speed in mph (converted to m/s) 
    and actual wind speed in m/s over time from a DataFrame.

    Parameters:
    - df: DataFrame with columns:
        - 'windSpeed_mph': forecasted wind speed in mph
        - 'WTG1_R_WindSpeed_mps': actual wind speed in m/s
        - 'timestamp': timestamps for each data point
    """
    # Convert forecasted wind speed from mph to m/s
    df['forecast_wind_mps'] = df['windSpeed_mph_0'] * 0.44704  # 1 mph ≈ 0.44704 m/s

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(df['forecast_wind_mps'], df['WTG1_R_WindSpeed_mps'], alpha=0.6, color='purple')

    # actual line of best fit
    slope, intercept = np.polyfit(df['forecast_wind_mps'], df['WTG1_R_WindSpeed_mps'], 1)
    line_x = np.linspace(df['forecast_wind_mps'].min(), df['forecast_wind_mps'].max(), 100)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', linewidth=2, label='Actual Line')

    #ideal line plot
    plt.plot([df['forecast_wind_mps'].min(), df['forecast_wind_mps'].max()], 
             [df['forecast_wind_mps'].min(), df['forecast_wind_mps'].max()], 
             color='black', linestyle='--', label='Ideal Line (y = x)')
    
    #determine MAE
    mae = np.mean(np.abs(df['forecast_wind_mps'] - df['WTG1_R_WindSpeed_mps']))
    plt.text(0.05, 0.95, f'MAE: {mae:.2f} m/s', transform=plt.gca().transAxes,
             fontsize=12, color='red', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Adding titles and labels
    plt.title(f"Forecasted vs Actual Wind Speed for {hoursOut} hours out")
    plt.xlabel("Forecasted Wind Speed (m/s)")
    plt.ylabel("Actual Wind Speed (m/s)")
    plt.tight_layout()

    plt.legend(loc='upper right')
    plt.savefig(f'plots/forecastAccuracy/windspeed_accuracy_scatterplot_{hoursOut}')
    plt.show()

if __name__ == "__main__":
    plotHoursOutAccuracy()
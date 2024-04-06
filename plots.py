
import os
import logging
import numpy as np
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

def plotPrediction(timestamp, actual, prediction, model):
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plotPrediction")

    #SCATTER PLOT
    sea.set_theme('paper', style='whitegrid')
    sea.jointplot(x=actual, y=prediction,
                height=10, ratio=5,
                marginal_ticks=True, color='blue',
                kind='reg',
                marginal_kws=dict(color='green'),
                joint_kws={'line_kws': {'color': 'red'}},
                scatter_kws={'alpha': 0.6},
                label='Predicted vs Actual',
    )
    plt.yticks(np.arange(-10, 56, step=1))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig(f'./plots/prediction_plots/{model}_scatter.png')
    plt.show()

    #LINE PLOT
    plt.figure(figsize=(10, 5))
    sea.lineplot(x=timestamp, y=actual, label='Actual', color='blue')
    sea.lineplot(x=timestamp, y=prediction, label='Predicted', color='red')
    plt.xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.ylabel('Value')
    plt.yticks(np.arange(-10, 56, step=1))
    date_start = timestamp.min()
    date_stop = timestamp.max()
    plt.title(f'{model} Line Plot of Actual vs Predicted\nDate Range: {date_start} to {date_stop}')
    plt.legend()
    plt.savefig(f'./plots/prediction_plots/{model}_lineplot.png')
    plt.show()

def plot_PowerVSActualWind(df, power, actualWindSpeed):
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_PowerVSActualWind")

    #create plot
    sea.set_theme('paper', style='whitegrid')
    sea.jointplot(data=df, x=actualWindSpeed, y=power, 
                height=10, ratio=5,
                marginal_ticks=True,
                kind='reg',
                joint_kws={'line_kws': {'color': 'red'}},
                marginal_kws=dict(color='green'),
                scatter_kws={'alpha': 0.5},
    )
        
    #output
    plt.savefig('./plots/actualWindspeed_vs_power/jointplot_powerVSactualWind.png')
    plt.show()


def plot_PowerVSForecastWind(df, power, forecastWindspeed):
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_PowerVSForecastWind")

    plotType = 'reg'

    #hex plot
    if (plotType == 'hex'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=forecastWindspeed, y=power,
                    height=10, ratio=5,
                    marginal_ticks=True, color='red',
                    kind='hex',
                    marginal_kws=dict(color='green'),
        )
        plt.savefig('./plots/forecastWindspeed_vs_power/hex_powerVsforecastWind.png')

    #hist plot
    if (plotType == 'hist'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=forecastWindspeed, y=power,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='hist',
                    marginal_kws=dict(color='green'),
        )
        plt.savefig('./plots/forecastWindspeed_vs_power/hist_powerVsforecastWind.png')

    #kde plot
    if (plotType == 'kde'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=forecastWindspeed, y=power,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='kde',
                    alpha=.9,
                    marginal_kws=dict(color='green'),
        )
        plt.savefig('./plots/forecastWindspeed_vs_power/kde_powerVsforecastWind.png')

    #reg plot
    if (plotType == 'reg'):
        sea.set_theme('paper', style='whitegrid')
        bins = 30
        order=1

        sea.jointplot(data=df, x=forecastWindspeed, y=power,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='reg',
                    order=order,
                    marginal_kws=dict(color='green'),
                    joint_kws={'line_kws': {'color': 'red'}},
                    scatter_kws={'alpha': 0.5},         
        )

        plt.savefig(f'./plots/forecastWindspeed_vs_power/reg{order}_powerVsforecastWind.png')

    #resid plot
    if (plotType == 'resid'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=forecastWindspeed, y=power,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='resid',
                    scatter_kws={'alpha': 0.5},
                    marginal_kws=dict(color='green'),
        )
        plt.savefig('./plots/forecastWindspeed_vs_power/resid_powerVsforecastWind.png')

    #scatter plot
    if (plotType == 'scatter'):
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=forecastWindspeed, y=power,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='scatter',
                    marginal_kws=dict(color='green'),
                    alpha= 0.5,
        )
        plt.savefig('./plots/forecastWindspeed_vs_power/scatter_powerVsforecastWind.png')

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
    plt.show()
    plt.savefig(f'./plots/counts/{column}_count.png')

    #write counts to file
    #counts.to_csv(f"{column}_counts.csv", index=False)
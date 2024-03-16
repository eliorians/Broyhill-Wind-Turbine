
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

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    
    #output
    plt.tight_layout()
    plt.savefig(f'./plots/prediction_plots/{model}_actualVSprediction.png')
    plt.show()


def plot_PowerVSActualWind(df, power, actualWindSpeed):
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_PowerVSActualWind")

    threshold = 3
    z_scores_actualWind = np.abs((df[actualWindSpeed] - df[actualWindSpeed].mean()) / df[actualWindSpeed].std())
    z_scores_power = np.abs((df[power] - df[power].mean()) / df[power].std())
    outliers = df[(z_scores_actualWind > threshold) | (z_scores_power > threshold)]

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
    
    plt.scatter(outliers[actualWindSpeed], outliers[power], label='Outliers', color='black', marker='x')
    
    #output
    plt.savefig('./plots/jointplot_powerVSactualWind_outliers.png')
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
        order=3
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




import os
import logging
import numpy as np
import seaborn as sea
from matplotlib import pyplot as plt

logger = logging.getLogger('plots')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def logging_setup():
    # Create a "logs" directory if it doesn't exist
    logs_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_directory, exist_ok=True)
    # Set up logging to a file in the "logs" directory
    log_file = os.path.join(logs_directory, 'plots.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Send log messages to the console
            logging.FileHandler(log_file)  # Save log messages to a file in the "logs" directory
        ])

def plotPrediction(timestamp, actual, prediction, model):

    logger.info("in plotPrediction")

    try:
        # Create scatter plot using Seaborn
        sea.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sea.scatterplot(x=actual, y=prediction, label='Predicted vs Actual', alpha=0.6, kind='reg')
        
        # Plot setup
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted for {model}')
        plt.legend()
        plt.grid(True)
        
        # Save and show the plot
        plt.tight_layout()
        plt.savefig(f'./plots/prediction_plots/scatterplot_{model}_prediction.png')
        plt.show()

    except Exception as e:
        logger.error(f"Error: {str(e)}")

    return



def plot_PowerVSActualWind(df, power, actualWindSpeed):
    logger.info("in plot_PowerVSActualWind")

    try:
        # Create the scatter plot using seaborn
        sea.set(style="whitegrid")
        plt.figure(figsize=(8, 5))
        sea.jointplot(data=df, x=actualWindSpeed, y=power, kind='reg')
        
        # Set labels and title
        plt.xlabel(actualWindSpeed)
        plt.ylabel(power)
        plt.title(f'Scatter Plot of {actualWindSpeed} against {power}')
        
        # Save and show the plot
        plt.tight_layout()
        plt.savefig('./plots/jointplot_powerVSactualWind.png')
        plt.show()

    except Exception as e:
        logger.error(f"Error: {str(e)}")

    return


def plot_PowerVSForecastWind(df, power, forecastWindspeed):
    logger.info("in plot_PowerVSForecastWind")

    try:
        #custom bin work
        min_value_x = np.min(df[forecastWindspeed])
        max_value_x = np.max(df[forecastWindspeed])
        min_value_y = np.min(df[power])
        max_value_y = np.max(df[power])
        bin_width_x = 1.0
        bin_width_y = 1.0 

        custom_bins_x = np.arange(min_value_x, max_value_x + bin_width_x, bin_width_x)
        custom_bins_y = np.arange(min_value_y, max_value_y + bin_width_y, bin_width_y)

        # Create the scatter plot using seaborn
        sea.set(style="whitegrid")
        plt.figure(figsize=(8, 5))
        sea.jointplot(data=df, x=forecastWindspeed, y=power, kind='reg', bins=[custom_bins_x, custom_bins_y])
        
        # Set labels and title
        plt.xlabel(forecastWindspeed)
        plt.ylabel(power)
        plt.title(f'Scatter Plot of {forecastWindspeed} against {power}')
        
        # Save and show the plot
        plt.tight_layout()
        plt.savefig('./plots/jointplot_powerVSforecastWind.png')
        plt.show()

    except Exception as e:
        logger.error(f"Error: {str(e)}")

    return
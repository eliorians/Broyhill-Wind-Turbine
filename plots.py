
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

    try:
        # Create scatter plot using Seaborn
        sea.set(style="whitegrid")
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
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_PowerVSActualWind")

    try:
        #create plot
        sea.set(style="whitegrid")
        sea.jointplot(data=df, x=actualWindSpeed, y=power, 
                    height=10, ratio=5,
                    kind='reg',)
        
        #labels
        plt.xlabel(actualWindSpeed)
        plt.ylabel(power)
        plt.title(f'{actualWindSpeed} against {power}')
        
        #output
        plt.savefig('./plots/jointplot_powerVSactualWind.png')
        plt.show()

    except Exception as e:
        logger.error(f"Error: {str(e)}")


def plot_PowerVSForecastWind(df, power, forecastWindspeed):
    logging_setup()
    logger = logging.getLogger('plots')
    logger.info("in plot_PowerVSForecastWind")

    try:
        #create plot
        sea.set_theme('paper', style='whitegrid')
        sea.jointplot(data=df, x=forecastWindspeed, y=power,
                    height=10, ratio=5,
                    marginal_ticks=True, color='blue',
                    kind='hist',
                    #alpha=.9,
                    marginal_kws=dict(color='green'),
                    #scatter_kws={'alpha': 0.5},
                    #joint_kws={'line_kws': {'color': 'red'}},
                    )

        #output
        plt.savefig('./plots/hist_powerVSforecastWind.png')
        plt.show()

    except Exception as e:
        logger.error(f"Error: {str(e)}")

    return



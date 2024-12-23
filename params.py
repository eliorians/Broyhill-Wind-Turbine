from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR


#names of columns that can be used as features. add a _# up to hoursToForecast-1 (use generateFeatures())
#removed windSpeed_mph for windSpeed_knots
featsList=['windSpeed_knots', 'windDirection_x', 'windDirection_y', 'probabilityOfPrecipitation_percent', 'dewpoint_degC', 'relativeHumidity_percent', 'temperature_F']

turbineFeatsList=['WTG1_R_WindSpeed_mps']


#List of models able to be used
modelList = {
    'baseline'              : 'baseline',
    'linear_regression'     : LinearRegression(n_jobs=-1),
    'random_forest'         : RandomForestRegressor(n_jobs=-1, verbose=0),
    'polynomial_regression' : make_pipeline(StandardScaler(), PolynomialFeatures(), LinearRegression(n_jobs=-1)),
    'decision_tree'         : DecisionTreeRegressor(),
    'gradient_boosted_reg'  : GradientBoostingRegressor(verbose=0),
    'ridge_cv'              : RidgeCV(),
    'lasso_cv'              : LassoCV(n_jobs=-1),
    'elastic_net_cv'        : ElasticNetCV(n_jobs=-1),
    'svr'                   : SVR(),
    'kernal_ridge'          : KernelRidge(),
    'bagging'               : BaggingRegressor(n_jobs=-1),
    'ada_boost'             : AdaBoostRegressor(),
    'mlp_regressor'         : MLPRegressor(),
    'gaussian'              : GaussianProcessRegressor()
}

#param_grid associated with each model
#some params commented out as too many result in too long of grid search
paramList = {
    'linear_regression'     : {'fit_intercept' : [True, False],
                                'copy_X'       : [True, False],
                                'positive'     : [True, False],
    },

    'random_forest'         : {'n_estimators'             : [100, 500],
                              # 'criterion'                : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                               'max_depth'                : [None, 10],
                               'min_samples_split'        : [1, 10],
                               'min_samples_leaf'         : [1, 2],
                              # 'min_weight_fraction_leaf' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                               'max_features'             : ['sqrt', 'log2'],
                              # 'max_leaf_nodes'           : [None, 1, 10],
                              # 'min_impurity_decrease'    : [0.0, 1.0, 10.0],
                               'bootstrap'                : [True],
                              # 'oob_score'                : [True, False],
                              # 'random_state'             : [None, 0, 1, 2],
                              # 'warm_start'               : [True, False],
                              # 'ccp_alpha'                : [0.0, 1.0, 10.0],
                              # 'max_samples'              : [None, 0.0, 0.5, 1.0],
                              # 'monotonic_cst'            : [None],
    },

    'polynomial_regression' : { 'polynomialfeatures__degree'          : [1, 2, 3, 4, 5, 6],
                                'polynomialfeatures__interaction_only': [True, False],
                                'linearregression__fit_intercept'     : [True, False],
                                'linearregression__copy_X'            : [True, False],
                                'linearregression__positive'          : [True, False],
    },

    'decision_tree'         : {'criterion'                : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                               'splitter'                 : ['best', 'random'],
                               'max_depth'                : [None, 10],
                               'min_samples_split'        : [1, 2, 10, .5],
                               'min_samples_leaf'         : [1, 2, 10, .5],
                               'min_weight_fraction_leaf' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                               'max_features'             : ['sqrt', 'log2', None],
                               'random_state'             : [None],
                               'max_leaf_nodes'           : [None],
                               'min_impurity_decrease'    : [0.0, 1.0, 10.0],
                               'ccp_alpha'                : [0.0, 1.0, 10.0],
                               'monotonic_cst'            : [None],
    },

    'gradient_boosted_reg'  : {'loss'                     : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                               'learning_rate'            : [0.1, 0.2, 0.5, 1, 10],
                               'n_estimators'             : [10, 100, 500],
                               'subsample'                : [0.1, 0.2, 1.0],
                               #'criterion'                : ['friedman_mse', 'squared_error'],
                               'min_samples_split'        : [1, 2, 5, 10],
                               #'min_weight_fraction_leaf' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                               'max_depth'                : [None, 3, 10],
                               'min_impurity_decrease'    : [0.0, 1.0, 10.0],
                               #'init'                     : [None],
                               #'random_state'             : [None],
                               'max_features'             : ['sqrt', 'log2', None],
                               #'alpha'                    : [0.0, 0.1, 0.5, .8, .9, 1.0],
                               #'max_leaf_nodes'           : [None],
                               #'warm_start'               : [True, False],
                               #'validation_fraction'      : [0.0, 0.1, 0.5, 0.9, 1.0],
                               #'n_iter_no_change'         : [None, 1, 10],
                               #'tol'                      : [],
                               #'ccp_alpha'                : [0.0, 0.5, 1.0],
    },

    'ridge_cv'              : {'alphas'        : [[0.1, 1.0, 10.0], [0.01, 0.1, 1.0, 10.0, 100.0]],
                               'fit_intercept' : [True, False],
                               'scoring'       : ['neg_mean_squared_error'],
                               'cv'            : [TimeSeriesSplit(n_splits=5), TimeSeriesSplit(n_splits=10)],
    },

    'lasso_cv'              : {'alphas': [[0.1, 1.0, 10.0], [0.01, 0.1, 1.0, 10.0, 100.0]],
                                'fit_intercept': [True, False],
                                'max_iter': [1000, 10000],
                                'tol': [0.0001, 0.001],
                                'cv': [TimeSeriesSplit(n_splits=5), TimeSeriesSplit(n_splits=10)],
    },

    'elastic_net_cv'        : {'alphas': [[0.1, 1.0, 10.0], [0.01, 0.1, 1.0, 10.0, 100.0]],
                                'l1_ratio': [0.1, 0.5, 0.9],
                                'fit_intercept': [True, False],
                                'max_iter': [1000, 10000],
                                'tol': [0.0001, 0.001],
                                'cv' : [TimeSeriesSplit(n_splits=5), TimeSeriesSplit(n_splits=10)],
                                'n_alphas' : [100, 1000, 10000]
    },

    'svr'                   : {'kernel': ['linear', 'poly', 'rbf'],
                                'degree': [1, 2, 3, 4, 5],
                                #'gamma': [scale, auto],
                                #'coef0': [],
                                'tol': [0.0001, 0.001],
                                #'C' : [1],
                                #'epsilon' : [.01],
                                'shrinking' : [True, False],
    },

    'kernal_ridge'              : {'alpha'  : [1e-5, 1e-2, .1, 10],
                                'kernel' : ['polynomial', 'rbf'],
                                'gamma' : [.1, 1, 10], 
                                'degree': [1, 2, 3], 
    },

    'bagging'                   : {'estimator' : [DecisionTreeRegressor(), make_pipeline(PolynomialFeatures(degree=3), LinearRegression(n_jobs=-1))],
                                   'n_estimators' : [50, 100, 250, 500],
                                   'max_samples' : [.5, .7, 1],
                                   'max_features' : [None, .5, 1],
                                   'bootstrap' : [True, False],
                                   'bootstrap_features' : [True, False],
                                   'oob_score' : [True, False],
                                   'warm_start' : [True, False],
    },

    'ada_boost'                 : {'estimator' : [DecisionTreeRegressor(), make_pipeline(PolynomialFeatures(degree=3))],
                                   'n_estimators' : [50, 100, 250, 500],
                                   'learning_rate' : [.01, .1, .5, 1],
                                   'loss' : ['linear', 'square', 'exponential'],
    },

    'mlp_regressor'             : {'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],  # Number of neurons in each hidden layer
                                    'activation': ['relu', 'tanh'],  # Common activation functions for better convergence in time series
                                    'solver': ['adam'],  # 'adam' is generally good for most datasets
                                    'alpha': [0.0001, 0.001, 0.01],  # Regularization term
                                    'learning_rate': ['adaptive'],  # Learning rate schedule
                                    'max_iter': [500, 1000],  # Maximum number of iterations
                                    'early_stopping': [True],  # Use early stopping to avoid overfitting
                                    #'validation_fraction': [0.1],  # Fraction of training data to use for validation
                                    'batch_size': ['auto'],  # Size of minibatches for stochastic optimizers
                                    #'momentum': [0.9, 0.99],  # Momentum for SGD (if using SGD solver)
                                    #'nesterovs_momentum': [True, False],  # Whether to use Nesterov's momentum
    },

    'gaussian'                  : {
                                    'kernel': [
                                        RBF(length_scale=1.0),                              # Radial basis function kernel
                                        #Matern(length_scale=1.0, nu=1.5),                # Matern kernel with ν=1.5
                                        #RationalQuadratic(length_scale=1.0, alpha=0.01), # Rational Quadratic kernel
                                        #DotProduct(sigma_0=1.0) + WhiteKernel(),        # Dot product kernel with added noise
                                    ],
                                    'alpha': [1e-10],                               # Noise level in the target values
                                    'n_restarts_optimizer': [0, 5, 10],               # Number of restarts to avoid local minima
    },
}
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

#names of columns that can be used as features. add a _# up to hoursToForecast-1 (use generateFeatures())
featsList=['windSpeed_mph', 'windDirection_x', 'windDirection_y', 'probabilityOfPrecipitation_percent', 'dewpoint_degC', 'relativeHumidity_percent', 'temperature_F']

#List of models able to be used
modelList = {
    'baseline'              : 'baseline',
    'linear_regression'     : LinearRegression(n_jobs=-1),
    'random_forest'         : RandomForestRegressor(n_jobs=-1, verbose=0),
    'polynomial_regression' : make_pipeline(PolynomialFeatures(), LinearRegression(n_jobs=-1)),
    'decision_tree'         : DecisionTreeRegressor(),
    'gradient_boosted_reg'  : GradientBoostingRegressor(verbose=0),
}

#param_grid associated with each model
#some params commented out as too many result in too long of grid search
paramList = {
    'linear_regression'     : {'fit_intercept' : [True, False],
                                'copy_X'       : [True, False],
                                'positive'     : [True, False],
                                },

    'random_forest'         : {'n_estimators'             : [10, 100, 500],
                              # 'criterion'                : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                               'max_depth'                : [None, 10],
                               'min_samples_split'        : [1, 2, 10, .5],
                               'min_samples_leaf'         : [1, 2, 10, .5],
                              # 'min_weight_fraction_leaf' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                               'max_features'             : ['sqrt', 'log2', None],
                              # 'max_leaf_nodes'           : [None, 1, 10],
                              # 'min_impurity_decrease'    : [0.0, 1.0, 10.0],
                               'bootstrap'                : [True, False],
                              # 'oob_score'                : [True, False],
                              # 'random_state'             : [None, 0, 1, 2],
                              # 'warm_start'               : [True, False],
                              # 'ccp_alpha'                : [0.0, 1.0, 10.0],
                              # 'max_samples'              : [None, 0.0, 0.5, 1.0],
                              # 'monotonic_cst'            : [None],
                               },

    'polynomial_regression' : { 'polynomialfeatures__degree'          : [1, 2, 3],
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
}
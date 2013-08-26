from scipy.stats import linregress
from sklearn import metrics
import numpy as np

# Methods to transform an array
def poly_transform(x, deg):
    return np.array(x) ** deg

def exp_transform(x):
    return np.exp(x)

def sigmoid_transform(x):
    return 1.0 / (1.0 + np.exp(-np.array(x)))

# Methods to compute regression coefficients for transformed array
def remove_nan(v):
    rv = list(v)
    if np.isnan(rv[3]):
        rv[3] = 0
    if np.isnan(rv[4]):
        rv[4] = 1000000
    return rv
    
def poly_linregress(x, y, x_deg, y_deg):
    return_value = linregress(poly_transform(x, x_deg), poly_transform(y, y_deg))
    return remove_nan(return_value)

def sigmoidx_linregress(x, y, y_deg):
    return_value = linregress(sigmoid_transform(x), poly_transform(y, y_deg))
    return remove_nan(return_value)

def sigmoidy_linregress(x, x_deg, y):
    return_value = linregress(poly_transform(x, x_deg), sigmoid_transform(y))
    return remove_nan(return_value)


# Finally transform the input into all polynomial combination linear regression coefficients
def complex_regress( (x, y) ):
    features = []
    for x_deg in range(1, 4):
        for y_deg in range(1, 4):
            if x_deg == 1 and y_deg == 1:
                continue
            features = features + list(poly_linregress(x, y, x_deg, y_deg))

    for x_deg in range(1, 4):
        features = features + list(sigmoidy_linregress(x, x_deg, y))

    for y_deg in range(1, 4):
        features = features + list(sigmoidx_linregress(x, y, y_deg))

    return features

def adjusted_mutual_information( (x, y) ):
    return metrics.adjusted_mutual_info_score(x, y)

def adjusted_rand( (x, y) ):
    return metrics.adjusted_rand_score(x, y)

def mutual_information( (x, y) ):
    return metrics.mutual_info_score(x, y)

def homogeneity_completeness( (x, y) ):
    return list(metrics.homogeneity_completeness_v_measure(x, y))

def normalized_mutual_information( (x, y) ):
    return metrics.normalized_mutual_info_score(x, y)

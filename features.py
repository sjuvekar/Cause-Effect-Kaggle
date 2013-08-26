import data_io
import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy import stats
from scipy.spatial import distance
import copy
import multiprocessing


class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def get_params(self, deep=1):
        return {"features": self.features}

    def fit(self, X, y=None):
        for feature_name, column_names, extractor in self.features:
            extractor.fit(X[column_names], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.transform(X[column_names])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            ans = np.concatenate(extracted, axis=1)
        else: 
            ans = extracted[0]
        return ans

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            print feature_name
            fea = extractor.fit_transform(X[column_names], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            ans = np.concatenate(extracted, axis=1)
        else:
            ans = extracted[0]
        return ans

def identity(x):
    return x

def rng(x):
    return (np.max(x) - np.min(x))

def median(x):
    return np.median(x)

def percentile(x, q):
    return np.percentile(x, q)

def percentile25(x):
    return np.percentile(x, 25)

def percentile75(x):
    return np.percentile(x, 75)

def sharpe(x):
    v = stats.variation(x)
    if np.isinf(v):
      return 0
    else:
      return v

def bollinger(x):
    m = np.mean(x)
    mx = np.max(x)
    mn = np.min(x)
    if np.allclose(mx, mn):
      return (mx - m)
    else:
      return ( (mx - m)/(mx - mn) )

def count_unique(x):
    return len(set(x))

def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    
    hx = 0.0;
    for i in range(len(x)-1):
        delta = x[i+1] - x[i];
        if delta != 0:
            hx += np.log(np.abs(delta));
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1);

    return hx

def entropy_difference( (x, y) ):
    nx = normalized_entropy(x)
    ny = normalized_entropy(y)
    #return normalized_entropy(x) - normalized_entropy(y)
    return [nx, ny, nx - ny]

def correlation( (x, y) ):
    return list(pearsonr(x, y))
    #return distance.correlation(x, y)

def correlation_magnitude( (x, y) ):
    return np.abs(correlation( (x, y) ))

def linregress( (x, y) ):
    return stats.linregress(x, y)

def ttest_ind( (x, y) ):
    return list(stats.ttest_ind(x, y))

def ttest_rel_t( (x, y) ):
    return float(stats.ttest_rel(x, y)[0])

def ttest_rel_p( (x, y) ):
    return stats.ttest_rel(x, y)[1]

def ks_2samp( (x, y) ):
    return stats.ks_2samp(x, y)

def kruskal( (x, y) ):
    return stats.kruskal(x, y)

def bartlett( (x, y) ):
    return stats.bartlett(x, y)

def levene( (x, y) ):
    return stats.levene(x, y)

def shapiro(x):
    return stats.shapiro(x)

def fligner( (x, y) ):
    return stats.fligner(x, y)

def mood( (x, y) ):
    return stats.mood(x, y)

def oneway( (x, y) ):
    return stats.oneway(x, y)

#Distance based measures
def braycurtis( (x, y) ):
    return distance.braycurtis(x, y)

def canberra( (x, y) ):
    return distance.canberra(x, y)

def chebyshev( (x, y) ):
    return distance.chebyshev(x, y)

def cityblock( (x, y) ):
    return distance.cityblock(x, y)

def cosine( (x, y) ):
    return distance.cosine(x, y)

def hamming( (x, y) ):
    return distance.hamming(x, y)

def minkowski( (x, y) ):
    ret = []
    for p in range(2, 6):
      ret += [ distance.minkowski(x, y, p) ]
    return ret

def sqeuclidean( (x, y) ):
    return distance.sqeuclidean(x, y)

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return_value = np.array([self.transformer(x) for x in X], ndmin=2).T
        if return_value.shape[1] == 1:
            return return_value
        else:
            return return_value.T


# Important: Define G_PROCESS_POOL just before using it and after defining all relevant methods.
G_PROCESS_POOL = multiprocessing.Pool()

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return_value = G_PROCESS_POOL.map(self.transformer, [x[1] for x in X.iterrows()])
        return_value = np.array(return_value, ndmin=2).T
        #return_value = np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T
        if return_value.shape[1] == 1:
            return return_value
        else:
            return return_value.T

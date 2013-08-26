from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import independent_component as indcomp
from scipy.stats import pearsonr

def anm_fit( (x, y) ):
  newX = np.array(x).reshape(len(x), 1)
  clf = GradientBoostingRegressor()
  clf.fit(newX, y)
  err = y - clf.predict(newX)
  ret =  [clf.score(newX, y)] + list(pearsonr(x, err))
  return ret

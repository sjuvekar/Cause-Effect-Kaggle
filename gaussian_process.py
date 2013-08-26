from sklearn.gaussian_process import GaussianProcess
import numpy as np
import independent_component as indcomp
from scipy.stats import pearsonr

def gaussian_fit_likelihood(x, y):
    # Remove duplicates
    n_dupl = 0
    d = dict()
    for i in range(len(x)):
      try:
        if d[x[i]] != y[i]:
          n_dupl += 1
          d.pop(x[i], None)
      except:
        d[x[i]] = y[i]

    ret = [n_dupl]

    try:  
      newX = np.atleast_2d(d.keys()).T
      newY = np.array(d.values()).ravel()
      g = GaussianProcess(theta0=1e5, thetaL=1e-4, thetaU=1e-1)
      #g = GaussianProcess()
      g.fit(newX, newY)
      err = newY - g.predict(newX)
      p = pearsonr(err, newX)
      ret += [g.reduced_likelihood_function_value_]
      ret += p
    except:
      #fp = open("bad_pt.txt", "a")
      #fp.write("1")
      #fp.close()
      ret += [0.0, 0.0, 0.0]
   
    print ret
    return ret

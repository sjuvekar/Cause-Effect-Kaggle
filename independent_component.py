from sklearn.decomposition import FastICA
import numpy as np

def independent_component( (x, y) ):
    lenX = len(x)
    newX = np.array(x).reshape(lenX, 1)
    g = FastICA()
    g.fit(newX, y)
    ret = [0.0, 0.0, 0.0, 0.0]
    ret[0] = g.components_[0][0]
    ret[1] = g.get_mixing_matrix()[0][0]
    sources = g.sources_.flatten()
    ret[2] = max(sources)
    ret[3] = min(sources)
    return ret

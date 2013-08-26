import scipy.sparse
import scipy

def kernel_matrix(x):
  d = dict()
  for i in range(len(x)):
    try:
      d[x[i]].append(i)
    except:
      d[x[i]] = [i]

  l = len(x)
  m = scipy.sparse.lil_matrix( (l, l), dtype=float)
  for k in d.keys():
    for i in range(len(d[k])):
      for j in range(i+1, len(d[k])):
        m[d[k][i], d[k][j]] = 1
        m[d[k][j], d[k][i]] = 1
  
  return m


def hsic_score( x, y ):
  print len(x)
  ret_val = 0
  mX = kernel_matrix(x)
  mY = kernel_matrix(y)
  l = len(x)
  if False:
    ans = mX * mY
    ret_val = scipy.matrix.trace(ans.todense())
  else:
    H = scipy.sparse.identity(l) - 1./l * scipy.ones( (l, l) )
    ans = mX * H * mY * H
    ret_val = scipy.matrix.trace(ans)

  print ret_val
  return ret_val[0, 0] / ((l-1) ** 2)

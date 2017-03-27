"""
a module to generate training data.
"""
import numpy as np
def getCircleData(r,mx):
  """
  label = -1 if (i,j) in a circle.
  label = 1 if (i,j) out of the circle.
  r ==> radius
  mx ==> training set and test set size.
  """
  pass
def getLineData(k,mx):
  """
  let y = kx be a split line.
  if y - k*x > 0 ,label -1;
  if y - k*x < 0,label 1;
  mx ==> training and test set size.
  """
  exc = 20
  x,tx = np.random.rand(mx,2)*exc,np.random.rand(mx,2)*exc
  y,ty = np.zeros(mx),np.zeros(mx)
  for m in xrange(mx):
    if x[m,1] - x[m,0]*k > 0:y[m] = -1
    else:y[m] = 1
    if tx[m,1] - tx[m,0]*k > 0:ty[m] = -1
    else:ty[m] = 1
  return (x,y),(tx,ty)

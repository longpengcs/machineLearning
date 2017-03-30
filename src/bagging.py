"""
some method about bagging.
"""
import random
import numpy as np
from collections import Counter as cc
def randomBagging(x,y):
  randint = random.randint
  xx,yy = [],[]
  mx = len(y) - 1
  for m in y:
    m = randint(0,mx)
    xx.append(x[m])
    yy.append(y[m])
  if 'numpy' in str(type(x)):
    return np.array(xx),np.array(yy)
  return xx,yy

def predict(x,machines):
  y = [machine.predict(x) for machine in machines]
  y = cc(y)
  return max(y.items(),key=lambda x:x[1])[0]

def test(x,y):
  wrong = 0
  for n,m in enumerate(x):
    if predict(m) != y[n]:wrong += 1
  print 'fail rate: %d/%d = %.2f%%' %(wrong,len(y),wrong*100.0/len(y))

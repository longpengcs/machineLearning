"""
a general and strong KNN algorithm.
"""

from collections import Counter as cc
import numpy as np
from sklearn.metrics import classification_report as report

class knn(object):
  def __init__(self,x,y):
  #some init operator,such as normalization.
    self.x,self.mi,self.delta = self.normalize(x)
    self.y = y
  def normalize(self,x):
  #normalize feature space to 0~1
    mx = x.max(0)
    mi = x.min(0)
    x -= mi
    delta = mx - mi
  #in some feature space,max == min,this is a invalid feature.
    delta[delta == 0] = 1
    x /= delta
    return x,mi,delta
  def predict(self,x,k):
    #normalize x
    x -= self.mi
    x /= self.delta
    delta2 = np.sum((x - self.x)**2,axis=1)
    index = delta2.argsort()[:k]
    cls = cc(self.y[index])
    return max(cls.items(),key=lambda x:x[1])[0]
  def test(self,tx,ty,k):
    y = np.array([-1.0 for _ in ty])
    for n,m in enumerate(tx):
      y[n] = self.predict(m,k)
    print report(y,ty)
    y -= ty
    r = np.nonzero(y)[0]
    print 'fail rate: %d/%d = %.2f' %(len(r),len(y),len(r)*100.0/len(y))

if __name__ == '__main__':
  import loadMnist,argument
  arg = argument.getArgument()
  mnist = loadMnist.mnist()
  train,valid,test = mnist.getNumber(10,True,arg.mx)
  sol = knn(train[0],train[1])
  sol.test(valid[0][:1000],valid[1][:1000],arg.knn)

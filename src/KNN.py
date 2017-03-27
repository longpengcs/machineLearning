"""
a pratice KNN program.
data set and algorithm come from book machine learning in action.
"""
from collections import Counter as cc
import numpy as np
from sklearn.metrics import classification_report as report
class knn(object):
  def __init__(self,data,k):
    #init data set and k
    self.data,self.k = data,k
  def __predict(self,x):
    #calculate Euler distance (a - b)**2
    delta = x - self.x
    delta = np.sum(delta**2,axis=1)
    #sort and select k sample in the front.
    index = delta.argsort()[:self.k]
    y = self.y[index]
    #select the class which has max frequeuce.
    count = cc(y)
    return max(count.items(),key=lambda x:x[1])[0]
  def test(self,r):
    #random select r*data.size() data set to be training set.
    #the others is test set.
    np.random.shuffle(self.data)
    mx = int(r*len(self.data))
    x,y = self.data[:,:3],self.data[:,3]
    x = self.normalize(x)
    self.x,self.y,tx,ty = x[:mx],y[:mx],x[mx:],y[mx:]
    y = ty.copy()
    for n,m in enumerate(tx):
      y[n] = self.__predict(m)
    print report(y,ty)
    y -= ty
    #if r is nonzero,it is a wrong prediction.
    r = np.nonzero(y)[0]
    print 'fail rate is %d/%d = %.2f%%' %(len(r),len(y),len(r)*100.0/len(y))
  def normalize(self,x):
    #normalize feature to 0~1
    mx = x.max(0)
    mi = x.min(0)
    x -= mi
    x /= (mx - mi)
    return x

if __name__ == '__main__':
  data = np.loadtxt('../data/datingTestSet2.txt')
  sol = knn(data,20)
  sol.test(0.9)

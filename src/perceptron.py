"""
a very simply machine method perceptron.
"""
import loadMnist
import argument
from sklearn.metrics import classification_report as report
import numpy as np
from testing import wrongRate
class perceptron(object):
  def __init__(self,train):
    """
    train has two element x,y.
    x ==> a numpy.ndarray,every row is a sample.
    y ==> the mark array.
    """
    self.x,self.y = train
    m,n = self.x.shape
    self.w = np.zeros(n)
    self.b = 0.0
  def train(self,eta,batchs,test=None):
    x,y,w = self.x,self.y,self.w
    j = 0
    while j < batchs:
      for n,m in enumerate(x):
        if y[n]*(w.dot(m)+self.b) <= 0:
          w += eta*y[n]*m
          self.b += eta*y[n]
      print 'the %d batch over.' %j
      if test:
        self.test(test)
      j += 1
  def predict(self,x):
    y = np.dot(x,self.w) + self.b
    y[y>=0] = 1
    y[y<0] = -1
    return y
  def test(self,test):
    tx,ty = test
    y = self.predict(tx)
    wrongRate(y,ty)

if __name__ == '__main__':
  arg = argument.getArgument()
  mnist = loadMnist.mnist()
  train,valid,test = mnist.getNumber(1,True,mx=arg.mx)
  x,y = train
  y[y==0] = -1
  tx,ty = valid
  ty[ty==0] = -1
  sol = perceptron(train)
  sol.train(0.3,5,valid)

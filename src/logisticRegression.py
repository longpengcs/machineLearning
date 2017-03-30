"""a more powerful logitic regression classifiter,
this one can classify 0,1,2,3,4,5,6,7,8,9.
"""

import argument,theano
import sys
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax
import numpy as np
randn = np.random.randn
#import the module loadMnist
import loadMnist

class logisticRegression(object):
  def __init__(self,train,test,classNu,eta,lamda=0):
    #the dimension size if weight vector
    length = len(train[0][0])
    w = theano.shared(randn(length,classNu),name='w')
    b = theano.shared(randn(classNu),name='b')
    x = T.matrix('x')
    y = T.ivector('y')
    z = T.dot(x,w)+b
    h = softmax(z)
    pred = T.argmax(h,axis=1)
    cost = -T.mean(T.log(h[T.arange(y.shape[0]),y])) + lamda*(w**2).sum()
    gw,gb = T.grad(cost,[w,b])
    self.train = theano.function([x,y],updates=((w,w-eta*gw),(b,b-eta*gb)))
    self.prediction = theano.function([x],pred)
    self.train_x,self.train_y = train
    self.test_x,self.test_y = test
    #because the T.ivector is int32,but numpy default is int64,so we need turn it.
    self.train_y = np.array(self.train_y,dtype='int32')
    self.test_y = np.array(self.test_y,dtype='int32')
    self.train_max = len(self.train_y)
    self.test_max = len(self.test_y)
  def SGD(self,epochs,size,test=False):
    """SGD stochastic gradient descent,
       epochs is the max train times,
       size represent how much train data we input to function self.train.
    """
    for m in xrange(epochs):
      for n in xrange(0,self.train_max,size):
        self.train(self.train_x[n:n+size],self.train_y[n:n+size])
      if __name__ == '__main__':
        print 'accomplish %d epoch:' %(m)
      if test:
        succ = self.successful(self.prediction(self.test_x),self.test_y)
        print 'successful rate in valid set {}/{} = {:.2f}%'.format(succ,
              self.test_max,succ*100.0/self.test_max)
        succ = self.successful(self.prediction(self.train_x),self.train_y)
        print 'successful rate in training set {}/{} = {:.2f}%'.format(succ,
              self.train_max,succ*100.0/self.train_max)
  def successful(self,pred,real):
    """calculate the successful rate of our classifiter now."""
    return len(pred) - np.sum((pred ^ real) > 0)

if __name__ == '__main__':
  arg = argument.getArgument()
  mnist = loadMnist.mnist()
  train,valid,test = mnist.getNumber(10,ranges=True)
  eta,lamda,potchs,size,mx = arg.eta,arg.lamda,arg.potchs,arg.size,arg.mx
  print arg
  train = train[0][:mx],train[1][:mx]
  test = test[0][:mx],test[1][:mx]
  valid = valid[0][:mx],valid[1][:mx]
  #classifier = logisticRegression(train,test,10,eta,lamda)
  classifier = logisticRegression(train,valid,10,eta,lamda)
  #classifier.train(classifier.train_x,classifier.train_y)
  classifier.SGD(potchs,size,True)

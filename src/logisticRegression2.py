"""
using logistic regression to classify 0 and 1 in mnist.
"""
import sys
import theano
import theano.tensor as T
import numpy as np
randn = np.random.randn
#import the module loadMnist
import loadMnist

class logisticRegression(object):
  def __init__(self,train,test,eta,lamda=0):
    #the dimension size if weight vector
    length = len(train[0][0])
    w = theano.shared(randn(length),name='w')
    b = theano.shared(0.,name='b')
    x = T.matrix('x')
    y = T.ivector(name='y')
    h = 1/(1+T.exp(-T.dot(x,w)+b))
    #cost function
    xent = -y*T.log(h) - (1-y)*T.log(1-h)
    #add regulization
    cost = xent.mean() + lamda*(w**2).sum()
    #get gradient.
    gw,gb = T.grad(cost,[w,b])
    pred = h > 0.5
    #traning function,everytime we call this function,it will calculate the 
    #gradient,and update w,b.
    self.train = theano.function([x,y],pred,updates=((w,w-eta*gw),(b,b-eta*gb)))
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
      print 'accomplish %d epoch:',m
      if test:
        succ = self.successful()
        print 'successful rate {}/{} = {:.2f}%'.format(succ,
              self.test_max,succ*100.0/self.test_max)
  def successful(self):
    """calculate the successful rate of our classifiter now."""
    pred = self.prediction(self.test_x)
    return self.test_max - np.sum(pred ^ self.test_y)

if __name__ == '__main__':
  mnist = loadMnist.mnist()
  train,valid,test = mnist.getNumber(1,ranges=True)
  print np.sum(train[1]),len(train[1]),np.sum(test[1]),len(test[1])
  print np.sum(train[1]^1),len(train[1]),np.sum(test[1]^1),len(test[1])
  if len(sys.argv) != 6:
    print 'please give training arguments,the order is:'
    print 'eta/lambda/potch number/SGD length/how much data to use'
    sys.exit()
  else:print sys.argv[1:]
  eta,lamda = map(float,sys.argv[1:3])
  potchs,length,mx = map(int,sys.argv[3:])
  train = train[0][:mx],train[1][:mx]
  test = test[0][:mx],test[1][:mx]
  valid = valid[0][:mx],valid[1][:mx]
  #classifier = logisticRegression(train,test,eta,lamda)
  classifier = logisticRegression(train,valid,eta,lamda)
  #classifier.train(classifier.train_x,classifier.train_y)
  classifier.SGD(potchs,length,True)

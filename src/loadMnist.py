"""
some class and functions to load the data set mnist.
"""
import gzip
import cPickle
import numpy as np
class mnist(object):
  def __init__(self,name='../data/mnist.pkl.gz'):
    f = gzip.open(name, 'rb')
    #training set,validation set,test set.
    train,valid,test = cPickle.load(f)
    self.train_x,self.train_y = train
    self.valid_x,self.valid_y = valid
    self.test_x,self.test_y = test
  #get a number's data set(such as 0)
  #if ranges == True,it will return the data set of numbers which less or equal num
  def getNumber(self,num,ranges=False,mx=False):
    return self.getNumAboutASet(num,self.train_x,self.train_y,ranges,mx),\
           self.getNumAboutASet(num,self.valid_x,self.valid_y,ranges,mx),\
           self.getNumAboutASet(num,self.test_x,self.test_y,ranges,mx)
  def getNumAboutASet(self,num,x,y,ranges=False,mx=False):
  #np.where will filter these data which not fill condition.
    if(ranges):numIndex = np.where(y<=num)[0]
    else:numIndex = np.where(y==num)[0]
    if mx:numIndex = numIndex[:mx]
    return x[numIndex],y[numIndex]

if __name__ == '__main__':
  mm = mnist()
  one = mm.getNumber(1)
  two = mm.getNumber(1,True)
  print one[0][1][:10]
  print two[0][1][:10]

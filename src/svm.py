"""
a heuristic method to active SVM.
SMO SEQUENTIAL MINIMAL OPTMIZATION.
"""
import numpy as np
import argument
import loadMnist
class svm(object):
  def __init__(self,train,test):
    self.x,self.y = train
    self.test_x,self.test_y = test
    self.m,self.n = self.x.shape
    self.alpha = np.zeros(self.m)
    self.b = 0.0
    self.cache = np.zeros(self.m)
    #self.getK(self.normalKernel)
    self.getK(self.gaussKernel)
  def train(self,c,bias,batchs):
    x,y,k,alpha = self.x,self.y,self.k,self.alpha
    batch = 0;FULL = True;alphaChanged = 0
    while batch < batchs and (FULL or alphaChanged):
      alphaChanged = 0
      if FULL:
        for i in xrange(self.m):
          Ei = self.EK(i)
          if (Ei*y[i] < -bias and alpha[i] < c) or\
             (Ei*y[i] > bias and alpha[i] > 0):
            j,Ej = self.selectJ(i,Ei)
            alphaChanged += self.update(i,j,Ei,Ej,c)
      else:
        inn = np.nonzero( (alpha > 0) & (alpha < c) )[0]
        for i in inn:
          Ei = self.EK(i)
          if (Ei*y[i] < -bias and alpha[i] < c) or\
              (Ei*y[i] > bias and alpha[i] > 0):
             j,Ej = self.selectJ(i,Ei)
             alphaChanged += self.update(i,j,Ei,Ej,c)
      if FULL:FULL = 0
      elif not alphaChanged:FULL = 1
      batch += 1
      print "iter %d batch over!!!" %batch
      self.test(self.test_x,self.test_y)
      self.test(x,y)
  def update(self,i,j,Ei,Ej,c):
    x,y,k,alpha = self.x,self.y,self.k,self.alpha
    alphaI,alphaJ = alpha[i],alpha[j]
    if y[i] != y[j]:
      L = max(0,alphaJ - alphaI)
      H = min(c,c+alphaJ - alphaI)
    else:
      L = max(0,alphaI+alphaJ-c)
      H = min(c,alphaI+alphaJ)
    if L == H:print 'L == H';return 0
    eta = k[i,i] + k[j,j] - 2*k[i,j]
    if eta <= 0:print 'eta <= 0';return 0
    alpha[j] += y[j]*(Ei - Ej)/eta
    if alpha[j] > H:alpha[j] = H
    if alpha[j] < L:alpha[j] = L
    if alpha[j] - alphaJ < 0.0001:
      print 'alphaJ has no changed'
      return 0
    alpha[i] += y[i]*y[j]*(alphaJ-alpha[j])
    #print self.alpha
    #print self.y
    #print 'b EI EJ alphaI alpha[i] alphaJ alpha[j] y[i] y[j]'
    #print self.b,Ei,Ej,alphaI,alpha[i],alphaJ,alpha[j],y[i],y[j]
    b1 = self.b - Ei - y[i]*(alphaI-alpha[i])*k[i,i] - y[j]*(alpha[j] - alphaJ)*k[i,j]
    b2 = self.b - Ej - y[i]*(alphaI-alpha[i])*k[i,j] - y[j]*(alpha[j] - alphaJ)*k[j,j]
    if 0 < alphaI < c:self.b = b1
    elif 0 < alphaJ < c:self.b = b2
    else:self.b = (b1+b2)/2.0
    return 1
  def selectJ(self,i,Ei):
    self.cache[i] = 1
    valid = np.nonzero(self.cache)[0]
    mxDelta = -1;Ej = -1;mxJ = -1
    if len(valid > 1):
      for j in valid:
        if j == i:continue
        Ek = self.EK(j)
        delta = abs(Ek - Ei)
        if delta > mxDelta:Ej,mxJ,maDelta = Ek,j,delta
      return mxJ,Ej
    else:
      return self.randSelextJ(i)
  def randSelextJ(self,i):
    j = np.random.randint(0,self.m)
    while j == i:j = np.random.randint(0,self.m)
    return j,self.EK(j)

  def EK(self,k):
    fxi = np.dot(self.alpha,self.y*self.k[k])+self.b
    return fxi - self.y[k]
  def getK(self,kernel):
    self.k = np.zeros((self.m,self.m))
    for i in xrange(self.m):
      for j in xrange(i,self.m):
        self.k[i,j] = kernel(self.x[i],self.x[j])
        self.k[j,i] = self.k[i,j]
  def normalKernel(self,x,y):
    return x.dot(y)
  def gaussKernel(self,x,y,k=30):
    delta = x - y
    z = -delta.dot(delta)
    return np.exp(z/(k**2))
  def test(self,x,y):
    fx = np.dot(self.alpha*self.y,self.k)+self.b
    fx[fx <= 0] = -1;fx[fx > 0] = 1
    fx -= y;fx = np.abs(fx)
    fx = fx > 0.001
    print 'fail rate is %d/%d = %.2f%%' %(np.sum(fx),len(fx),np.sum(fx)*100.0/len(fx))

if __name__ == '__main__':
  arg = argument.getArgument()
  mnist = loadMnist.mnist()
  train,vaild,test = mnist.getNumber(10,ranges=True,mx=arg.mx)
  x,y = train
  tx,ty = test
  y[y!=1] = -1
  ty[ty!=1] = -1
  sol = svm(train,test)
  sol.train(20,0.0001,arg.potchs)

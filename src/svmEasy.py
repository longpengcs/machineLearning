"""
a easy version of algorithm svm.
1.find the first alpha which break KKT.
2.random select a second alpha to opt.
"""
import numpy as np
import loadMnist
import argument
import random
class svm(object):
  def __init__(self,train,test,c=1.0,bias=0.01,potchs = 20):
    x,y = train
    test_x,test_y = test
    m,n = x.shape
    a = np.ones((m,1))
    b = 0;it = 0
    while it < potchs:
      alphaPairChanged = 0
      for i in xrange(m):
        print a
        x[:10,:10] = 1
        print y
        print x[:10,:10]
        w = np.sum(((a.T*y).T)*x,axis=0)
        print w[:10]
        return
        fxi = w.dot(x[i]) + b
        ei = fxi - y[i]
        if (y[i]*ei < -bias and a[i] < c) or (y[i]*ei > bias and a[i] > 0):
          j = self.randomSelect(i,m)
          fxj = w.dot(x[j]) + b
          ej = fxj - y[j]
          ai,aj = a[i],a[j]
          if y[i] != y[j]:
            L = max(0,aj-ai)
            H = min(c,c+aj-ai)
          else:
            L = max(0,aj+ai-c)
            H = min(c,a[j]+a[i])
          if L == H:print 'L==H';continue
          eta = 2 * x[i].dot(x[j]) - x[i].dot(x[i]) - x[j].dot(x[j])
          if eta >=0 :print 'eta >= 0';continue
          a[j] -= y[j]*(ei-ej)/eta
          if a[j] < L:a[j] = L
          if a[j] > H:a[j] = H
          if(abs(a[j] - aj) < 0.00001):print 'j not moving enough';continue
          a[i] += y[i]*y[j]*(aj-a[j])
          b1 = b - ei - y[i] * (a[i] - ai) * x[i].dot(x[i]) - y[j] * ( a[j] - aj) * x[i].dot(x[j])
          b2 = b - ej - y[i] * (a[i] - ai) * x[i].dot(x[j]) - y[j] * ( a[j] - aj) * x[j].dot(x[j])
          if 0 < a[i] and c > a[i]:b = b1
          elif 0 < a[j] and c > a[j]:b = b2
          else:b = (b1+b2)/2.0
          alphaPairChanged+=1
          print "iter: %d i:%d,pairs changed %d" %(it,i,alphaPairChanged)
      if alphaPairChanged == 0:it += 1
      else:it = 0
      print "iteration number: %d" % it
      w = np.sum(((a.T*y).T)*x,axis=0)
      self.test(w,b,test_x,test_y)
      self.test(w,b,x,y)
  def test(self,w,b,tx,ty):
    y = tx.dot(w) + b
    y[y>=0] = 1;y[y<0] = -1
    y -= ty
    y = (y == 0)
    print "success rate %d/%d = %.2f" %(np.sum(y),len(y),np.sum(y)*100.0/len(y))
  def randomSelect(self,i,m):
    j = i
    while j == i: j = random.randint(0,m-1)
    return j
if __name__ == '__main__':
  mnist = loadMnist.mnist()
  arg = argument.getArgument()
  train,valid,test = mnist.getNumber(1,True)
  x,y = train
  t_x,t_y = test
  x,y = x[:arg.mx],y[:arg.mx]
  t_x,t_y = t_x[:arg.mx],t_y[:arg.mx]
  y[y==0] = -1
  t_y[t_y==0] = -1
  SVM = svm((x,y),(t_x,t_y),1,bias=arg.bias,potchs=arg.mx)

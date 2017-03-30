"""
using bagging algorithm to improve logicial regression.
"""
from logisticRegression import logisticRegression as LR
import bagging
from sklearn.metrics import classification_report as report
import numpy as np
def train(training,maxBase=4):
  x,y = training
  datas = [bagging.randomBagging(x,y) for _ in xrange(maxBase)]
  machines = [LR(training,training,10,0.3) for x,y in datas]
  for machine in machines:
    machine.SGD(20,10)
  return machines

def test(x,y,machines):
  result = [machine.prediction(x) for machine in machines]
  length = len(y)
  for n,m in enumerate(result):
    print 'the %dth predicter\'s result:' %n
    print report(m,y)
    r = len(np.nonzero(m - y)[0])
    print 'fail rate: %d/%d = %.2f%%\n' %(r,length,r*100.0/length)
  yy = np.zeros((length,10),dtype=int)
  for r in result:
    for n,m in enumerate(r):yy[n,m] += 1
  r = np.argmax(yy,axis=1)
  print 'the integrator result:'
  print report(r,y)
  index = np.nonzero(r-y)[0]
  index = index[:10]
  print yy[index],y[index]
  r = len(np.nonzero(r-y)[0])
  print 'fail rate: %d/%d = %.2f%%' %(r,length,r*100.0/length)

if __name__ == '__main__':
  import loadMnist
  mnist = loadMnist.mnist()
  training,vaild,testing = mnist.getNumber(10,ranges=True,mx=10000)
  machines = train(training,11)
  test(vaild[0],vaild[1],machines)

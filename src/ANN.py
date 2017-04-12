"""
using tersorflow to active ann.
"""
import tensorflow as tf
import numpy as np

class ANN(object):
  def train(self,x,y,vx,vy,length,botchs):
    yy,vyy = y,vy
    ny,nvy = np.zeros((len(y),length),dtype=int),np.zeros((len(vy),length),dtype=int)
    for n,m in enumerate(y):ny[n,m] = 1
    for n,m in enumerate(vy):nvy[n,m] = 1
    y,vy = ny,nvy
    x,y,vx,vy = map(tf.constant,[x,y,vx,vy])
    m,n = x.shape
    n = 28*28
    w = tf.Variable(tf.truncated_normal([n,length]))
    b = tf.Variable(tf.zeros([length]))
    z = tf.matmul(x,w) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=z))
    opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    tp = tf.nn.softmax(z)
    vp = tf.nn.softmax(tf.matmul(vx,w)+b)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      print 'Initialized'
      for m in xrange(botchs):
        _,l,pred = sess.run([opt,loss,tp])
        if m % 100 == 0:
          print 'loss at step %d: %f' %(m,l)
          print 'Train accuracy: %.2f%%' % self.accuracy(pred,yy)
          print 'Vaild accuracy: %.2f%%' % self.accuracy(sess.run(vp),vyy)
  def accuracy(self,pred,y):
    return 100.0 * np.sum(np.argmax(pred,axis=1) == np.argmax(y,axis=1)) / len(y)
  def SGD(self,x,y,vx,vy,length,botchs,batchSize,lamda=0.1):
    ny,nvy = np.zeros((len(y),length),dtype=int),np.zeros((len(vy),length),dtype=int)
    for n,m in enumerate(y):ny[n,m] = 1
    for n,m in enumerate(vy):nvy[n,m] = 1
    y,vy = ny,nvy
    n = 28*28
    xx,yy,vxx,vyy = x,y,vx,vy
    x = tf.placeholder(tf.float32,[batchSize,n])
    y = tf.placeholder(tf.float32,[batchSize,length])
    vx = tf.constant(vx)
    w = tf.Variable(tf.truncated_normal([n,length]))
    b = tf.Variable(tf.zeros([length]))
    z = tf.matmul(x,w) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=z)) + lamda*tf.nn.l2_loss(w)
    opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    tp = tf.nn.softmax(z)
    vp = tf.nn.softmax(tf.matmul(vx,w)+b)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      print 'Initialized'
      for m in xrange(botchs):
        offset = (m * batchSize) % (len(yy))
        bx = xx[offset:(offset+batchSize),:]
        by = yy[offset:(offset+batchSize),:]
        feed_dict = {x : bx,y : by}
        _,l,pred = sess.run([opt,loss,tp],feed_dict=feed_dict)
        if m % 100 == 0:
          print 'loss at step %d: %f' %(m,l)
          print 'Train accuracy: %.2f%%' % self.accuracy(pred,by)
          print 'Vaild accuracy: %.2f%%' % self.accuracy(vp.eval(),vyy)
  def SGDWithDropout(self,x,y,vx,vy,length,botchs,batchSize,drop=0.5):
    ny,nvy = np.zeros((len(y),length),dtype=int),np.zeros((len(vy),length),dtype=int)
    for n,m in enumerate(y):ny[n,m] = 1
    for n,m in enumerate(vy):nvy[n,m] = 1
    y,vy = ny,nvy
    n = 28*28
    xx,yy,vxx,vyy = x,y,vx,vy
    drop = tf.constant(drop)
    x = tf.placeholder(tf.float32,[batchSize,n])
    x = tf.nn.dropout(x,drop)
    y = tf.placeholder(tf.float32,[batchSize,length])
    vx = tf.constant(vx)
    w = tf.Variable(tf.truncated_normal([n,length]))
    b = tf.Variable(tf.zeros([length]))
    z = tf.matmul(x,w) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=z))
    opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    tp = tf.nn.softmax(z)
    vp = tf.nn.softmax(tf.matmul(vx,w)+b)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      print 'Initialized'
      for m in xrange(botchs):
        offset = (m * batchSize) % (len(yy))
        bx = xx[offset:(offset+batchSize),:]
        by = yy[offset:(offset+batchSize),:]
        feed_dict = {x : bx,y : by}
        _,l,pred = sess.run([opt,loss,tp],feed_dict=feed_dict)
        if m % 100 == 0:
          print 'loss at step %d: %f' %(m,l)
          print 'Train accuracy: %.2f%%' % self.accuracy(pred,by)
          print 'Vaild accuracy: %.2f%%' % self.accuracy(vp.eval(),vyy)
if __name__ == '__main__':
  import loadMnist
  mnist = loadMnist.mnist()
  train,vaild,test = mnist.getNumber(10,True,1000)
  x,y = train
  vx,vy = vaild
  sol = ANN()
  sol.SGD(x,y,vx,vy,10,1000,20,0)
  sol.SGDWithDropout(x,y,vx,vy,10,1000,20,0.5)

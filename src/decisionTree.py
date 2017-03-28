"""
a pratice of decision tree.
data set and algorithm come from book machine learning in action.
"""
from collections import Counter as cc
import numpy as np
from sklearn.metrics import classification_report as report

class decisionTree(object):
  def __init__(self,train,featureName=None):
    self.x,self.y = train
    self.tree = self.creativeTree(self.x,self.y,set(range(self.x.shape[1])))
  def calcEntropy(self,y):
    #calculate the information entropy of a set
    count = cc(y)
    values = np.array(count.values())
    values = values*1.0 / len(y)
    return -np.sum(values*np.log(values))
  def creativeTree(self,x,y,featureSet):
    #recurse creative a decision tree.
    #all sample belong to one class,return.
    if len(set(y)) == 1:return y[0]
    #no feature to choose,choose a class which occour most time.
    if not featureSet:
      count = cc(y).items()
      return max(count,key=lambda x:x[1])[0]
    #choose a best split feature
    feature,datas = self.chooseBestFeature(x,y,featureSet)
    #mark this node 
    tree = {'feature':feature}
    featureSet.remove(feature)
    for data in datas:
      xx,yy,val = data
      #recurse creative tree.
      tree[val] = self.creativeTree(xx,yy,featureSet)
    featureSet.add(feature)
    return tree
  def judge(self,x,tree=None):
    #judge a sample's class using a decision tree.
    if tree == None:tree = self.tree
    while isinstance(tree,dict):
      #find this node's mark number and get relate feature value.
      val = x[tree['feature']]
      tree = tree[val]
    return tree
  def test(self,x,y):
    r = y.copy()
    for n,m in enumerate(x):
      r[n] = self.judge(m)
    l = len(y)
    c = l - np.sum(r == y)
    print 'fail rate is: %d/%d = %.2f%%' %(c,l,c*100.0/l)
    
  def chooseBestFeature(self,x,y,featureSet):
    minEntropy = 1000000;minFeature = -1;split = None
    size = len(y)*1.0
    #find the max information gain is equal to find the min 
    #information entropy of the split sets.
    for m in featureSet:
      datas = self.splitDataByFeature(x,y,m)
      entropy = 0
      for n in datas:
        entropy += len(n[1])/size*self.calcEntropy(n[1])
      if entropy < minEntropy:
        minEntropy = entropy
        minFeature = m
        split = datas
    return minFeature,split
  def splitDataByFeature(self,x,y,feature):
    #split a data set by a feature.
    xf = x[:,feature]
    values = set(xf)
    datas = []
    for m in values:
      index = np.where(xf==m)
      datas.append((x[index],y[index],m))
    return datas
  def display(self,tree=None):
    if not tree:tree = self.tree
    pass

if __name__ == '__main__':
  x = np.loadtxt('../data/lenses.txt',usecols=(0,1,2,3),dtype='str')
  y = np.loadtxt('../data/lenses.txt',usecols=(4,),dtype='str')
  tree = decisionTree((x,y))
  tree.test(x,y)

"""
the classficiation and regression tree.
"""

from collections import Counter as cc
import numpy as np
from sklearn.metrics import classification_report as report

class CART(object):
  def __init__(self):
    #is the a leaf node
    self.leaf = False
  def train(self,x,y,maxS,res):
    """
    x ==> input array.
    y ==> mark array.
    maxp ==> max sample sample point in a leaf node.
    res ==> resolution of split a node.
    """
  def Gini(self,y):
    marks = cc(y)
    length = float(len(y))
    last = 0
    for m in marks:
      last += (marks[m]/length)**2
    return 1 - last
  def chooseBestSplit(self,x,y,features):
    

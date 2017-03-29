"""
a pratice of native bayes algorithm.
data set and algorithm come from book machine learning in action.
"""
from collections import Counter as cc
import numpy as np
from sklearn.metrics import classification_report as report

class nativeBayes(object):
  def __init__(self,train,dic):
    """
    train is a tuple which has 2 element.
    x --> muti list of words(or a numpy str matrix.)
    y --> a list of mark of every list in x,it's element mush be int or str,float maybe make a error.
    """
    self.x,self.y = train
    self.word2Index = {word:n for n,word in enumerate(dic)}
    self.length = len(self.word2Index)
    self.mark2Index = {mark:n for n,mark in enumerate(set(self.y))}
    self.index2Mark = {n:mark for n,mark in enumerate(set(self.y))}
    yy = np.zeros(len(self.y),dtype=int)
    for n,m in enumerate(self.y):yy[n] = self.mark2Index[m]
    self.y = yy
    self.x = self.wordsList2Matrix(self.x)
  
  def wordsList2Matrix(self,x,dic=None):
    if not dic:dic = self.word2Index
    mtx = np.zeros((len(x),self.length),dtype=int)
    for n,words in enumerate(x):
      for word in words:
        if word in dic:
          m = dic[word]
          mtx[n,m] = 1
    return mtx

  def train(self):
    cn = max(self.y)
    mtx = np.ones((cn+1,self.length))
    for m in xrange(cn+1):
      index = np.where(self.y == m)[0]
      freq = np.sum(self.x[index],axis=0)
      mtx[m,:] = freq + 1
    full = np.sum(mtx,axis=0)
    mtx /= full
    mtx = np.log(mtx)
    count = cc(self.y)
    freq = np.zeros(cn+1)
    full = len(self.y)
    for n,m in count.items():freq[n] = m*1.0/full
    freq = np.log(freq)
    self.likehoodOfWordInClass = mtx
    self.likehoodOfClass = freq
  def predict(self,words):
    vec = self.words2vec(words)
    index = np.where(vec == 1)[0]
    result = np.sum(self.likehoodOfWordInClass[:,index],axis=1) + self.likehoodOfClass
    return self.index2Mark[np.argmax(result)]

  def words2vec(self,words,dic=None):
    if not dic:dic = self.word2Index
    vec = np.zeros(self.length,dtype=int)
    for word in words:
      if word in dic:vec[dic[word]] = 1
    return vec
  def test(self,x,y):
    wrong = 0
    for n,m in enumerate(x):
      if self.predict(m) != y[n]:wrong += 1
    print 'fail rate: %d/%d = %.2f%%' %(wrong,len(y),wrong*100.0/len(y))

import re
def sentence2words(sentence):
  condition = re.compile('[a-zA-z0-9]+')
  return [word for word in re.findall(condition,sentence) if len(word) > 3]

def crossTest(place1,place2,mx=20):
  import random
  wordsList = [_ for _ in place1]
  wordsList.extend(place2)
  dic = cc()
  for words in wordsList:dic += cc(words)
  dic = sorted(dic.items(),key=lambda x:x[1],reverse=True)
  dic = [word for word,nu in dic[60:]]
  y = [0 for _ in xrange(len(place1))]
  y.extend(1 for _ in xrange(len(place2)))
  tx = [];ty = []
  for m in xrange(mx):
    m = random.randint(0,len(wordsList)-1)
    tx.append(wordsList[m])
    wordsList.remove(wordsList[m])
    ty.append(y[m])
    y.remove(y[m])
  sol = nativeBayes((wordsList,y),dic)
  sol.train()
  wrong = 0
  print ty
  for n,m in enumerate(tx):
    if sol.predict(m) != y[n]:wrong += 1
  print 'fail rate: %d/%d = %.2f%%' %(wrong,len(ty),wrong*100.0/len(ty))
if __name__ == '__main__':
  import pickle
  data = open('../data/rss.pkl','rb')
  ny,rs = pickle.load(data)
  data.close()
  ny = [sentence2words(ny['entries'][m]['summary']) for m in xrange(len(ny['entries']))]
  rs = [sentence2words(rs['entries'][m]['summary']) for m in xrange(len(rs['entries']))]
  crossTest(ny,rs)

"""
some method about input argument.
"""

import argparse

def getArgment():
  argment = argparse.ArgumentParser(description='the input argment of machine learning test.')
  argment.add_argument('-e','--eta',dest='eta',default=0.1,help='the training step length.')
  argment.add_argument('-l','--lambda',dest='lamda',default=0,help='the regularation coefficient.')
  argment.add_argument('-m','--max',dest='mx',default=2**30,help='how much training data to be used.')
  argment.add_argument('-p','--potchs',dest='potchs',default=5,help='how much times we will training.')
  argment.add_argument('-s','--size',dest='size',default=10,help='SGD mini batch size.')
  argment = argment.parse_args()
  return argment

if __name__ == '__main__':
  arg = getArgment()
  print arg

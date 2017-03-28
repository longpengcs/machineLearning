"""
some method about input argument.
"""

import argparse

def getArgument():
  argument = argparse.ArgumentParser(description='the input argment of machine learning test.')
  argument.add_argument('-e','--eta',dest='eta',default=0.1,help='the training step length.')
  argument.add_argument('-l','--lambda',dest='lamda',default=0,help='the regularation coefficient.')
  argument.add_argument('-m','--max',dest='mx',default=2**30,help='how much training data to be used.')
  argument.add_argument('-p','--potchs',dest='potchs',default=5,help='how much times we will training.')
  argument.add_argument('-s','--size',dest='size',default=10,help='SGD mini batch size.')
  argument.add_argument('-b','--bias',dest='bias',default=0.001,help='SGD mini batch size.')
  argument.add_argument('-k','--knn',dest='knn',default=10,help='KNN k top argument.')
  argument = argument.parse_args()
  argument.eta = float(argument.eta)
  argument.lamda = float(argument.lamda)
  argument.mx = int(argument.mx)
  argument.potchs = int(argument.potchs)
  argument.size = int(argument.size)
  argument.bias = float(argument.bias)
  argument.knn = int(argument.knn)
  return argument
if __name__ == '__main__':
  arg = getArgment()
  print arg

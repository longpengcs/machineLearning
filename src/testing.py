"""
some method about test.
"""

def wrongRate(r,y):
  """
  r ==> predict result.
  y ==> real result.
  numpy.array or python list.
  """
  wrong = 0
  for n,m in enumerate(r):
    if y[n] != m:wrong += 1
  length = len(y)
  print 'fail rate: %d/%d = %.2f%%' %(wrong,length,wrong*100.0/length)

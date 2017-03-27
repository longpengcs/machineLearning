"""
using svm in  scikit-learn.
"""

import loadMnist
import argument
from sklearn import svm
from sklearn.metrics import classification_report as report
arg = argument.getArgument()
mnist = loadMnist.mnist()
train,valid,test = mnist.getNumber(10,ranges=True,mx=arg.mx)
sol = svm.SVC()
sol.fit(train[0],train[1])
result = sol.predict(valid[0])
print report(result,valid[1])

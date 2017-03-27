"""
using logitical regression in scikit-learn.
"""
import loadMnist
import argument
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report as report
arg = argument.getArgument()
mnist = loadMnist.mnist()
train,valid,test = mnist.getNumber(10,ranges=True,mx=arg.mx)
sol = LR()
sol.fit(train[0],train[1])
result = sol.predict(valid[0])
print report(result,valid[1])

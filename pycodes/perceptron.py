# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:34:14 2019

@author: KC
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
iris_X =iris.data[:100,[1,3]]
iris_y=iris.target[:100]
colors=['red','blue','green','pink','yellow']
for i in range(len(iris_X[0])):
    plt.scatter(iris_X[iris_y==i,0], iris_X[iris_y==i,1], color=colors[i], marker ='+',linewidths=2)
plt.xlabel('sepal width (cm)')
plt.ylabel('petal width (cm)')
plt.title('Iris Data set')
plt.legend()
plt.show()

# implementing  Perceptron 
from sklearn.linear_model import Perceptron
perceptron_model = Perceptron(loss="log",penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False)

perceptron_model.fit(iris_X,iris_y)
w0 =perceptron_model.intercept_
w1,w2=perceptron_model.coef_[0,:]
x1=np.array((iris_X[:,0].min(),iris_X[:,0].max()))
x2 = -( w1*x1 +w0)/w2
plt.plot(x1,x2,color='green')

for i in range(15):
    perceptron_model = Perceptron(loss="log",penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=i, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False) 
    perceptron_model.fit(iris_X,iris_y)
    w0 =perceptron_model.intercept_
    w1,w2=perceptron_model.coef_[0,:]
    x1=np.array((iris_X[:,0].min(),iris_X[:,0].max()))
    x2 = -( w1*x1 +w0)/w2
    plt.plot(x1,x2,color='green')


from sklearn.linear_model import SGDClassifier
sdgclassifier = SGDClassifier(loss="log", penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, n_jobs=None, random_state=None, learning_rate="constant", eta0=0.25, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)

sdgclassifier_model =sdgclassifier.fit(iris_X,iris_y)
sdgw0=sdgclassifier_model.intercept_[0]
sdgw1,sdgw2=sdgclassifier_model.coef_[0,:]
sdgx2 = -( sdgw1*x1 +sdgw0)/sdgw2


plt.plot(x1,sdgx2,color='orange')



from sklearn.linear_model import SGDClassifier
sdgpclassifier = SGDClassifier(loss="perceptron", penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, n_jobs=None, random_state=None, learning_rate="constant", eta0=0.25, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)

sdgclassifier_model =sdgpclassifier.fit(iris_X,iris_y)
sdgpw0=sdgclassifier_model.intercept_[0]
sdgpw1,sdgpw2=sdgclassifier_model.coef_[0,:]
sdgpx2 = -( sdgw1*x1 +sdgw0)/sdgw2
plt.plot(x1,sdgpx2,color='pink')

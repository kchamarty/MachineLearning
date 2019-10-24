# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:39:30 2019

@author: KC
"""
import pandas as pd

fpath=r'C:\Users\KC\Documents\Metro College\ML\data'
fname=r'mnist.csv'
file= '{}\{}'.format(fpath,fname)

mnist_df =pd.read_csv(file,header=None)
mnist_X=mnist_df.iloc[:,1:]
mnist_y=mnist_df.iloc[:,0]
mnist_y.value_counts


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(mnist_X, mnist_y, random_state=9,test_size=0.75,stratify=mnist_y)

##  ------- KNN Classifier -------------------------
from sklearn.neighbors import KNeighborsClassifier
KNC_model = KNeighborsClassifier(n_neighbors=5)
KNC_model.fit(X_train,y_train)
y_pred = KNC_model.predict(X_test)
acc_score = accuracy_score(y_pred,y_test)

##  ------- SDGC Clssifier for LR -------------------------
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(mnist_X, mnist_y, random_state=9,test_size=0.25,stratify=mnist_y)
LR_model.fit(X_train,y_train)
y_pred=LR_model.predict(X_test)

from sklearn.linear_model import SGDClassifier
sdgclassifier = SGDClassifier(loss="log", penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, n_jobs=None, random_state=None, learning_rate="constant", eta0=0.25, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)

sdgclassifier_model =sdgclassifier.fit(mnist_X,mnist_y)
sdgw0=sdgclassifier_model.intercept_[0]
sdgw1,sdgw2=sdgclassifier_model.coef_[0,:]


from sklearn.linear_model import Perceptron
perceptron_model = Perceptron(loss="log",penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False)






from sklearn.model_selection import GridSearchCV
sdgclassifier = SGDClassifier(loss="log")
param_grid = {'eta0':[0.001,0.01,0.1],'learning_rate':['constant','optimal','invscaling']}

grid=GridSearchCV(sdgclassifier,param_grid)
grid.fit(X_train,y_train)
grid.param_grid
grid.best_params_
grid.best_estimator_

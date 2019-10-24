# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:59:51 2019

@author: KC
"""


from sklearn.datasets import load_digits

#import data and extract feature set and targets separately

digits =load_digits()
dg_X=digits.data
dg_y=digits.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(dg_X,dg_y,random_state=0)
len(X_train)
len(X_train[0])
#importing model and training it.

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))



#######################################
X_train,X_test,y_train,y_test =train_test_split(dg_X,dg_y,random_state=0)
for i in range(1,21):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(' When number of  neighbors is : {}, Accuracy is {}'.format(i,accuracy_score(y_pred,y_test)))
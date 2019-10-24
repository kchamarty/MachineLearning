# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:55:35 2019

@author: 16479
"""

# Import our libraries

import pandas as pd
import matplotlib.pyplot as plt
import time

fpath=r'C:\Users\KC\Documents\Metro College\ML\data'
fname=r'HTRU_2.csv'
file= '{}\{}'.format(fpath,fname)

'''1. Mean of the integrated profile.
	2. Standard deviation of the integrated profile.
	3. Excess kurtosis of the integrated profile.
	4. Skewness of the integrated profile.
	5. Mean of the DM-SNR curve.	
    6. Standard deviation of the DM-SNR curve.
	7. Excess kurtosis of the DM-SNR curve.
	8. Skewness of the DM-SNR curve.
'''
columns=['ip_mean', 'ip_std','ip_xskurt', 'ip_skew', 'dm_snr_mean', 'dm_snr_std', 'dm_snr_xskurt', 'dm_skew','target']

df=pd.read_csv(file,header=None,names=columns)

nrows,ncols=df.shape
df.head()
#checking columns and total records and null Values
df.info()
df_columns=df.columns
print(df_columns)

des=df.describe()
df.tail()

# Copy of the dataframe
original_df = df.copy()
df.dtypes


df['target'].value_counts()




"""Data cleansing"""
#Do we have duplicated data?
#Duplicate Data
df_nodup=df.drop_duplicates()
print(df.shape,df_nodup.shape)
# no duplicated data

df_nodup.target.value_counts()

#Outliers?
#Normal - Z-score Method:

#For this outlier detection method, the mean and standard deviation of the residuals are calculated and compared. If a value is a certain number of standard deviations away from the mean, that data point is identified as an outlier. The specified number of standard deviations is called the threshold. The default value is 3.

for i in df_nodup.columns:
    if i!='target':
        df_nodup_mean=df_nodup[i].mean()
        df_nodup_std=df_nodup[i].std()
        df_noout=df_nodup[((df_nodup[i]-df_nodup_mean)/df_nodup_std).abs()<3]
print(df_nodup.shape,df_noout.shape)

df_noout.target.value_counts()


 #Missing values

#Is there missing data in your datasets?

df_noout.isnull().sum()
# the number of missing values are significantly less..

df_clean=df_noout.dropna()
df_clean.isnull().sum()

#prepping for 
df_X=df_clean.drop('target',axis=1)
df_y=df_clean.target
df_y.value_counts()

#scaled our data
df_X_scaled=df_X.copy()
for column in df_X.columns:
    df_X_scaled[column] = (df_X_scaled[column]-df_X_scaled[column].mean())/df_X_scaled[column].std()


#splitting test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

y_train.value_counts()

#import classifier
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()

#train model
LR.fit(X_train,y_train)

#predict on test set
y_pred = LR.predict(X_test)
y_pred_probs = LR.predict_proba(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
precision_score(y_test,y_pred,average='micro')
precision_score(y_test,y_pred,average='macro')
recall_score(y_test,y_pred,average='micro')

from sklearn.model_selection import cross_val_score
clf = LogisticRegression()
cross_val_score(clf,df_X,df_y,cv=5)
cross_val_score(clf,df_X,df_y,cv=5).mean()

df_X.isnull().sum()
df_y

probabilities = LR.predict_proba(X_test)
y_pred_prob = LR.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)

plt.plot(fpr,tpr)
plt.xlabel("FPR")
plt.ylabel('TPR')

roc_auc_score(y_test,y_pred_prob)


############################
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


# ------------- Without scaling and random seed=9 ------------------
start_time = time.time()
max_acc,max_num_neighbors = 0,0;
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=9)

for i in range(1,21):
        KNC = KNeighborsClassifier(n_neighbors=i)
        KNC.fit(X_train,y_train)
        y_pred = KNC.predict(X_test)
        acc_score = accuracy_score(y_pred,y_test)
        if acc_score > max_acc:
            max_acc,max_num_neighbors=acc_score,i
        print(' When number of  neighbors is : {}, Accuracy is {}'.format(i,acc_score))
print('Max Accuracy is {} and occurs when number of neighbors is {}'.format(max_acc,max_num_neighbors))

print("--- %s seconds ---" % (time.time() - start_time))

# ------------- With scaling and random seed = 9 ------------------
start_time = time.time()
max_acc,max_num_neighbors = 0,0;
X_train, X_test, y_train, y_test = train_test_split(df_X_scaled, df_y, random_state=9)
for i in range(1,21):
        KNC = KNeighborsClassifier(n_neighbors=i)
        KNC.fit(X_train,y_train)
        y_pred = KNC.predict(X_test)
        acc_score = accuracy_score(y_pred,y_test)
        if acc_score > max_acc:
            max_acc,max_num_neighbors=acc_score,i
        print(' When number of  neighbors is : {}, Accuracy is {}'.format(i,acc_score))
print('Max Accuracy is {} and occurs when number of neighbors is {}'.format(max_acc,max_num_neighbors))

print("--- %s seconds ---" % (time.time() - start_time))
# ------------- K=10 - Fold without scaling and random seed = 9 ------------------


kfold = KFold(n_splits=10,random_state=9, shuffle=False)
max_fold,max_acc,max_num_neighbors = 0,0,0;
for fold, (train_index, test_index) in zip(range(1,11),kfold.split(df_X)):
   print('For Fold# : {}'.format(fold))
   X_train, X_test = df_X_scaled.iloc[train_index,:], df_X.iloc[test_index,:]
   y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]
   for i in range(1,21):
        KNC = KNeighborsClassifier(n_neighbors=i)
        KNC.fit(X_train,y_train)
        y_pred = KNC.predict(X_test)
        acc_score = accuracy_score(y_pred,y_test)
        if acc_score > max_acc:
            max_fold,max_acc,max_num_neighbors=fold,acc_score,i  
        print('\tWhen number of  neighbors is : {}, Accuracy is {}'.format(i,acc_score))
print('Max Accuracy is {} and occurs when number of neighbors is {} and for Fold# : {}'.format(max_acc,max_num_neighbors,max_fold))
      
      
# ------------- K=10 - Fold with scaling and random seed = 9 ------------------


kfold = KFold(n_splits=10,random_state=9, shuffle=False)
max_fold,max_acc,max_num_neighbors = 0,0,0;
for fold, (train_index, test_index) in zip(range(1,11),kfold.split(df_X_scaled)):
   print('For Fold# : {}'.format(fold))
   X_train, X_test = df_X_scaled.iloc[train_index,:], df_X_scaled.iloc[test_index,:]
   y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]
   for i in range(1,21):
        KNC = KNeighborsClassifier(n_neighbors=i)
        KNC.fit(X_train,y_train)
        y_pred = KNC.predict(X_test)
        acc_score = accuracy_score(y_pred,y_test)
        if acc_score > max_acc:
            max_fold,max_acc,max_num_neighbors=fold,acc_score,i  
        print('\tWhen number of  neighbors is : {}, Accuracy is {}'.format(i,acc_score))
print('Max Accuracy is {} and occurs when number of neighbors is {} and for Fold# : {}'.format(max_acc,max_num_neighbors,max_fold))
      
              
# ------------- KNN using GridSearchCV ------------------
              
from sklearn.model_selection import GridSearchCV
k_range=range(1,21)
parameters= {'n_neighbors':k_range}
KNC = KNeighborsClassifier()

grid=GridSearchCV(KNC,parameters, scoring='accuracy',cv=20)
grid.fit(df_X,df_y)
grid.best_params_

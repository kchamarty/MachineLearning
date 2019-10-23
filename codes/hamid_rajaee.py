# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:55:35 2019

@author: 16479
"""

# Import our libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#changing directory
import os
os.chdir(r'C:\Users\hrajaee\Desktop\Data mining\EXAM')
os.getcwd()
'''1. Mean of the integrated profile.

	2. Standard deviation of the integrated profile.
	3. Excess kurtosis of the integrated profile.

	4. Skewness of the integrated profile.
	5. Mean of the DM-SNR curve.
	
6. Standard deviation of the DM-SNR curve.
	7. Excess kurtosis of the DM-SNR curve.

	8. Skewness of the DM-SNR curve.
'''
columns=['Mean',
         'Standard_deviation',
         'Excess_kurtosis',
         'Skewness',
        'Mean_DM_SNR',
        'Standard_deviation_DM_SNR',

         'Excess_kurtosis_DM-SNR',
         'Skewness_DM-SNR','target']


df=pd.read_csv(r'HTRU_2.csv',header=None,names=columns)

df.shape
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

df.describe().transpose()




df['target'].value_counts()




"""Data cleansing"""
#Do we have duplicated data?
#Duplicate Data
df_dub=df.drop_duplicates()
print(df.shape,df_dub.shape)
# no duplicated data


#Outliers
#Is there outlier based on standard deviation method?
#Mean and Standard Deviation Method:
#For this outlier detection method, the mean and standard deviation of the residuals are calculated and compared. If a value is a certain number of standard deviations away from the mean, that data point is identified as an outlier. The specified number of standard deviations is called the threshold. The default value is 3.


for i in df.columns:
   df2=df[((df[i]-df[i].mean())/df[i].std()).abs()<3]
print(df.shape,df2.shape)

df2.describe()

df1=df
 #Missing values

#Is there missing data in your datasets?

df1.isnull().sum()



df1.isnull().mean()
df1=df1.fillna(df1.mean())



"""scaling"""
X=df1.drop('target',axis=1)

y=df1.target

#scaled our data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

#splitting test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)


#import classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#train model
model.fit(X_train,y_train)

#predict on test set


y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))


precision_score(y_test,y_pred,average='micro')
precision_score(y_test,y_pred,average='macro')

recall_score(y_test,y_pred,average='micro')

from sklearn.model_selection import cross_val_score
clf = LogisticRegression()
cross_val_score(clf,X,y,cv=5)
cross_val_score(clf,X,y,cv=5).mean()

probabilities = model.predict_proba(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.xlabel("FPR")
plt.ylabel('TPR')

roc_auc_score(y_test,y_pred_prob)





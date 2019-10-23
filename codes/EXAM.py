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
os.chdir(r'C:\Users\16479\Desktop\data mining metro college\final project')
os.getcwd()



df=pd.read_csv(r'car_evaluation (1).csv')
df.shape
df.head()
#checking columns and total records and null Values
df.info()
df_columns=df.columns
print(df_columns)

# Renaming some columns
df=df.rename(columns={'vhigh':'Price','vhigh.1':'Maintenance Cost','2':'Number of Doors',
                     '2.1':'Capacity' ,'small':'Size of Luggage Boot',
                     'low':'safety','unacc':'Class'})
df.head()
df.tail()

# Copy of the dataframe
original_df = df.copy()


df.info()

df.dtypes

df.describe().transpose()



"""Encoding"""

#fixing categorical variables which are ordinal variables
df['Price'].value_counts()
df['Price']=df['Price'].map({'low':1,'med':2,'high':3,'vhigh':4})


df['Maintenance Cost'].value_counts()
df['Maintenance Cost']=df['Maintenance Cost'].map({'low':1,'med':2,'high':3,'vhigh':4})


df['Number of Doors'].value_counts()
df['Number of Doors']=df['Number of Doors'].map({'2':1,'3':2,'4':3,'5more':4})


df['Capacity'].value_counts()
df['Capacity']=df['Capacity'].map({'2':1,'4':2,'more':3})


df['Size of Luggage Boot'].value_counts()
df['Size of Luggage Boot']=df['Size of Luggage Boot'].map({'small':1,'med':2,'big':3})


df['safety'].value_counts()
df['safety']=df['safety'].map({'low':1,'med':2,'high':3})


df['Class'].value_counts()
df['Class']=df['Class'].map({'unacc':1,'acc':2,'good':3,'vgood':4})

df.info()
df2 = df.copy()

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
   df1=df[((df[i]-df[i].mean())/df[i].std()).abs()<3]
print(df.shape,df1.shape)

df1.describe()
#there are 65 outlier based on standard deviation method that all of them is in class 4(vgood),so it droped class 4(I checeked accuracy_score for both and saw this elimination increase that 6%)

 #Missing values

#Is there missing data in your datasets?

df1.isnull().sum()

#there no missing values

# Copy of the dataframe
original_df1 = df1.copy()




"""scaling"""
X=df1.drop('Class',axis=1)

y=df1.Class

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
cross_val_score(clf,X,y,cv=4).mean()


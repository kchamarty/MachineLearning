# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:02:07 2019

@author: KC
"""

import pandas as pd
import matplotlib.pyplot as plt
import time

fpath=r'C:\Users\KC\Documents\Metro College\ML\data'
fname=r'Mall_Customers.csv'
file= '{}\{}'.format(fpath,fname)

df = pd.read_csv(file)

st_time =time.time()
from sklearn.cluster import KMeans
kmn =KMeans(n_clusters = 3 )
kmn.fit(df)

classes=kmn.predict(df)

print('Execution time is : {}'.format(time.time()-st_time))

kmn.cluster_centers_
#visualization
plt.scatter(df.iloc[classes==0,0],df.iloc[classes==0,1],color='red',marker ='+')
plt.scatter(df.iloc[classes==1,0],df.iloc[classes==1,1],color='blue',marker ='+')
plt.scatter(df.iloc[classes==2,0],df.iloc[classes==2,1],color='green',marker ='+')

plt.scatter(kmn.cluster_centers_[:,0],kmn.cluster_centers_[:,1],color='orange',marker='*')


wcss=[]
kx=range(1,11)
kmn.inertia_
for k in kx:
    kmn =KMeans(n_clusters = k )
    kmn.fit(df)
    wcss.append(kmn.inertia_)


plt.plot(kx,wcss)


# We have determined that k=5
from sklearn.cluster import KMeans
kmn =KMeans(n_clusters = 5 )
kmn.fit(df)
colors=['red','blue','green','pink','yellow']
classes=kmn.predict(df)
#visualization
for i in range(5):
    plt.scatter(df.iloc[classes==i,0], df.iloc[classes==i,1], color=colors[i], marker ='+')


plt.scatter(kmn.cluster_centers_[:,0],kmn.cluster_centers_[:,1],color='orange',marker='*')
plt.title('Cluster of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')

from sklearn.model_selection import GridSearchCV
k_range=range(1,21)
param_grid = {'n_clusters':k_range}
kmn =KMeans()

grid=GridSearchCV(kmn,param_grid,cv=4)
grid.fit(df)
grid.param_grid
grid.best_params_
grid.best_estimator_

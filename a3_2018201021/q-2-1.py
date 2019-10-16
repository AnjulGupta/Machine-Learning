#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import operator
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.metrics import classification_report,mean_absolute_error,mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


# ## Sigmoid func

# In[2]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ## Loss function

# In[3]:


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# ## Gradient Descent

# In[4]:


def grad_decent(train,test,train_y,test_y) :
 
    X=train.values
    z = np.ones((len(train),1))
    X=np.append(z,X, axis=1)
    y=train_y
    beta = np.zeros(X.shape[1]) 


    X1=test.values
    z1 = np.ones((len(test),1))
    X1=np.append(z1,X1, axis=1)
    y1=test_y
    beta1 = np.zeros(X1.shape[1]) 


    # print(type(X))
    lr=0.01

    for i in range(30000):
        z = np.dot(X, beta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        beta -= lr * gradient


    z1 = np.dot(X1, beta1)
    h = sigmoid(z1)
    ans=sigmoid(np.dot(X1, beta1))
    ans=np.where(ans>=0.5,1,0)


    corr=0
    for i in range(len(X1)):
        if (test_y[i]==ans[i]):
            corr+=1
    print(corr/len(X1))


# ## Reading data from files

# In[5]:


df = pd.read_csv('AdmissionDataset/data.csv') 
df=df.drop(['Serial No.'], axis = 1 )

train,test = train_test_split(df, test_size=0.2)

#converting train and test label into binary groups 0 or 1 
train_y = train['Chance of Admit ']
test_y = test['Chance of Admit ']
train_y=np.where(train_y>=0.5,1,0)
test_y=np.where(test_y>=0.5,1,0)


# df=df.drop(['Serial No.','Chance of Admit '], axis = 1 )
train = train.drop('Chance of Admit ',axis=1)
test = test.drop('Chance of Admit ',axis=1)


grad_decent(train,test,train_y,test_y)


# ## using SKLearn 

# In[6]:


reg = linear_model.LogisticRegression(solver='lbfgs') 
reg.fit(train,train_y)
pred2=reg.predict(test)
accuracy_score(test_y,pred2)


# In[ ]:





# In[ ]:





# In[ ]:





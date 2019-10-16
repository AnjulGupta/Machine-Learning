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


import matplotlib.pyplot as plt 


# 
# ## Sigmoid func

# In[2]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ## Loss function

# In[3]:


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# In[4]:


def printing(tp,fp,tn,fn):
    a = (tp+fp)
    b = (tp+fn)
    if a:
        rc=tp/a
    if b:
        pc=tp/b
        
    accu=((tp+tn)/(tp+tn+fp+fn))
    
    recall_list.append(rc)
    precision_list.append(pc)
    print("Recall: ", rc)
    print("Precision: ", pc)
    print("Accuracy",accu)
    accuracy_list.append(accu)


# In[5]:


def predict(X1,test_y1,ans):
    tp,fp,fn,tn=0,0,0,0
    corr= 0
    for i in range(len(X1)):
        if test_y1[i]==1 and ans[i]==1:
            tp+=1
        if test_y1[i]==1 and ans[i]==0:
            fp+=1
        if test_y1[i]==0 and ans[i]==0:
            tn+=1
        if test_y1[i]==0 and ans[i]==1:
            fn+=1

    printing(tp,fp,tn,fn)
    
    


# ## Gradient Descent

# In[6]:


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
    
    global accuracy_list,recall_list,precision_list
    accuracy_list=[]
    recall_list=[]
    precision_list=[]
    for x in range(2,9):
        accu=0
        x=x*0.1
        train_y1=np.where(train_y>=x,1,0)
        y=train_y1
        test_y1=np.where(test_y>=x,1,0)
        beta = np.zeros(X.shape[1]) 
        lr=0.01
        for i in range(50000):
            z = np.dot(X, beta)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            beta -= lr * gradient
        z = np.dot(X1, beta)
        h = sigmoid(z)
        ans=sigmoid(np.dot(X1, beta))
        ans=np.where(ans>=x,1,0)
        
        predict(X1,test_y1,ans)


# ## Reading data from files

# In[7]:


df = pd.read_csv('AdmissionDataset/data.csv') 
df=df.drop(['Serial No.'], axis = 1 )

train,test = train_test_split(df, test_size=0.2)

#converting train and test label into binary groups 0 or 1 
train_y = train['Chance of Admit ']
test_y = test['Chance of Admit ']


# df=df.drop(['Serial No.','Chance of Admit '], axis = 1 )
train = train.drop('Chance of Admit ',axis=1)
test = test.drop('Chance of Admit ',axis=1)


grad_decent(train,test,train_y,test_y)


# 

# In[8]:


plt.plot([0.2,0.3,0.4,0.5,0.6,0.7,0.8], recall_list, label = "Recall")
plt.plot([0.2,0.3,0.4,0.5,0.6,0.7,0.8], precision_list, label = "Precision")
plt.xlabel('Threshold Value') 
plt.ylabel('Precision - Recall Value') 
plt.legend() 
plt.grid()
plt.title('Threshold V/s Precision-Recall') 
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





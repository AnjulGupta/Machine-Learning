#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[34]:


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


# # USING LOGISTIC REGRESSION

# ## Sigmoid func

# In[35]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ## Loss function

# In[36]:


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# ## Gradient Descent

# In[37]:


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

# In[38]:


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

# In[39]:


reg = linear_model.LogisticRegression(solver='lbfgs') 
reg.fit(train,train_y)
pred2=reg.predict(test)
accuracy_score(test_y,pred2)


# In[ ]:





# # USING KNN CLASSIFIER

# In[40]:


def Knn_classifier(train,row, k):
    #to store {euclid sum , label} 
    my_dist = []
    #l = no of cols in a row
    l = len(row)
    
    #for finding label corresponding to a row 
    count = 0
    
    #calculating Euclidian distance for each row in validate by every row in train data
    for r in train[:,:-1]:
        sum = 0
        #for each col in a row calculate Euclid dstance from rth row of train 
        for x in range(l):
            sum+=pow(row[x] - r[x],2)
            label = train [count][7]
#             print(label)
        count+=1
        sum = math.sqrt(sum)
        #appending distance of validate row with each train row in list
        my_dist.append((sum,label))
    
    #sort distances in increasing order
    my_dist.sort(key=lambda x: x[0])

    #stores {label -> count) for k nearest rows  
    predlabel = {}
    
    for x in range(k):
        res = my_dist[x][1]
        #if predlabel for that label empty, val=1
        if res not in predlabel:
            predlabel[res] = 1
        #otherwise val+=1
        predlabel[res] += 1
    #sort predlabel in decreasing order
    ans = sorted(predlabel.items(),reverse = True)
    #return 1st value (maximum level count in K nearest neighbours)
    k = ans[0][0]
    return k


# In[41]:


def pfunction():
    global tp,tn,fp,fn
    print("tp :",tp,", tn :",tn,", fp :",fp,", fn :",fn)

    if(tp!=0 and fn!=0):
        rc=tp/(tp+fn)
    else:
        rc = 1
    if(tp!=0 and fp!=0):
        pc=tp/(tp+fp)
    else:
        pc = 1
    f1=2/((1/pc)+(1/rc))
    print("recall :",rc,", precision :",pc,", F1 score :",f1)
    
    return


# In[42]:


def find_correct(myans , myvalidate):
    global correct,tp,tn,fp,fn
    a = myvalidate[:][7]

    if (myans == a):
        correct +=1
    if(myans==1 and a==1):
        tp+=1
    if(myans==1 and a==0):
        fp+=1    
    if(myans==0 and a==0):
        tn+=1
    if(myans==0 and a==1):
        fn+=1
        
    return correct


# In[43]:


df = pd.read_csv('AdmissionDataset/data.csv')
df=df.drop(['Serial No.'], axis = 1 )

train,validate = train_test_split(df, test_size = 0.2)
train['Chance of Admit ']=np.where(train['Chance of Admit ']>=0.5,1,0)
validate['Chance of Admit ']=np.where(validate['Chance of Admit ']>=0.5,1,0)
random.seed(0)

train1=train
val1=validate

# taking all attr except last(no use) in train and myvalidate
train = train.values
myvalidate = validate.values

# taking all attr except 1st(label) and last(no use) in validate_arr
validate_arr = validate.iloc[:, 1:].astype(float)
validate_arr = validate_arr.values

acc_list = []
correct = 0
tp,fp,tn,fn=0,0,0,0
max_accuracy = 0
k_for_max_acc = 0
for i in range(1,22) :
    for row in range(len(validate_arr)):
        #sending validate_arr row by row in KNN classifier
        myans = Knn_classifier(train,validate_arr[row], i)
        #matching predicted answer from correct for this row
        corr = find_correct(myans , myvalidate[row])
    #calculating accuracy and appending in list
    accu = (corr/float(len(validate_arr)))*100
    acc_list.append(accu)
    if(accu>max_accuracy):
        max_accuracy = accu
        k_for_max_acc = i
    correct = 0
#     pfunction()
#     print("no of test samples", len(validate_arr))
#     print("correct",corr)
#     print("accuracy",accu)
print("\nno of test samples", len(validate_arr))
print("Max_accuracy : ",max_accuracy, "for k :",k_for_max_acc)


# In[ ]:





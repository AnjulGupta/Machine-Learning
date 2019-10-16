#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import math
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from pprint import pprint


# In[2]:

# //////////////////
# filename=sys.argv[1]
# df2=pd.read_csv(filename)
# test=df2.values
# //////////////////

df = pd.read_csv("./input_data/data.csv")
# print(df)

train,test = train_test_split(df,test_size =0.2)


trainx = train[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
# print(x)
trainy = train['Chance of Admit ']
# label

testx = test[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
# print(x)
testy = test['Chance of Admit ']
# label


x = trainx.values
y = trainy.values
# print("x")
temp=np.ones((len(train),1))
x=np.append(temp,x,axis=1)
# print(x)
# y = y.values
x_t =  np.transpose(x)
# print("x_t")
# print(x_t)

x_tx = np.dot(x_t,x)
x_ty = np.dot(x_t,y)

x_tx_inv = inv(x_tx)
# x_tx_inv


b = np.dot(x_tx_inv,x_ty)
# pprint(b)


ybar =[]
error = []

testx = testx.values
testy = testy.values
temp=np.ones((len(test),1))
testx=np.append(temp,testx,axis=1)


for row in testx :
    pro = 0
    for i in range(len(row)):
        pro += b[i]*row[i]
    ybar.append(pro)

pprint(len(ybar))
pprint((ybar))

for row in range(len(testy)) :
    err = 0
    err = testy[row] - ybar[row]
    error.append(err)
 

# pprint(error)
#Mean Absolute Error
MAE=(abs(ybar-testy).sum())/len(testy)
print("MAE :",MAE)

#Mean Square Error
MSE=((ybar-testy)**2).sum()/len(testy)
print("MSE :",MSE)

#Mean Percentage Error
MPE=((((ybar-testy)/testy).sum())/len(testy))*100
print("MPE :",MPE)


# In[ ]:





# In[ ]:





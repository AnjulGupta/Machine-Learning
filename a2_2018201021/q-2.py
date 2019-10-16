#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[64]:


# from future import division
import numpy as np
import pandas as pd
import random
import math
from math import sqrt,pi,e
from pprint import pprint
from sklearn.model_selection import train_test_split
from statistics import mean 
from statistics import stdev 
import sys


# ## Calculate Prob of Label(1 and 0)

# In[65]:


def count_label(data):
    l1 = len(data[(data['label']==1)])
    l0 = len(data[(data['label']==0)])
    global pl0,pl1
    pl1 = l1/len(data)
    pl0 = l0/len(data)
   


# ## Divide Attributes (Categorical or Continuous)

# In[66]:


def divide_attr(data):
    for attr in data:
        if attr == 'label':
            continue
        arr = np.array(data[attr])
        unival,unicount = np.unique(arr,return_counts = True)
        if(len(unival)>10):
            cont.append(attr)
        else:
            cat.append(attr)


# ## Calculating 'F' For Continuous Data

# In[67]:


def fvalue(cont, train):
    global mnstd
    for i in cont:
        mnstd[i] = {}
        arr = np.array(train[i])
        mn = mean(arr)
        std = stdev(arr)
        mnstd[i]['mean'] = mn
        mnstd[i]['stdev'] = std


# ## Bayes Naive Classifier function 

# In[68]:


def naive_classifier(data,mydict):
    temp = {}
    
    for attr in data:
        temp[attr] = {}
 
        arr = np.array(data[attr])
        unival = (np.unique(arr))
        
        for val in unival:
            
            temp[attr][val] = {}
            
            valcount = len(data[data[attr]==val])
            pval = valcount/len(data)
            temp[attr][val]['pval'] = pval
            
            vl1 = len(data[(data[attr]==val) & (data['label']==1)])
            
            vl0 = valcount - vl1
            pvl1 = float(vl1)/float(valcount)
            pvl0 = float(vl0)/float(valcount)
            
            temp[attr][val]['ptrue'] = pvl1
            temp[attr][val]['pfalse'] = pvl0           
            
        mydict.update(temp)
        temp = {}


# ## Predicting the Test data

# In[69]:


def prediction(row, dic ):
    pptrue = 1
    ppfalse = 1
    ppval = 0
    
    global pl0,pl1
    
    for i in range(len(row)-1):
        if(row.keys()[i] in cat):
            try:
                pptrue *= dic[row.keys()[i]][row.values[i]]['ptrue']
                ppfalse *= dic[row.keys()[i]][row.values[i]]['pfalse']
                ppval += dic[row.keys()[i]][row.values[i]]['pval']
            except:
                continue
        else:
            attr = row.keys()[i]
            x =  row.values[i]
            
            for key in mnstd :
                if(key == attr):
                    m = mnstd[key]['mean']
                    s = mnstd[key]['stdev']
                    temp = (1/(sqrt(2*pi)*s))
                    
                    ff = temp*(e**(-(x - m)**2))/(2*(s**2))
                    if(ff != 0):
                        pptrue *= ff
                        ppfalse *= ff
                    try:
                        ppval += dic[row.keys()[i]][row.values[i]]['pval']
                    except:
                        continue
    pptrue *= pl1
    ppfalse *= pl0
    ppval = ppval
    if (ppval!=0): 
        pyes = pptrue/ppval
        pno = ppfalse/ppval
    else:
        pyes = pptrue
        pno = ppfalse
        
    global count
    
    if(pyes > pno and row.values[len(row)-1] == 1 ):
        count +=1
    if(pyes < pno and row.values[len(row)-1] == 0 ):
        count +=1


# ## Reading data from file and Calculating Accuracy

# In[72]:

# //////////////////
# filename=sys.argv[1]
# df2=pd.read_csv(filename,,names = ['id','age','exp','Inc','zip','family','expend','edu','mortgage','label','sa','cd','ib','cc'])
# test=df2.values
# //////////////////

df =  pd.read_csv("data.csv",names = ['id','age','exp','Inc','zip','family','expend','edu','mortgage','label','sa','cd','ib','cc'])
# remove first row errornous
df = df.iloc[1:]
# print(len(df))

x = len(df.loc[(df['exp'] > df['age'])])
# x = len(df[df.exp < 0].index)
# print(x)

# removed rows with negative exp
df = df.drop(df[df.exp < 0].index)
# print(len(df))

# remove useless columns
df.drop(df.columns[[0,4,]], axis=1, inplace=True)

# rearrange - move label to last
neworder = [0,1,2,3,4,5,6,8,9,10,11,7]
df = df[df.columns[neworder]]
# print((df))


train,validate = train_test_split(df, test_size = 0.2)
#remove last column from datas as insignificant
# print(len(train))

pl0 = 0
pl1 = 0
count_label(train) 

cat = []
cont = []
mnstd = {}

divide_attr(train)    
fvalue(cont,train)

mydict = {}
count = 0

naive_classifier(train,mydict)   
pprint(mydict)

for x in range(len(validate)):
    prediction(validate.iloc[x], mydict)

accuracy = count/len(validate)
print(accuracy)


# ## Accuracy on SKLEARN

# In[71]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

trax = train.iloc[:,:-1]
tray = train.iloc[:,-1]

trax = np.array(trax)
tray = np.array(tray)

clf = GaussianNB()
clf.fit(trax,tray )


testx = validate.iloc[:,:-1]
testy = validate.iloc[:,-1]

testx = np.array(testx)
testy = np.array(testy)

predicts = clf.predict(testx)

# y = np.array(training_labels)

print ("Accuracy Rate is: %f" % accuracy_score(testy, predicts))


# In[ ]:





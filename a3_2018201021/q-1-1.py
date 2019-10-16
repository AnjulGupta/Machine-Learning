#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import operator
from numpy import linalg as la


# ## Calculate Eigen values and vectors

# In[14]:


def calculate_eigens(df):
    #finding atttr_wise(col_wise) mean and std
    mn = np.mean(df)
    std = np.std(df)
    #normalizing data to (0-1) range
    normalised = (df - mn)/std
    #finding covariance matrix to make mat of order(attr * attr)
    cov_mat = np.cov(normalised.T)
    #calculating eval and evec
    evalue,evec = la.eig(cov_mat)
    return evalue,evec,normalised


# ## Selecting  dimensions with 10% tolerance using PCA

# In[15]:


def select_dimensions(mydict,evalue,evec,ndata):
    #storing eval,evec in dict as key-value
    for i in range (len(evalue)) :
        mydict[evalue[i]] = evec[i]
    
    #summing evalues 
    evalue_sum = sum(evalue)
    #print(evalue_sum)
    
    #sorting evalues in decreasing order 
    evalue_sorted = sorted(evalue, reverse = True)
    
    min_val = .90
    curr_val = 0
    
    #sel_dimensions list till tolerance becomes less than 0.1
    dim_list = []
    
    
    for i in range (len(evalue_sorted)) :
        #while curr_tolerance is less than .9, include dimension in dim_list
        curr_val = curr_val + evalue_sorted[i]/evalue_sum
        dim_list.append(mydict[evalue_sorted[i]])
        if curr_val > 0.9 :
            break
        
#     print (curr_val)
#     print (dim_list)
    
    #final reduced dimensions 
    dim_list = np.asarray(dim_list)
    dim_list = dim_list.T
    dimensions = np.dot(ndata,dim_list)
    
#     print(ndata.shape, dim_list.shape, dimensions.shape)
    
    #final reduced data is of dimensions (rows * reduced_attr)
    print ('Accuracy is :',curr_val)
    print('Reduced Dimensions :',dimensions.shape[1])
    


# ## Reading data from files

# In[16]:


df = pd.read_csv('intrusion_detection/data.csv') 
label = df['xAttack']
df = df.drop(['xAttack'],axis =1)


evalue,evec,ndata = calculate_eigens(df)

#store e_val,evec in dict as key-value
mydict = {}

select_dimensions(mydict,evalue,evec,ndata)


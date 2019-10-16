#!/usr/bin/env python
# coding: utf-8

# In[442]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pprint import pprint


# In[443]:


def Entropy(prob):
    if prob==0:
        return 0
    elif prob==1:
        return 0
    else:
        return -(prob*math.log2(prob) + (1-prob)*math.log2(1-prob))


# In[444]:


def CreateTree(root,data,data_check):
    count1=len(data[data.left==1])
    
#cheking purity of data -leaf node (all left 1 or 0)
#if all 0     
    if(count1==0):
        root.update({'value': 0})
        return
#if all 1     
    if(count1==len(data)):
        root.update({'value': 1})
        return
        
#finding probability and entropy          
    prob=count1/len(data)
    entropy = Entropy(prob)
    
# select the best attribute     
    newNode=Best_Attribute_Select(entropy,data)

# if no attribute left -leaf node  
    if(newNode is None):
        #count1=len(data[data.left==1])
        count0=len(data)-count1
        if(count1 >= count0):
            root.update({'value':1})
        else:
            root.update({'value':0})
        return

# update new attribute in tree and move pointer to new attribute    
    root.update({newNode:{}})
    root=root[newNode]

    global n_count
    n_count+=1
    
#     if n_count == 5:
    acc=calcAccuracy(data_check.to_dict('records'))
    acc_list.append(acc)

    node_list.append(n_count)


# finding unique values in newNode
    uniqvals = np.unique(data[newNode])
    for val in uniqvals:
        #update selected val of attribute in tree         
        root.update({val:{}})
        # take all those rows from data (of file) having selected val of attribute         
        p = data[newNode]
        trow=data[p==val]
        #recursively call function on new data and new attribute val
        newdata = trow.drop(columns=[newNode])
        CreateTree(root[val],newdata,data_check)


# In[445]:


def Best_Attribute_Select(entropy,data):
    info_gain = { }
    for attr in data:
        if(attr!="left"):
            #for each attr in data , find unique val with count 
            a=np.array(data[attr])
            unique_val, val_count=np.unique(a,return_counts=True)

            avg_entropy=0

    #         mydict[unique_val]='val_count'
    #         mydict = {'unique_val':'val_count'}
            x = zip(unique_val,val_count)
            mydict = dict(x)

            for attr_val in mydict:
                # select rows with attributr val = attrval and left = 1              
                left1 = data[(data[attr]==attr_val) & (data['left']==1)]
                #calulate prob
                l1 = len(left1)
                lt= len(data)
                prob = l1/mydict[attr_val]
                ent = Entropy(prob)
                avg_entropy+=((mydict[attr_val]/lt)* ent)
            #calculate info gain    
            info_gain[attr]= entropy - avg_entropy
        try:
            #select the attribute with max info gain
            tmp=-1
            attr_selected=None
            for k in info_gain.keys():
                if(info_gain[k]>tmp):
                    tmp=info_gain[k]
                    attr_selected=k
        except:
            return None
        
        else:
            continue
    
    return attr_selected


# In[446]:


def calcAccuracy(data):
    accuracy=0
    
    for i in data:
        x = predict(Tree,Tree,i)
        
        if(predict(Tree,Tree,i)==i['left']):
            accuracy+=1
    accuracy = (accuracy * 100)/len(data)
    return accuracy


# In[447]:


def predict(model,root,row):
#     print(row)
    key=list(root.keys())[0]
#     print(key)
    if(key=='value'):
        return root[key]
    try:
        root=root[key]
        if row[key] in root.keys():
            return predict(model,root[row[key]],row)
        else:
            for k in root.keys():
                if(row[key]<=k):
                    return predict(model,root[k],row)
            return 0
    except:
        return 0


# In[448]:


def con_to_cat(data):
    for attr in data:
        arr=np.array(data[attr])
#calculate unique values with their count         
        unique_val, val_count=np.unique(arr,return_counts=True)
#     
        x = zip(unique_val,val_count)
        mydict = dict(x)

        l = len(mydict)
        
        if(l>12):
            x=l//12 + 1 #x =number of classes
            
            k = mydict.keys()
            keys = list(k)
            newCol=[]
            
            r = data[attr]
            for i in r:
                ind=keys.index(i)
                upperlimit=(ind//x + 1)*x
                if(upperlimit>=l):
                    upperlimit=-1
                ul = keys[upperlimit]    
                newCol.append(ul)
            
            data[attr]=newCol
    
    return data


# In[449]:


#reading data from file
df = pd.read_csv("train.csv")
train1,test = train_test_split(df, test_size= 0.2)
train=con_to_cat(train1)

#error for training error
n_count=0
error=[]
acc_list=[]
node_list=[]
Tree={}
CreateTree(Tree,train,train)

for acc in acc_list:
    error.append(1-acc)
plt.plot(node_list,error,label="Training Error")


#error for validation error
n_count=0
error=[]
acc_list=[]
node_list=[]
Tree={}
CreateTree(Tree,train,test)
for acc in acc_list:
    error.append(1-acc)

#plotting graphs    
plt.plot(node_list,error,label="Validation Error")
plt.title('Error vs Nodes')
plt.xlabel('Nodes count')
plt.ylabel('Errors count')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:





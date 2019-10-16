#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from pprint import pprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sea


# In[2]:


# Gini index
def GiniIndex(prob):
    if prob==0:
        return 0
    elif prob==1:
        return 0
    else:
        return (2*(prob)*(1-prob))

# # misclassification
# def Entropy(prob):
#     if prob==0:
#         return 0
#     elif prob==1:
#         return 0
#     else:
#         return min((prob),(1-prob))

# entropy
# def Entropy(prob):
#     if prob==0:
#         return 0
#     elif prob==1:
#         return 0
#     else:
#         return -(prob*math.log2(prob) + (1-prob)*math.log2(1-prob))


# In[3]:


def con_to_cat(data):
    for attr in data:
        arr=np.array(data[attr])
#calculate unique values with their count         
        unique_val, val_count=np.unique(arr,return_counts=True)
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
    


# In[4]:


def CreateTree(root,data):
    count0=len(data[data.left==0])
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
    l = len(data)
    prob=count1/l
    entropy = GiniIndex(prob)
    
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

# update new attribute in tree     
    root.update({newNode:{}})
# move pointer to new attribute
    root=root[newNode]

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
        
        CreateTree(root[val],newdata)


# In[5]:


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
                gi = GiniIndex(prob)
                avg_entropy+=((mydict[attr_val]/lt)* gi)
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



# In[6]:


def predictAttr(data):
    accuracy,tp,fp,tn,fn=0,0,0,0,0

    for i in data:
        if(predict(Tree,Tree,i)==i['left']):
            accuracy+=1
        if(predict(Tree,Tree,i)==1 and i['left']==1):
            tp+=1
        if(predict(Tree,Tree,i)==1 and i['left']==0):
            fp+=1    
        if(predict(Tree,Tree,i)==0 and i['left']==0):
            tn+=1
        if(predict(Tree,Tree,i)==0 and i['left']==1):
            fn+=1    

    pfunction(accuracy,tp,fp,tn,fn,data)


# In[7]:


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


# In[8]:


def pfunction(accuracy,tp,fp,tn,fn, data):
    accuracy = (accuracy * 100)/len(data)
    print("Accuracy is :",accuracy)
    print("tp :",tp)
    print("tn :",tn)
    print("fp :",fp)
    print("fn :",fn)
    if(tp!=0 and fn!=0):
        rc=tp/(tp+fn)
    else:
        rc = 1
    if(tp!=0 and fp!=0):
        pc=tp/(tp+fp)
    else:
        pc = 1
    f1=2/((1/pc)+(1/rc))
    print("recall :",rc)
    print("precision :",pc)
    print("F1 score :",f1)
    
    return
    


# In[9]:


df = pd.read_csv("train.csv")
train1,test = train_test_split(df, test_size= 0.2)
train=con_to_cat(train1)
Tree={}
CreateTree(Tree,train)

#convert test df to dictionary
x = test.to_dict('records') 
predictAttr(x)


# In[10]:


pprint(Tree)


# In[11]:


#Comparing result with in-built(scikit-learn) decision tree function to check correctness of algorithm used

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv("train.csv")
# df=df.drop(['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company'], axis = 1 )
from sklearn import tree
model = tree.DecisionTreeClassifier()
sal = LabelEncoder()
sale=LabelEncoder()
df['salary_n'] = sale.fit_transform(df['salary'])
df['sales_n'] = sal.fit_transform(df['sales'])

df=df.drop(['sales','salary'],axis='columns')

#dividing the data into training and testing data(for validation)
msk = np.random.rand(len(df)) < 0.8
test2 = df[~msk]
train2 = df[msk]
trainx=train2.drop(['left'],axis='columns')
trainy=train2['left']

#training the model
model.fit(trainx,trainy)
testy=test2['left']
testx=test2.drop(['left'],axis='columns')

#predicting over the test data
pred=model.predict(testx)
print(accuracy_score(testy,pred))
print (confusion_matrix(testy,pred))
print (classification_report(testy,pred))


# In[ ]:





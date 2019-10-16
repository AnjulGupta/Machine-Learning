#!/usr/bin/env python
# coding: utf-8

# In[249]:


import numpy as np
import pandas as pd
import math
from pprint import pprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sea


# In[250]:


def makeTree(root,data):
    
#counting 1's in result
    ll = data.left
    count1=len(data[ll==1])
    
#cheking purity of data -leaf node (all left 1 or 0)
    #if all 0     
    if(count1==0):
        root.update({'value': 0})
        return

    #if all 1     
    if(count1==len(data)):
        root.update({'value': 1})
        
#finding probability and entropy          
    prob=count1/len(data)
    entropy = Entropy(prob)
    
# select the best attribute     
    newNode=Best_Attribute_Select(entropy,data)

# if no attribute left -leaf node  
    if(newNode is None):
        #calculate 0's
        count1=len(data[ll==1])
        count0=len(data)-count1
        
        if(count1 < count0):
            root.update({'value':0})
        else:
            root.update({'value':1})
        return

# update new attribute in tree and move pointer to new attribute    
    root.update({newNode:{}})
    root=root[newNode]

# finding unique values in newNode
    uniqvals = np.unique(data[newNode])
    for val in uniqvals:
        #update selected val of attribute in tree         
        root.update({val:{}})
        # take all those rows from data (of file) having selected val of attribute         
        temp=data[data[newNode]==val]
        #recursively call function on new data and new attribute val
        newdata = temp.drop(columns=[newNode])
        makeTree(root[val],newdata)


# In[251]:


def Entropy(prob):
    if prob==0:
        return 0
    elif prob==1:
        return 0
    else:
        return -(prob*math.log2(prob) + (1-prob)*math.log2(1-prob))


# In[252]:


def Best_Attribute_Select(entropy,data):
    
    for attr in data:
        if(attr!="left"):
            #for each attr in data , find unique val with count 
            a=np.array(data[attr])
            unique_val, val_count=np.unique(a,return_counts=True)

            # print(zip(unique_val,val_count))
            x = zip(unique_val,val_count)
            #create dict of val and count for attr         
            mydict = dict(x)

    #         print(uValCount)
    #         uValCount=makeDict(data[attr])

            info_gain={}
            avg_entropy=0

            for attr_val in mydict:
                # select rows with attributr val = attrval and left = 1              
                left1 = data[(data[attr]==attr_val) & (data['left']==1)]
                #calulate prob
                prob = len(left1)/mydict[attr_val]
                avg_entropy+=((mydict[attr_val]/len(data))*Entropy(prob))
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


# In[253]:


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
    


# In[254]:


def predictAttr(data):
    accuracy= 0
    tp,fp,tn,fn=0,0,0,0

    root=Tree

    for i in data:
        if(checkValue(i,root)==i['left']):
            accuracy+=1
        if(checkValue(i,root)==1 and i['left']==1):
            tp+=1
        if(checkValue(i,root)==1 and i['left']==0):
            fp+=1    
        if(checkValue(i,root)==0 and i['left']==0):
            tn+=1
        if(checkValue(i,root)==0 and i['left']==1):
            fn+=1    

    pfunction(accuracy,tp,fp,tn,fn,data)
    


# In[255]:


def checkValue(test_data_row, root):
    x = root.keys()
    node=list(x)[0]

    if(node!='value'):
        try:
            root=root[node]
            r = row[node]
            return checkValue(test_data_row, root[r])
        except:
            return 0
    else:
        k = root[key]
        return k
        


# In[256]:


df = pd.read_csv("train.csv")
df=df.drop(['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company'], axis = 1 )
train,test = train_test_split(df, test_size= 0.2)
Tree={}

makeTree(Tree,train)

# pprint(Tree)
predictAttr(test.to_dict('records'))


# In[257]:


#Comparing result with in-built(scikit-learn) decision tree function to check correctness of algorithm used

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv("train.csv")
df=df.drop(['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company'], axis = 1 )
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





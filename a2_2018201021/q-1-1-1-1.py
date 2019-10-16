#!/usr/bin/env python
# coding: utf-8

# In[138]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from pprint import pprint
import random
import matplotlib.pyplot as plt  


# In[139]:


def Knn_classifier(train,row, k):
    #store {euclid sum , label} 
    my_dist = []
    l = len(row)
    count = 0
    for r in train[:,1:]:
        sum = 0
        for x in range(l):
            sum+=pow(row[x] - r[x],2)
            label = train [count][0]
        count+=1
        sum = math.sqrt(sum)
        my_dist.append((sum,label))
    
    my_dist.sort(key=lambda x: x[0])

    #stores {label , count) 
    predlabel = {}
    for x in range(k):
        res = my_dist[x][1]
        if res not in predlabel:
            predlabel[res] = 1
        predlabel[res] += 1
    ans = sorted(predlabel.items(),reverse = True)
    k = ans[0][0]
    return k


# In[140]:


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


# In[141]:


def find_correct(myans , myvalidate):
    global correct,tp,tn,fp,fn
    a = myvalidate[:][0]

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


# //////////////////
# filename=sys.argv[1]
# df2=pd.read_csv(filename)
# df2=df2.iloc[:,:-1] 
# test=df2.values
# //////////////////

df = pd.read_csv("Robot1", header=None, delimiter=r"\s+")
# validate = pd.read_csv("Robot1", header=None, delimiter=r"\s+")
train,validate = train_test_split(df, test_size = 0.2)
#remove last column from datas as insignificant
train = train.iloc[:, :-1]
validate = validate.iloc[:, :-1]

train1=train
val1=validate

train = train.values
myvalidate = validate.values

validate_arr = validate.iloc[:, 1:].astype(float)
validate_arr = validate_arr.values

acc_list = []
correct = 0
tp,fp,tn,fn=0,0,0,0
max_accuracy = 0
k_for_max_acc = 0
for i in range(1,12) :
    for row in range(len(validate_arr)):
        myans = Knn_classifier(train,validate_arr[row], i)
        corr = find_correct(myans , myvalidate[row]) 
    accu = (corr/float(len(validate_arr)))*100
    acc_list.append(accu)
    if(accu>max_accuracy):
        max_accuracy = accu
        k_for_max_acc = i
    correct = 0
    pfunction()
    
print("\nno of test samples", len(validate_arr))
print("Max_accuracy : ",max_accuracy, "for k :",k_for_max_acc)


# In[ ]:





# In[143]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 12), acc_list)
plt.title('Acc vs K Value')  
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy') 
        


# In[144]:


y_train = train1.pop(0)
x_train = train1
y_test = val1.pop(0)
x_test = val1

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=15)  
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))



# In[ ]:





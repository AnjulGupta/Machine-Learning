#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from pprint import pprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("train.csv")
train_df,test_df = train_test_split(df, test_size= 0.1)

grph = sns.FacetGrid(data=train_df, hue="left", aspect=2, height=5)
grph.map(plt.scatter,'satisfaction_level','average_montly_hours').add_legend()
grph.fig.suptitle('2-D plot of 2 attributes')


# In[ ]:





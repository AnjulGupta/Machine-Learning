#!/usr/bin/env python
# coding: utf-8

# # GIVEN PROBLEM

# ## Original data of network user activity is given. Original data also contains categorical feature.If you were to do dimensionality reduction on original data, could you use PCA? Justify. Write a paragraph in report for your explaination/justification.

# ## Answer -

# ### If data contains categorical feature, we cannot use PCA.For calculating PCA, we calculate the covariance matrix which uses Mean and standard Deviations, and calculating mean and standard devations for catergorical data (by converting them into integer classes) will not give any relevant values
# ### However, for continuous features we can use PCA.
# 
# 
# 

# In[ ]:





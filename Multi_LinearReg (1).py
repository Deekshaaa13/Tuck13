#!/usr/bin/env python
# coding: utf-8

# In[229]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[230]:


data=pd.read_csv("50_Startups.csv")


# In[231]:


x=data[['R&D Spend','Administration','Marketing Spend','State']]
y=data[['Profit']]


# In[ ]:


##Encodingg categorical variable as dummy variables of 0's and 1's aldo removing
##the first column of dummy variables to avoid dummy variable trap.


# In[232]:


x=pd.get_dummies(data=x,drop_first=True)
x.head()


# In[233]:


frames=[x,y]
df=pd.concat(frames,axis=1)


# In[247]:


df.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing'},inplace=True)


# In[251]:


df.head()


# In[252]:


y=data[['Profit']]
x=df[['RnD','Administration','Marketing','State_Florida','State_New York']]


# In[256]:


df.corr()


# In[236]:


import statsmodels.api as sm
x_sm=sm.add_constant(x)
model=sm.OLS(y,x_sm).fit()
model.summary() ##Administration and marketing spend coefficients are insignificant.


# In[246]:


import statsmodels.formula.api as smf


# In[253]:


ml_a= smf.ols("Profit~Administration",data=df).fit()
ml_a.summary() ##Building model with administration alone and the coefficient is still not significant.


# In[254]:


ml_m= smf.ols("Profit~Marketing",data=df).fit()
ml_m.summary() ##Building model with marketing alone and now the coefficient is significant.


# In[255]:


ml_am= smf.ols("Profit~Administration+Marketing",data=df).fit()
ml_am.summary()## Intercept becomes insignificant


# In[ ]:


##Finding the outlier and removing that entire row


# In[237]:


sm.graphics.influence_plot(model)


# In[258]:


df1= df.drop(df.index[[45,48,49]],axis=0) ## Removing rows 45,48 and 49


# In[275]:


y=df1[['Profit']]
x=df1[['RnD','Administration','Marketing','State_Florida','State_New York']]


# In[276]:


import statsmodels.api as sm
x_sm=sm.add_constant(x)
model1=sm.OLS(y,x_sm).fit()
model1.summary()


# In[270]:


sm.graphics.plot_partregress_grid(model1)## as the correlation value between Profit and Administration
#is low and the AV plot also shows the same.lets remove Administration variable


# In[277]:


y=df1[['Profit']]
x=df1[['RnD','Marketing','State_Florida','State_New York']]
x_sm=sm.add_constant(x)
modelfin=sm.OLS(y,x_sm).fit()
modelfin.summary() ## Building the final model


# In[264]:


y_pred=model.predict()


# In[330]:


df2=df1.copy()

x1=np.sqrt(df2[['RnD']])
x2=np.sqrt(df2[['Marketing']])
frames=[x1,x2]
x3=pd.concat(frames,axis=1)
x4=(df2[['State_Florida','State_New York']])
frames1=[x3,x4]
x5=pd.concat(frames1,axis=1)

y=np.sqrt(df2[['Profit']])


# In[331]:


x5.head()


# In[332]:


x_sm=sm.add_constant(x)
model2=sm.OLS(y,x_sm).fit()
model2.summary()


# In[333]:


df2=df1.copy()

x1=np.square(df2[['RnD']])
x2=np.square(df2[['Marketing']])
frames=[x1,x2]
x3=pd.concat(frames,axis=1)
x4=(df2[['State_Florida','State_New York']])
frames1=[x3,x4]
x5=pd.concat(frames1,axis=1)

y=np.square(df2[['Profit']])


# In[334]:


model3=sm.OLS(y,x_sm).fit()
model3.summary()


# In[335]:


df2=df1.copy()

x1=(df2[['RnD']])
x2=(df2[['Marketing']])
frames=[x1,x2]
x3=pd.concat(frames,axis=1)
x4=(df2[['State_Florida','State_New York']])
frames1=[x3,x4]
x5=pd.concat(frames1,axis=1)

y=np.log10(df2[['Profit']])


# In[336]:


model4=sm.OLS(y,x_sm).fit()
model4.summary()


# In[337]:


df2=df1.copy()

x1=(df2[['RnD']])
x2=(df2[['Marketing']])
frames=[x1,x2]
x3=pd.concat(frames,axis=1)
x4=(df2[['State_Florida','State_New York']])
frames1=[x3,x4]
x5=pd.concat(frames1,axis=1)

y=np.reciprocal(df2[['Profit']])


# In[338]:


model5=sm.OLS(y,x_sm).fit()
model5.summary()


# In[339]:


df2=df1.copy()

x1=np.sqrt(df2[['RnD']])
x2=np.sqrt(df2[['Marketing']])
frames=[x1,x2]
x3=pd.concat(frames,axis=1)
x4=(df2[['State_Florida','State_New York']])
frames1=[x3,x4]
x5=pd.concat(frames1,axis=1)

y=(df2[['Profit']])


# In[340]:


model6=sm.OLS(y,x_sm).fit()
model6.summary()


# In[ ]:





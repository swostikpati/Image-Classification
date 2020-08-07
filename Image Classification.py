
# coding: utf-8

# In[2]:


#importing libraries

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier # combines prediction of various estimators to improve generisability
from sklearn.cross_validation import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#usimg pands to read database
data = pd.read_csv('mnist.csv')
#data has how become a pandas function.
'''The mnist_train.csv file contains the 60,000 training examples and labels. 
The mnist_test.csv contains 10,000 test examples and labels. 
Each row consists of 785 values: the first value is the label (a number from 0 to 9) 
and the remaining 784 values are the pixel values (a number from 0 to 255).'''


# In[6]:


#viewing the database
#column heads
data.head()


# In[7]:


#The label value shows the numbr the pixels represent. The column before it represents the indexes of the position.


# In[32]:


#extracting data
a=data.iloc[3,1:].values # the third row is chosen with pixels from all the columns.
#iloc[row indexes,column indexes]


# In[33]:


# reshaping into a resonable size
#reshaping needs to be in such a way such that the dimensions are factors of the total size of the array.
a = a.reshape(28,28).astype('uint8') # reshaped as 28x28. uint8 = unsigned integer value of 8-bit(range of pixel)-optional
plt.imshow(a)


# In[29]:


#The output represents the actual image that the pixels if row 4 reprsent. 
#Now to train our model the label values aren't required. So we separate the labels from the pixels.


# In[35]:


#preparing the data
#separating labels and data values
df_x = data.iloc[:,1:]#everything except labels
df_y = data.iloc[:,0]#only labels


# In[38]:


#splitting data in test and train.

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#inbuilt function in sklearn. It spilts the data into test and train. The test_size represents the fraction of data splitting 
#random state represents randomisation in the splitting of data. There are various random states


# In[44]:


#check data
x_train.head()
y_train.head()


# In[45]:


#Randomforest classifier works on the basis of decision trees. Here the number of decision trees is 100(nestimator)


# In[46]:


#call rf classifier
rf = RandomForestClassifier(n_estimators=100)


# In[48]:


#fit the model-training the model
rf.fit(x_train,y_train)


# In[49]:


#prediction on test data
pred = rf.predict(x_test)


# In[50]:


pred


# In[51]:


#we gave the pixels to predict on and now we will compare the actual labels with the predition


# In[61]:


#check prediction accuracy
s= y_test.values
#calculate number of correctly predicted values
total=len(pred)
count=0
for i in range(total):
    if pred[i] == s[i]:
        count+=1


# In[62]:


#accuracy
count/total*100


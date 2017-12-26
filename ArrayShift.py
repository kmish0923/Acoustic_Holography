
# coding: utf-8

# In[11]:

pwd


# In[59]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.special import*
import math


# In[68]:

ab = np.loadtxt('11-05_amp_200_8kHz_waterbottle.txt',unpack=True)
a = np.loadtxt('11-05_amp_200_8kHz_background.txt',unpack=True)


# In[78]:

import numpy as np

shift = 0
m = len(a)
n = len(a[0])

array= np.empty([m,n])

for i in range(m): 
    for j in range(n):
        while j < shift:
            array[i][j] =0
            break
        while (j >= shift):
            array[i][j]= a[i][j-shift]
            print i
            print j-shift
            break
        
array = array[0:128]


# In[ ]:

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(a)
fig.add_subplot(1,2,2)
plt.imshow(array)


# In[ ]:

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(ab)
fig.add_subplot(1,2,2)
plt.imshow(array)


# In[77]:

af = array-a
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(af)


# In[ ]:




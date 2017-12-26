
# coding: utf-8

# <center> <h1> Water Bottle Analysis: Producing 3D images.

# In[8]:

# Import rayleighsommerfeld.
from rayleighsommerfeld import rayleighsommerfeld


# In[9]:

pwd


# In[10]:

get_ipython().magic(u'matplotlib inline')
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from aziavg import *
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.special import*
import math


# Loading images made with our custom acoustical camera.

# In[11]:

ab1 = np.loadtxt('11-03_amp_200_8kHz_background.txt',unpack=True)
pb1 = np.loadtxt('11-03_phi_200_8kHz_background.txt', unpack=True)
pb1 = (pb1 - pb1.min())
sb1 = ab1 * np.exp(1j * pb1)
'b1 = bare tunnel 1, before the run'


# In[12]:

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(ab1)
fig.add_subplot(1,2,2)
plt.imshow(pb1)


# In[13]:

aw = np.loadtxt('11-03_amp_200_8kHz_waterbottle.txt',unpack=True)
pw = np.loadtxt('11-03_phi_200_8kHz_waterbottle.txt', unpack=True)
pw = (pw - pw.min())
sw = aw * np.exp(1j * pw)


# In[14]:

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(aw)
fig.add_subplot(1,2,2)
plt.imshow(pw)
'w = water bottle'


# In[15]:

'f = final'
sf = sw-sb1
pf = np.angle(sf) 
af = np.absolute(sf)


# In[16]:

fig = plt.figure()
fig.add_subplot(1,2,1)
#plt.imshow(af, cmap='gray')
plt.imshow(af)
fig.add_subplot(1,2,2)
plt.imshow(pf, cmap='gray')


# In[17]:

amplitude_final = af
phase_final = pf
normalized_field = af * np.exp(1.j * pf)


# In[18]:

lamb = 0.042534 # [m]
mpp = 0.004885 # [m]
z = 0.38 # [m]
shifted_norm_field = rayleighsommerfeld(normalized_field, z, lamb, mpp)
shifted_norm_field = shifted_norm_field.reshape(128,128)
real_shifted_norm = np.real(shifted_norm_field)


# In[25]:

z = 380+(1200*19) # [m]


for i in range (0,z,200):
    i = float(i)
    z_val = float(i/1000)
    shifted_norm_field = rayleighsommerfeld(normalized_field, z_val, lamb, mpp)
    shifted_norm_field = shifted_norm_field.reshape(128,128)
    real_shifted_norm = np.real(shifted_norm_field)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(np.real(shifted_norm_field))
    plt.savefig("Instant{}.png".format(i))

   

    
#anim = animation.FuncAnimation(fig, animate, init_func=init,
                               #frames=200, interval=20, blit=True)


# In[23]:

z = 380+(1200*19) # [m]

shifted_norm_field = rayleighsommerfeld(normalized_field, z, lamb, mpp)
shifted_norm_field = shifted_norm_field.reshape(128,128)
real_shifted_norm = np.real(shifted_norm_field)
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(np.real(shifted_norm_field))



# In[ ]:

np.shape()


# In[52]:

images = []
filenames = "/Users/kim/Desktop/GRIER/filenames"
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images)


# In[12]:

print(np.mean(real_shifted_norm))


# In[13]:

fig, ax = plt.subplots(figsize=(8,8))

ax.imshow(np.real(shifted_norm_field))


# In[14]:

ab2 = np.loadtxt('WB/11-03_amp_200_8kHz_background2.txt',unpack=True)
pb2 = np.loadtxt('WB/11-03_phi_200_8kHz_background2.txt', unpack=True)
pb2 = (pb2 - pb2.min())
sb2 = ab2 * np.exp(1j * pb2)
'b2 = bare tunnel 2, after the run'


# In[15]:

'e = empty'
se = sb2-sb1
pe = np.angle(se) 
ae = np.absolute(se)
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(ae)
fig.add_subplot(1,2,2)
plt.imshow(pe)


# In[16]:

'f1 = final other'
sf1 = sw-sb2
pf1 = np.angle(sf1) 
af1 = np.absolute(sf1)
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(af1)
fig.add_subplot(1,2,2)
plt.imshow(pf1)


# In[ ]:




# In[17]:

ab1 = np.loadtxt('WB/11-05_amp_200_8kHz_background.txt',unpack=True)
pb1 = np.loadtxt('WB/11-05_phi_200_8kHz_background.txt', unpack=True)
pb1 = (pb1 - pb1.min())
sb1 = ab1 * np.exp(1j * pb1)
aw = np.loadtxt('WB/11-05_amp_200_8kHz_waterbottle.txt',unpack=True)
pw = np.loadtxt('WB/11-05_phi_200_8kHz_waterbottle.txt', unpack=True)
pw = (pw - pw.min())
sw = aw * np.exp(1j * pw)
ab2 = np.loadtxt('WB/11-05_amp_200_8kHz_background2.txt',unpack=True)
pb2 = np.loadtxt('WB/11-05_phi_200_8kHz_background2.txt', unpack=True)
pb2 = (pb2 - pb2.min())
sb2 = ab2 * np.exp(1j * pb2)
sf = sw-sb1
pf = np.angle(sf) 
af = np.absolute(sf)
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(af)
fig.add_subplot(1,2,2)
plt.imshow(pf)


# In[ ]:

se = sb2-sb1
pe = np.angle(se) 
ae = np.absolute(se)
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(ae)
fig.add_subplot(1,2,2)
plt.imshow(pe)


# In[ ]:

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(ab1)
fig.add_subplot(1,2,2)
plt.imshow(pb1)


# In[ ]:

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(aw)
fig.add_subplot(1,2,2)
plt.imshow(pw)


# In[ ]:




# In[ ]:

#mean values of amplitudes 
#shifting by pixels 
#one can be louder- (mean values of amp/pressures = same on average?)

#data in array form, make a second array that's the same size and loop through every row 
#and column- (columns should map over), for loop 


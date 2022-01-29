#!/usr/bin/env python
# coding: utf-8

# # HOMEWORK ONE

# In[1]:


#Checking my system
from platform import python_version
print(python_version())


# In[2]:


#Imports
import decimal 
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from time import time


# ## Problem One

# - Part A

# In[3]:


A = 0.1
AA = decimal.Decimal(A)

print("Initialized Value = " +str(A))
print("Value given double percision = " +str(AA))
print("Floating Point Error for Double = 0.0" +str(str(AA)[3:]))
print("Number of Decimal Places = " +str(len(str(AA)) - 2)) #Subtract one for decimal and one for the zero before the decimal


# In[4]:


AAA = np.float32(0.1)

print("Value given single percision = "+str('%.57f' % AAA))
print("Floating Point Error for Single= 0.0" +str(str('%.57f' % AAA)[3:]))


# Changing from double percision to single percision seems to have decreased the level of accuracy that we get. Meaning that we go from a floating point error value of about 5.55e-18 (double) to a value of 1.5e-9 (single). This makes sense, since we are keeping track of less info.

# - Part B

# In[5]:


#Values 
B = np.arange(100)
ee = np.float32(0.1)
e = 0.1


# In[6]:


#Roundoff error for the single
for i in B:
    ee = ee / 10
    #print(1+ee)
    if (1.0+ee) == 1.0:
        print("Round-off error for single = " +str("{:30e}".format(ee)))
        break


# In[7]:


#Roundoff error for the double
for i in B:
    e = e / 10
    #print(1+e)
    if (1.0+e) == 1.0:
        print("Round-off error for Double = " +str("{:30e}".format(e)))
        break


# I'm relizing that maybe I didn't need to do both of the methods for this part of the problem. Testing the foat32 this way isn't testing what I thought it would initially. 

# - Part C

# I checked my values with Kaelee. She has a Macbook Pro 2.7 GHz Intel core i5. We ended up getting about the same values for a and b. Other people in class with different computers all seem to be getting very similar (or exact) results as well. Since it didn't seem to matter which computer setup we used, I wonder when it would start to make a difference?

# ## Problem Two

# - Part A

# In[8]:


#Definitions for the two methods -- version to see what's happeing 
    #Ended up rewriting this because I couldn't get the leading and trailing to work right
def Rectangle(function, a, b, n): 
    
    #Values and Arrays
    area_array = []
    i_array = []
    
    #Equations
    delta_x = np.divide((b - a), n)
    
    for i in np.arange(n):
        area_array.append(function(a + (i - 1) * delta_x))
        i_array.append(i)
    
    return [delta_x, area_array, np.sum(area_array) * delta_x, i_array]

def Trapezoid(function, a, b, n):
    
    #Values and Arrays
    area_array = []
    i_array = []
    
    #Equations
    delta_x = np.divide((b - a), n)
    
    for i in np.arange(n):
        area_array.append((function(a + (i - 1) * delta_x) + function(a + (i) * delta_x)) / 2)
        i_array.append(i)
    
    return [delta_x, area_array, np.sum(area_array) * delta_x, i_array]


# In[9]:


#Definitions for the two methods -- Faster version for calculations
def Rectangle_F(function, a, b, n): 
    
    #Equations
    delta_x = np.divide((b - a), n)
    N = np.arange(1, n)
    area = np.sum([function(a + (i - 1)* delta_x) for i in N])
    
    return area * delta_x

def Trapezoid_F(function, a, b, n):
    
    #Equations
    delta_x = np.divide((b - a), n)
    N = np.arange(1, n)
    area = np.sum([((function(a + (i - 1) * delta_x) + function(a + (i) * delta_x)) / 2) for i in N]) 
    
    return (area) * delta_x


# In[10]:


#Defining the function
def Function(x):
    return x**(-3/2)


# In[11]:


#Checking the values --see what's happening
print(Trapezoid(Function, 1, 5, 10000)[0])
print(Rectangle(Function, 1, 5, 10000)[2])
print(Trapezoid(Function, 1, 5, 100000)[0])
print(Rectangle(Function, 1, 5, 100000)[2])

#print(Rectangle(Function, 1, 5, 10000)[1])
#print(Rectangle(Function, 1, 5, 10000)[3])


# In[12]:


#Testing to make sure its working -- Fast
print(Trapezoid_F(Function, 1, 5, 10000))
print(Rectangle_F(Function, 1, 5, 10000))
print(Trapezoid_F(Function, 1, 5, 100000))
print(Rectangle_F(Function, 1, 5, 100000))


# In[13]:


#Just to see what's happening 
fig = plt.figure(figsize = (20,8), facecolor = "black")
ax = plt.subplot(1,1,1) 
#plt.plot(Rectangle(Function, 1, 5, 10000)[1], Rectangle(Function, 1, 5, 10000)[3], color = "lightseagreen", linewidth = 3, linestyle = '-', label = "Rectangle", zorder = 0)  
plt.plot(Trapezoid(Function, 1, 5, 10000)[1], Trapezoid(Function, 1, 5, 10000)[3], color = "gold", linewidth = 3, linestyle = '--', label = "Triangle", zorder = 1)  

ax.tick_params(direction='inout', length=10, width=2, colors='white', grid_color='white', grid_alpha=0.5)
ax.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["right"].set_color("white")
plt.setp(ax.spines.values(), linewidth=2)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(3,4))
plt.rc('font', size = 15) #Fixes the scientific notation font size

plt.legend(facecolor = "black", labelcolor = "white", fontsize = 15, frameon = False, bbox_to_anchor=(1, 1), loc='upper left')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.xlabel(r'Redshift'  , size = '20', color = "white")
#plt.ylabel(r'Comoving Distance [Mpc]' , size = '20', color = "white")
plt.xticks(size = '15')
plt.yticks(size = '15')

plt.show()


# In[14]:


#Just to see what's happening 
fig = plt.figure(figsize = (20,8), facecolor = "black")
ax = plt.subplot(1,1,1) 
plt.plot(Rectangle(Function, 1, 5, 10000)[1], Rectangle(Function, 1, 5, 10000)[3], color = "lightseagreen", linewidth = 3, linestyle = '-', label = "Rectangle", zorder = 0)  
#plt.plot(Trapezoid(Function, 1, 5, 10000)[1], Trapezoid(Function, 1, 5, 10000)[3], color = "gold", linewidth = 3, linestyle = '--', label = "Triangle", zorder = 1)  

ax.tick_params(direction='inout', length=10, width=2, colors='white', grid_color='white', grid_alpha=0.5)
ax.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["right"].set_color("white")
plt.setp(ax.spines.values(), linewidth=2)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(3,4))
plt.rc('font', size = 15) #Fixes the scientific notation font size

plt.legend(facecolor = "black", labelcolor = "white", fontsize = 15, frameon = False, bbox_to_anchor=(1, 1), loc='upper left')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.xlabel(r'Redshift'  , size = '20', color = "white")
#plt.ylabel(r'Comoving Distance [Mpc]' , size = '20', color = "white")
plt.xticks(size = '15')
plt.yticks(size = '15')

plt.show()


# In[17]:


#Values and arrays
Exact_value = 1.1056
steps = np.arange(1, 10**3)

step_size = []
Fract_Error_Rec = []
Fract_Error_Tra = []


# In[18]:


#Plotting the error against step size
start = time()
for i in steps:
    Calc_value = Rectangle(Function, 1, 5, i)[2]
    Fract_Error_Rec.append(abs(Calc_value - Exact_value))
    step_size.append(Rectangle(Function, 1, 5, i)[0])
print(f'Time: {time() - start} seconds')

start = time()
for i in steps:
    Calc_value = Trapezoid(Function, 1, 5, i)[2]
    Fract_Error_Tra.append(abs(Calc_value - Exact_value))
print(f'Time: {time() - start} seconds')


# In[19]:


#Plotting the Error
fig = plt.figure(figsize = (20,8), facecolor = "black")
ax = plt.subplot(1,1,1) 
plt.loglog(step_size, Fract_Error_Rec, color = "lightseagreen", linewidth = 3, linestyle = '-', label = "Rectangle")  
plt.loglog(step_size, Fract_Error_Tra, color = "salmon", linewidth = 3, linestyle = '--', label = "Trapezoid")

ax.tick_params(direction='inout', length=10, width=2, colors='white', grid_color='white', grid_alpha=0.5)
ax.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["right"].set_color("white")
plt.setp(ax.spines.values(), linewidth=2)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(3,4))
plt.rc('font', size = 15) #Fixes the scientific notation font size

plt.legend(facecolor = "black", labelcolor = "white", fontsize = 15, frameon = False, bbox_to_anchor=(1, 1), loc='upper left')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel(r'Step Size'  , size = '20', color = "white")
plt.ylabel(r'Error' , size = '20', color = "white")
plt.xticks(size = '15')
plt.yticks(size = '15')

plt.show()


# In[20]:


#Values 
steps = np.arange(10**5)


# In[23]:


#Finding the number of steps 
start = time()
for i in steps:
    Calc_value = Trapezoid_F(Function, 1, 5, i)
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-3):
        print("Value = " +str(Calc_value))
        print("The fractional error for the Trapezoid method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break  
print(f'Time: {time() - start} seconds')

start = time()
for i in steps:
    Calc_value = Rectangle_F(Function, 1, 5, i)
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-3):
        print("Value = " +str(Calc_value))
        print("The fractional error for the Rectangle method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break
print(f'Time: {time() - start} seconds')


# In[32]:


#Values 
steps = np.arange(10**7, 10**8, 10000000) #Checked this range with large steps first to save time on computer
    #Ended up with a value above 10**8 (My computer was not having a good time so I went ahead an stopped it here.)
    #I'm sure I can figure out a better way to do this


# In[33]:


#Finding the number of steps 
start = time()
for i in steps:
    print(i)
    Calc_value = Rectangle_F(Function, 1, 5, i)
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-5):
        print("Value = " +str(Calc_value))
        print("The fractional error for the Rectangle method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break
print(f'Time: {time() - start} seconds')


# In[57]:


#Values 
steps = np.arange(10**7, 10**8, 10000000) #Checked this range with large steps first to save time on computer
    #Ended up with a value above 10**8 (same as for the Rectangle)


# In[ ]:


#Finding the number of steps 
start = time()
for i in steps:
    print(i)
    Calc_value = Trapezoid_F(Function, 1, 5, i)
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-5):
        print("Value = " +str(Calc_value))
        print("The fractional error for the Trapezoid method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break   
print(f'Time: {time() - start} seconds')


# Looking first at the 10^-3 value, we find that it takes about 1293 and 327 steps for the Rectangle and Trapezoid methods respectively. Then, looking at the 10^-5 value, we find that it takes over 10^8 and 25800 steps for the Rectangle and Trapezoid methods respectively. (This seems to be about what we expect to see in the relationship between these two methods)

# If we were to run more consuming code we would need to weigh the importance of each of these aspects carefully. From this coding exercise it seems that the method that takes less time to run (Trapezoid) actually reaches the desired error with fewer steps than the other (Rectangle) method.

# - Part B

# In[49]:


#Checking a Black-Box Version
start = time()
C = integrate.quad(Function, 1, 5)
print("Value = " +str(C[0]), "Uncertianty = " +str(C[1]))
print(f'Time: {time() - start} seconds')


# This was taking too long and almost crashed my computer 

# #Values 
# steps = np.arange(10**8, 10**9, 100000000) #Checked this range with large steps (10**6, 10**8, 100000000) first to save time on computer
#     #Ended up with a value above 177000

# start = time()
# #Change the step size intervals 
# 
# for i in steps:
#     print(i)
#     Calc_value = Trapezoid_F(Function, 1, 5, i)
#     Fractional_error = abs(Calc_value - Exact_value) / Exact_value
#     
#     if Fractional_error < (10**-6):
#         print("Value = " +str(Calc_value))
#         print("The fractional error for the Rectangle method = " +str("{:e}".format(Fractional_error)))
#         print("For this number of steps = " +str(i))
#         break 
# print(f'Time: {time() - start} seconds')

# Using scipy.integrate.quad it seems that we can get a value with an uncertianty of order of 10^-13.

# The defualt approach to this method of intergration comes from a fortran library QUADPACK according to the docs. From the QUADPACK wiki page, I found that this method utalizes a 21-point Gauss-Kronrod quadrature approach. Which seems to use a guassian curve instead of a rectangle or trapeziod and weighted sums. (I'm a little confused about what this would look like)
#     
# https://en.wikipedia.org/wiki/QUADPACK -> Link to wiki page that I found info from. 

# Using the Trapezoid method (since it needed less steps to reach the level of accuracy we asked for) is seems that in order to obtain the same 10^-13 level of accuracy we need to reach a step size of ... I was not able to run my code to a small enough step size value in order to reach the 10^-13 value (it tried to crash my computer). (I'm sure there is a better way to run the code, so that I could reach 10^-13 or at least higher than what I got. I'm just not currently aware of how to make it more efficient.)  

# ## Problem Three

# - Part B (Choosing the cosmology problem)

# In[50]:


#Values
Z = 2.0
Z_range = np.arange(0.1, 10 + 1, 0.1)


# In[51]:


#The integral 
def comov(z):
    return (((0.3 * (1 + z)**3) + 0.7)**(-1/2)) * 3000


# In[52]:


#The Comoving distance for a redshift of 2
print("The Comoving distance for z=2 is " +str(Trapezoid(comov, 0, Z, 1000)[0])+ " [h^-1 Mpc]")


# In[53]:


D = []

for i in Z_range:
    D.append(Trapezoid(comov , 0, i, 10**3)[0])


# In[54]:


fig = plt.figure(figsize = (20,8), facecolor = "black")
ax = plt.subplot(1,1,1) 
plt.plot(Z_range, D, color = "lightseagreen", linewidth = 3, linestyle = '-')  

ax.tick_params(direction='inout', length=10, width=2, colors='white', grid_color='white', grid_alpha=0.5)
ax.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["right"].set_color("white")
plt.setp(ax.spines.values(), linewidth=2)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(3,4))
plt.rc('font', size = 15) #Fixes the scientific notation font size

#plt.legend(facecolor = "black", labelcolor = "white", fontsize = 15, frameon = False, bbox_to_anchor=(1, 1), loc='upper left')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel(r'Redshift'  , size = '20', color = "white")
plt.ylabel(r'Comoving Distance [$h^{-1}$ Mpc]' , size = '20', color = "white")
plt.xticks(size = '15')
plt.yticks(size = '15')

plt.ylim(0, 7000)
plt.xlim(0, 10)
plt.show()


# I went ahead and left in the h^-1 factor in the above, since we tend to ignore or report values in terms of h^-1. 

# In[ ]:





# In[ ]:





# In[ ]:





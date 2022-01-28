#!/usr/bin/env python
# coding: utf-8

# # HOMEWORK ONE

# In[256]:


#Checking my system
from platform import python_version
print(python_version())


# In[257]:


#Imports
import decimal 
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from time import time


# ## Problem One

# - Part A

# In[258]:


A = 0.1
AA = decimal.Decimal(A)

print("Initialized Value = " +str(A))
print("Value given double percision = " +str(AA))
print("Floating Point Error for Double = 0.0" +str(str(AA)[3:]))
print("Number of Decimal Places = " +str(len(str(AA)) - 2)) #Subtract one for decimal and one for the zero before the decimal


# In[259]:


AAA = np.float32(0.1)

print("Value given single percision = "+str('%.57f' % AAA))
print("Floating Point Error for Single= 0.0" +str(str('%.57f' % AAA)[3:]))


# Changing from double percision to single percision seems to have decreased the level of accuracy that we get. Meaning that we go from a floating point error value of about 5.55e-18 (double) to a value of 1.5e-9 (single).

# - Part B

# In[260]:


#Values 
B = np.arange(100)
ee = np.float32(0.1)
e = 0.1


# In[261]:


#Roundoff error for the single
for i in B:
    ee = ee / 10
    #print(1+ee)
    if (1.0+ee) == 1.0:
        print("Round-off error for single = " +str("{:30e}".format(ee)))
        break


# In[262]:


#Roundoff error for the double
for i in B:
    e = e / 10
    #print(1+e)
    if (1.0+e) == 1.0:
        print("Round-off error for Double = " +str("{:30e}".format(e)))
        break


# I'm relizing that maybe I didn't need to do both of the methods for this part of the problem.

# - Part C

# I checked my values with Kaelee. She has a Macbook Pro 2.7 GHz Intel core i5. We ended up getting about the same values for a and b. Checking with others Mac book pro 2.7Ghz intel core i5. Other people in class with different computers all seem to be getting very similar (or exact) results as well. 

# ## Problem Two

# - Part A

# In[263]:


#Definitions for the two methods
def Rectangle(function, a, b, n): #Fix the left and not the midpoint
    
    leading_x = []
    trailing_x = []
    area_array = []
    
    area_total = 0
    
    delta_x = np.divide((b - a), n)

    x_1 = a
    x_2 = a + delta_x

    while (a <= x_2 <= b) or (a >= x_2 >= b):
        
        area = (function((a + (x_1-1)))) * delta_x
        area_total = area_total + area
        
        leading_x.append(x_1)
        trailing_x.append(x_2)
        area_array.append(area)
        
        x_1 = x_1 + delta_x
        x_2 = x_2 + delta_x

    return [area_total, leading_x, trailing_x, area_array, delta_x]


def Trapezoid(function , a, b, n):
    
    leading_x = []
    trailing_x = []
    area_array = []
    
    area_total = 0

    delta_x = np.divide((b - a), n)

    x_1 = a
    x_2 = a + delta_x

    while (a <= x_2 <= b) or (a >= x_2 >=b):
        
        area = ((function((a + (x_1-1))) + function(x_2)) / 2) * delta_x
        area_total = area_total + area
        
        leading_x.append(x_1)
        trailing_x.append(x_2)
        area_array.append(area)
        
        x_1 = x_1 + delta_x
        x_2 = x_2 + delta_x

    return [area_total, leading_x, trailing_x, area_array, delta_x]


# The above Rectangle and Trapezoid definitions where adapted from: http://specminor.org/2017/01/05/numerical-integration-python.html

# In[264]:


#Defining the function
def Function(x):
    return x**(-3/2)


# In[265]:


#Just to see what's happening 
fig = plt.figure(figsize = (20,8), facecolor = "black")
ax = plt.subplot(1,1,1) 
plt.plot(Rectangle(Function, 1, 5, 10000)[1], Rectangle(Function, 1, 5, 10000)[3], color = "lightseagreen", linewidth = 3, linestyle = '-', label = "Rectangle", zorder = 0)  
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


# In[266]:


#Checking the values 
print(Trapezoid(Function, 1, 5, 10000)[0])
print(Rectangle(Function, 1, 5, 10000)[0])
print(Trapezoid(Function, 1, 5, 100000)[0])
print(Rectangle(Function, 1, 5, 100000)[0])


# In[267]:


#Values and arrays
steps = np.arange(10**3)

step_size = []
Fract_Error_Rec = []
Fract_Error_Tra = []


# In[268]:


#Plotting the error against step size
start = time()
for i in steps:
    Calc_value = Rectangle(Function, 1, 5, i)[0]
    Fract_Error_Rec.append(abs(Calc_value - Exact_value))
    step_size.append(Rectangle(Function, 1, 5, i)[4])
print(f'Time: {time() - start} seconds')

start = time()
for i in steps:
    Calc_value = Trapezoid(Function, 1, 5, i)[0]
    Fract_Error_Tra.append(abs(Calc_value - Exact_value))
print(f'Time: {time() - start} seconds')


# In[269]:


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


# In[270]:


#Values 
steps = np.arange(10**5)


# In[271]:


#Finding the number of steps 

start = time()
for i in steps:
    Calc_value = Rectangle(Function, 1, 5, i)[0]
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-3):
        print("The fractional error for the Rectangle method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break
print(f'Time: {time() - start} seconds')

start = time()
for i in steps:
    Calc_value = Trapezoid(Function, 1, 5, i)[0]
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-3):
        print("The fractional error for the Trapezoid method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break  
print(f'Time: {time() - start} seconds')


# In[272]:


#Values 
steps = np.arange(10**5)


# In[ ]:


#Finding the number of steps 

start = time()
for i in steps:
    Calc_value = Rectangle(Function, 1, 5, i)[0]
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-5):
        print("The fractional error for the Rectangle method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break
print(f'Time: {time() - start} seconds')

start = time()
for i in steps:
    Calc_value = Trapezoid(Function, 1, 5, i)[0]
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-5):
        print("The fractional error for the Trapezoid method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break   
print(f'Time: {time() - start} seconds')


# Looking first at the 10^-3 value, we find that it takes about 1330 and 44 steps for the Rectangle and Trapezoid methods respectively. Then, looking at the 10^-5 value, we find that it takes about ___ and ___ steps for the Rectangle and Trapezoid methods respectively. (This seems to be about what we expect to see in the relationship between these two methods)

# If we were to run more consuming code we would need to weigh the importance of each of these aspects carefully. From this coding exercise it seems that the method that takes less time to run (Trapezoid) actually reaches the desired error with fewer steps than the other (Rectangle) method. (The run time has an error to it, but seems to keep a similar ratio with each run) 

# - Part B

# In[255]:


start = time()
C = integrate.quad(Function, 1, 5)
print("Value = " +str(C[0]), "Uncertianty = " +str(C[1]))
print(f'Time: {time() - start} seconds')


# In[ ]:


#Taking out some stuff to make this run better for the purpose of answering this question

def Trapezoid(function , a, b, n):
    
    area_total = 0

    delta_x = np.divide((b - a), n)

    x_1 = a
    x_2 = a + delta_x

    while (a <= x_2 <= b) or (a >= x_2 >=b):
        
        area = ((function(a + (x_2-1)) + function(x_2)) / 2) * delta_x
        area_total = area_total + area

        
        x_1 = x_1 + delta_x
        x_2 = x_2 + delta_x

    return area_total


# In[ ]:


start = time()
#Change the step size intervals 

for i in steps:
    Calc_value = Rectangle_f(Function, 1, 5, i)
    Fractional_error = abs(Calc_value - Exact_value) / Exact_value
    
    if Fractional_error < (10**-8):
        print("The fractional error for the Rectangle method = " +str("{:e}".format(Fractional_error)))
        print("For this number of steps = " +str(i))
        break
       
print(f'Time: {time() - start} seconds')


# Using scipy.integrate.quad it seems that we can get a value with an uncertianty of order of 10^-13.

# The defualt approach to this method of intergration comes from a fortran library QUADPACK according to the docs. From the QUADPACK wiki page, I found that this method utalizes a 21-point Gauss-Kronrod quadrature approach. 
#     
# https://en.wikipedia.org/wiki/QUADPACK -> Link to wiki page that I found info from. 

# Using the Trapezoid method (since it needed less steps to reach the level of accuracy we asked for) is seems that in order to obtain the same 10^-13 level of accuracy we need to reach a step size of ... I was not able to run my code to a small enough step size value in order to reach the 10^-13 value (it tried to crash my computer). The closest value I was able to get to was 10^-8 with a step size of 591. 

# ## Problem Three

# - Part B (Choosing the cosmology problem)

# In[ ]:


#Values
Z = 2.0
Z_range = np.arange(0.1, 10 + 1, 0.1)


# In[ ]:


#The integral 
def comov(z):
    return (((0.3 * (1 + z)**3) + 0.7)**(-1/2)) * 3000


# In[ ]:


#The Comoving distance for a redshift of 2
print("The Comoving distance for z=2 is " +str(Trapezoid(comov, 0, Z, 1000)[0])+ " [h^-1 Mpc]")


# In[ ]:


D = []

for i in Z_range:
    D.append(Trapezoid(comov , 0, i, 10**3)[0])


# In[ ]:


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


# I went ahead and left in the h^-1 factor in the above, because its "close" to one. 

# In[ ]:





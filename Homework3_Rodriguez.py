#importing packages 
import numpy as np 
import matplotlib.pyplot as plt
from astropy import constants as const 


#Homework 3 - Exercise 6.16: The Lagrange Point"
#Write a program that uses either Newton's method or the secant method to solve for the distance r from the Earth to the L_1 point. "


#Newton's Method: a method for finding successively better approximations to the root (or zeros) of a real-valued function. 

#Important equations needed for the Newton's Method 
#f'(x)=f(x)/delta x
#New guess for x': x' = x - delta x = x - f(x)/f'(x)


#Define constants needed for the function 
m = 7.348e22 #Moon's mass in kg
M = const.M_earth.value #Earth's mass in kg from astropy 
R = 3.844e8 #Earth-Moon distance in m 
w = 2.662e-6 #omega value in s^-1
G = const.G.value #Gravitational Constant in m^3/kg*s^2 from astropy 



#Let's define functions needed for the Newton's Method 
def grav_func(r): #equation when set = 0 
    return ((G*M)/(r**2))-((G*m)/((R-r)**2))-(w**2)*r
#Let's define derivative function for the Newton's Method
def div_grav_func(r):
    return ((-2*G*M)/(r**3))-((2*G*m)/((R-r)**3))-(w**2)


#Plotting the function against r to estimate the best guess for r value. 
#We want to make a guess an r value based on when it intercepts the 0 axis. 

r = np.linspace(1e8, 3.8e8, 1000) #R = 3.844e8 is my limit since that is the Earth-Moon distance. r can not equal 0 since you cannot divide by 0! (Related to the grav_func) 
plt.figure(figsize=(10, 6))
plt.plot(r, grav_func(r))
plt.xlabel('r (distance)')
plt.ylabel('Gravitational function')
plt.title('Gravitational Function vs Distance')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Zero line')

# Add tick marks to help determine guess r value. 
plt.minorticks_on()  # This adds minor tick marks
plt.tick_params(which='both', width=1)  
plt.tick_params(which='major', length=8)  # Major tick length
plt.tick_params(which='minor', length=5)  # Minor tick length

plt.legend()
plt.show()


#Now I will make an initial guess for r where it crosses the 0 axis. 
r_guess = 3.3e8 #since it crosses 0 between 3.0 and 3.5 r, I will estimate something in between. 

#I will now implement my guess using the Newton Method new guess form: x' = x - delta x = x - f(x)/f'(x). Where x is replace with r. 

def newton_method(r_guess, tolerance=1e-10, max_iterations=1000):  #tolerance: the pre-defined small error threshold that determines when the iterative process should stop because the approximations are sufficiently close to the true root. 
    r = r_guess
    iteration = 0
    
    while iteration < max_iterations:
        f_r = grav_func(r)
        df_r = div_grav_func(r)
        
        r_new = r - f_r / df_r
        
    
        if abs(r_new - r) < tolerance:
            return r_new, iteration
        
        r = r_new
        iteration += 1
        
    return r, max_iterations

#This allows me to print the distance r from the Earth to the L_1 point. "
result, iterations = newton_method(r_guess)
print(f"L1 point distance from Earth: {result:.2f} m ({result/1e6:.3f} Mm)")


#Plotting the function against r to estimate the best guess for r value. 
#We want to make a guess an r value based on when it intercepts the 0 axis. 

r = np.linspace(1e8, 3.8e8, 1000) #R = 3.844e8 is my limit since that is the Earth-Moon distance. r can not equal 0 since you cannot divide by 0! (Related to the grav_func) 
plt.figure(figsize=(10, 6))
plt.plot(r, grav_func(r))
plt.xlabel('r (distance)')
plt.ylabel('Gravitational function')
plt.title('Gravitational Function vs Distance [Newton]')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Zero line')

# Add tick marks to help determine guess r value. 
plt.minorticks_on()  # This adds minor tick marks
plt.tick_params(which='both', width=1)  
plt.tick_params(which='major', length=8)  # Major tick length
plt.tick_params(which='minor', length=5)  # Minor tick length

#marker to show the L1 point
r_marker = 326031897.53
plt.axvline(x=r_marker, color='g', linestyle='--', alpha=0.7, label=f'r = {r_marker:.2e} m')
plt.plot(r_marker, grav_func(r_marker), 'go', markersize=10, label=f'L1 point at r = {r_marker:.2e} m')


r_guess = 3.3e8
plt.axvline(x=r_guess, color='y', linestyle='--', alpha=0.7, label=f'r = {r_guess:.2e} m')
plt.plot(r_guess, grav_func(r_marker), 'yo', markersize=10, label=f'r guess at r = {r_guess:.2e} m')


plt.legend()
plt.show()

#I will now implement my 2 r guess using the Secant Method new guess form: x3 = x2 - f(x2) * (x2-x1)/(f(x2)-f(x1)). Where x is replace with r. 

def secant_method(func, r1, r2, tolerance=1e-6, max_iterations=1000):  #tolerance: the pre-defined small error threshold that determines when the iterative process should stop because the approximations are sufficiently close to the true root. 

    for i in range(max_iterations):
        f1 = func(r1)
        f2 = func(r2)

        if abs(f2-f1) < 1e-10:
            return r2

        #Implement the secant method here: 
        r3 = r2 - f2 * (r2 - r1) / (f2 - f1)
    
        if abs(r3 - r2) < tolerance:
            print(f"\nConverged after {i+1} iterations")
            return r3


        r1 = r2
        r2 = r3

    return r2

r0 = 3.1e8 
r1 = 3.3e8  

root = secant_method(grav_func, r0, r1)
print(f"\nFinal root: r = {root:.10f} m")


#Plotting the function against r to estimate the best guess for r value. 
#We want to make a guess an r value based on when it intercepts the 0 axis. 

r = np.linspace(1e8, 3.8e8, 1000) #R = 3.844e8 is my limit since that is the Earth-Moon distance. r can not equal 0 since you cannot divide by 0! (Related to the grav_func) 
plt.figure(figsize=(10, 6))
plt.plot(r, grav_func(r))
plt.xlabel('r (distance)')
plt.ylabel('Gravitational function')
plt.title('Gravitational Function vs Distance [Secant]')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Zero line')

# Add tick marks to help determine guess r value. 
plt.minorticks_on()  # This adds minor tick marks
plt.tick_params(which='both', width=1)  
plt.tick_params(which='major', length=8)  # Major tick length
plt.tick_params(which='minor', length=5)  # Minor tick length

#marker to show the L1 point
r_marker = 326031897.53
r_1 = 3.1e8 
r_2 = 3.3e8  

plt.axvline(x=r_marker, color='g', linestyle='--', alpha=0.7, label=f'r = {r_marker:.2e} m')
plt.plot(r_marker, grav_func(r_marker), 'go', markersize=10, label=f'L1 point at r = {r_marker:.2e} m')




plt.axvline(x=r_1, color='y', linestyle='--', alpha=0.7, label=f'r = {r_1:.2e} m')
plt.plot(r_1, grav_func(r_1), 'yo', markersize=10, label=f'r1 guess r = {r_1:.2e} m')

plt.axvline(x=r_2, color='b', linestyle='--', alpha=0.7, label=f'r = {r_2:.2e} m')
plt.plot(r_2, grav_func(r_2), 'bo', markersize=10, label=f'r2 guess r = {r_2:.2e} m')


plt.legend()
plt.show()



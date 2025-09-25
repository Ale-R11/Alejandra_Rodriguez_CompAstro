#importing packages needed to run the lines of code 
import numpy as np 
import math 
import matplotlib.pyplot as plt

#HW2-Exercise 5.3: Integration 

#Integration using Simpson's Rule 
#Want to calculate integral of E(x) = integral from 0 to x  e^(-t^2) dt for values of x from 0 to 3 in steps of 0.1. 

#Let's define our integrand function e^(-t^2)
def Eintegrand(t):
    return np.exp(-t**2)
    
#Let's define the function that will allow us to implement the Simpson's Rule
#Simpson's Rule: the integral from a to b f(t) dt is approximately equal to ((b-a)/6) * [f(a) + 4f((a+b)/2) + f(b)]
def simpsonrule(f,a,b,n): # f = the function you're integrating a = starting point of the interval b = the endpoint of the interval, n = number of subintervals 
    if n % 2 != 1: #the number of subintervals has to be even 
        n += 1 #this allows us to ensure n is even 

    h = (b-a) / n #this is the step size 
    x = np.linspace(a,b,n+1) #gives us points 
    y = f(x) #values at those points

    #this part applies the simpson's rule 
    integral_E = y[0] + y[-1] #evaluating the first and last terms 

    for i in range (1, n): 
        if i % 2 == 1: 
            integral_E += 4*y[i]
        else:
            integral_E += 2*y[i]

    integral_E *= h/3
    return integral_E

#this will compute the integral E(x) 
def E_func(x_values, n_intervals=1000):
    
    E_values = []

    for x in x_values:
        if x == 0:
            E_values.append(0.0)
        else:
            integral_E = simpsonrule(Eintegrand, 0, x, n_intervals)
            E_values.append(integral_E)

    return np.array(E_values)
        

x_values = np.arange(0, 3.1, 0.1) #generates x values from 0 to 3 in steps of 0.1
E_values = E_func(x_values, n_intervals=1000) #calculates E(x) using the Simpson's rule 

#this code allows the results to print out in a nice table 
print(f"{'x':>6} | {'E(x)':>10}")
print("-" * 20)
for i, (x, E) in enumerate(zip(x_values, E_values)):
    print(f"{x:6.1f} | {E:10.6f}")

plt.figure(figsize=(10, 6))
plt.plot(x_values, E_values, 'r-', linewidth=2, label='E(x) using Simpson\'s Rule')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Error Function: E(x) = ∫₀ˣ e^(-t²) dt')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 3)
plt.ylim(0, max(E_values) * 1.1)
plt.show()


#Integration using Trapezodial Rule 
#Want to calculate integral of E(x) = integral from 0 to x  e^(-t^2) dt for values of x from 0 to 3 in steps of 0.1. 

#Let's define our integrand function e^(-t^2)
def Eintegrand(t):
    return np.exp(-t**2)
    
#Let's define the function that will allow us to implement the Trapezodial's Rule
#Trapezoidal Rule:(h/2) * [f(x0) + 2*f(x1) + 2*f(x2) + ... + 2*f(xn-1) + f(xn)]
def trapezoidalrule(f,a,b,n): # f = the function you're integrating a = starting point of the interval b = the endpoint of the interval, n = number of subintervals 
    if n % 2 != 1: #the number of subintervals has to be even 
        n += 1 #this allows us to ensure n is even 

    h = (b-a) / n #this is the step size 
    x = np.linspace(a,b,n+1) #gives us points 
    y = f(x) #values at those points
 
    #this part applies the trapezoidal rule 
    integral_E = (y[0] + y[-1]) / 2+np.sum(y[1:-1])
    integral_E *= h
    return integral_E

#this will compute the integral E(x) 
def E_functrap(x_values, n_intervals=1000):
    E_values = []

    for x in x_values: 
        if x == 0:
            E_values.append(0.0)
        else:
            integral_E = trapezoidalrule(Eintegrand, 0, x, n_intervals)
            E_values.append(integral_E)

    return np.array(E_values)

x_values = np.arange(0, 3.1, 0.1) #generates x values from 0 to 3 in steps of 0.1
E_trapvalues = E_functrap(x_values, n_intervals=1000) #calculates E(x) using the trapezoidal rule 

#this code allows the results to print out in a nice table 
print(f"{'x':>6} | {'E(x)':>10}")
print("-" * 20)
for i, (x, E) in enumerate(zip(x_values, E_trapvalues)):
    print(f"{x:6.1f} | {E:10.6f}")


plt.figure(figsize=(10, 6))
plt.plot(x_values, E_trapvalues, 'b-', linewidth=2, label='E(x) using Trapezoidal Rule')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Error Function: E(x) = ∫₀ˣ e^(-t²) dt')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 3)
plt.ylim(0, max(E_values) * 1.1)
plt.show()



#Integration using Scipy integration tool
#Want to calculate integral of E(x) = integral from 0 to x  e^(-t^2) dt for values of x from 0 to 3 in steps of 0.1. 

from scipy.integrate import quad
#Let's define our integrand function e^(-t^2)
def Eintegrand(t):
    return np.exp(-t**2)

def E_scipy(x_values): #calculate E(x) using scipy quad integration 
    E_values = []
    
    for x in x_values:
        if x == 0:
            E_values.append(0.0)
        else:
            integral_E, _ = quad(Eintegrand, 0, x)
            E_values.append(integral_E)
    
    return np.array(E_values)

x_values = np.arange(0, 3.1, 0.1)
E_scipy = E_scipy(x_values)

#this code allows the results to print out in a nice table 
print(f"{'x':>6} | {'E(x)':>10}")
print("-" * 20)
for i, (x, E) in enumerate(zip(x_values, E_scipy)):
    print(f"{x:6.1f} | {E:10.6f}")


plt.figure(figsize=(10, 6))
plt.plot(x_values, E_scipy, 'y-', linewidth=2, label='E(x) using Scipy Quad Integration')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Error Function: E(x) = ∫₀ˣ e^(-t²) dt')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 3)
plt.ylim(0, max(E_values) * 1.1)
plt.show()


#Let's compare: as we can see all methods give us the same output 


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))


ax1.plot(x_values, E_values, 'r-', linewidth=2, label='E(x) using Simpson\'s Rule')
ax1.set_xlabel('x')  # Note: set_xlabel, not xlabel
ax1.set_ylabel('E(x)')  # Note: set_ylabel, not ylabel  
ax1.set_title('Error Function: E(x) = ∫₀ˣ e^(-t²) dt')  # Note: set_title, not title
ax1.grid(True, alpha=0.3)
ax1.legend()




ax2.plot(x_values, E_trapvalues, 'b-', linewidth=2, label='E(x) using Trapezoidal Rule')
ax2.set_xlabel('x')
ax2.set_ylabel('E(x)')
ax2.set_title('Error Function: E(x) = ∫₀ˣ e^(-t²) dt')
ax2.grid(True, alpha=0.3)
ax2.legend()



ax3.plot(x_values, E_scipy, 'y-', linewidth=2, label='E(x) using Scipy Quad Integration')
ax3.set_xlabel('x')
ax3.set_ylabel('E(x)')
ax3.set_title('Error Function: E(x) = ∫₀ˣ e^(-t²) dt')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.show()





#Integration using Simpson's Rule 
#Want to calculate integral of E(x) = integral from 0 to x  e^(-t^8) dt for values of x from 0 to 3 in steps of 0.1. 

#Let's define our integrand function e^(-t^8)
def Eintegrand(t):
    return np.exp(-t**8)
    
#Let's define the function that will allow us to implement the Simpson's Rule
#Simpson's Rule: the integral from a to b f(t) dt is approximately equal to ((b-a)/6) * [f(a) + 4f((a+b)/2) + f(b)]
def simpsonrule(f,a,b,n): # f = the function you're integrating a = starting point of the interval b = the endpoint of the interval, n = number of subintervals 
    if n % 2 != 1: #the number of subintervals has to be even 
        n += 1 #this allows us to ensure n is even 

    h = (b-a) / n #this is the step size 
    x = np.linspace(a,b,n+1) #gives us points 
    y = f(x) #values at those points

    #this part applies the simpson's rule 
    integral_E = y[0] + y[-1] #evaluating the first and last terms 

    for i in range (1, n): 
        if i % 2 == 1: 
            integral_E += 4*y[i]
        else:
            integral_E += 2*y[i]

    integral_E *= h/3
    return integral_E


#this will compute the integral E(x) 
def E_func(x_values, n_intervals=1000):
    
    E_values = []

    for x in x_values:
        if x == 0:
            E_values.append(0.0)
        else:
            integral_E = simpsonrule(Eintegrand, 0, x, n_intervals)
            E_values.append(integral_E)

    return np.array(E_values)

x_values = np.arange(0, 3.1, 0.1) #generates x values from 0 to 3 in steps of 0.1
E_values = E_func(x_values, n_intervals=1000) #calculates E(x) using the Simpson's rule 

#this code allows the results to print out in a nice table 
print(f"{'x':>6} | {'E(x)':>10}")
print("-" * 20)
for i, (x, E) in enumerate(zip(x_values, E_values)):
    print(f"{x:6.1f} | {E:10.6f}")


plt.figure(figsize=(10, 6))
plt.plot(x_values, E_values, 'r-', linewidth=2, label='E(x) using Simpson\'s Rule')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Error Function: E(x) = ∫₀ˣ e^(-t^8) dt')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 3)
plt.ylim(0, max(E_values) * 1.1)
plt.show()


#Integration using Simpson's Rule 
#Want to calculate integral of E(x) = integral from 0 to x  e^(-t^20) dt for values of x from 0 to 3 in steps of 0.1. 

#Let's define our integrand function e^(-t^20)
def Eintegrand(t):
    return np.exp(-t**20)
    
#Let's define the function that will allow us to implement the Simpson's Rule
#Simpson's Rule: the integral from a to b f(t) dt is approximately equal to ((b-a)/6) * [f(a) + 4f((a+b)/2) + f(b)]
def simpsonrule(f,a,b,n): # f = the function you're integrating a = starting point of the interval b = the endpoint of the interval, n = number of subintervals 
    if n % 2 != 1: #the number of subintervals has to be even 
        n += 1 #this allows us to ensure n is even 

    h = (b-a) / n #this is the step size 
    x = np.linspace(a,b,n+1) #gives us points 
    y = f(x) #values at those points

    #this part applies the simpson's rule 
    integral_E = y[0] + y[-1] #evaluating the first and last terms 

    for i in range (1, n): 
        if i % 2 == 1: 
            integral_E += 4*y[i]
        else:
            integral_E += 2*y[i]

    integral_E *= h/3
    return integral_E

#this will compute the integral E(x) 
def E_func(x_values, n_intervals=1000):
    
    E_values = []

    for x in x_values:
        if x == 0:
            E_values.append(0.0)
        else:
            integral_E = simpsonrule(Eintegrand, 0, x, n_intervals)
            E_values.append(integral_E)

    return np.array(E_values)
        

x_values = np.arange(0, 3.1, 0.1) #generates x values from 0 to 3 in steps of 0.1
E_values = E_func(x_values, n_intervals=1000) #calculates E(x) using the Simpson's rule 

#this code allows the results to print out in a nice table 
print(f"{'x':>6} | {'E(x)':>10}")
print("-" * 20)
for i, (x, E) in enumerate(zip(x_values, E_values)):
    print(f"{x:6.1f} | {E:10.6f}")


plt.figure(figsize=(10, 6))
plt.plot(x_values, E_values, 'r-', linewidth=2, label='E(x) using Simpson\'s Rule')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Error Function: E(x) = ∫₀ˣ e^(-t^20) dt')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 3)
plt.ylim(0, max(E_values) * 1.1)
plt.show()

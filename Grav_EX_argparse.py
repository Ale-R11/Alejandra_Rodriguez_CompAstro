#In this exercise we will find the time it takes for an object fall due to Gravity 
import math #import package to set up equation 
#equation for time due to height and gravity : t = sqrt((2*h)/g)) where h represents height of the object and g is the gravitational acceleration constant 
def time_duetogravity(height, gravity=9.8): #gravity is in m/s^2 #height in meters 
    return math.sqrt(2*height/gravity)
#print(time_duetogravity)


#In this section I will be using argparse 

import argparse 
parser = argparse.ArgumentParser(description="Do Something.")
parser.add_argument('h', type=float, help='height of object above ground')
parser.add_argument("--grav", type=float, default=9.8, help='gravitational acceleration')
args = parser.parse_args()
print(f"{(time_duetogravity(args.h, args.grav)):.3f}") #the f "" and :.3f allows me to keep 3 decimal points 


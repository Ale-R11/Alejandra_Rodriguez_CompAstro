#HW 5 - Monte Carlo Radiative Transfer 
#Evaluating the motion of radiation in a star using a Monte Carlo Simulation.
#Goal: Animate a photon's path traveling through a slab. 

#Importing packages 
import numpy as np 
import numpy.random as rand 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from random import random 


mfp = 150 #m mean free path 

frames = 1800
x_pos = [400]
y_pos = [400]
fig = plt.figure(figsize=(6,6))
ax = plt.axes(xlim=(0,1000), ylim=(0,1000))
particle = plt.Circle((0,0), radius=7, fc='c')  # Made radius bigger so you can see it
ax.add_patch(particle)
line, = ax.plot(x_pos, y_pos, c='g', alpha=0.5)  # Fixed syntax

def animate(i):
    pos = particle.center 
    x = pos[0]
    y = pos[1]
    
    # Use full range for random walk in all directions
    angle = np.random.uniform(0, 2*np.pi)
    
    # Step size (adjust as needed)
    step_size = np.random.exponential(mfp)  #Different mfp for Sun 
    x += step_size * np.cos(angle)
    y += step_size * np.sin(angle)
    
    # Keep within bounds
    x = np.clip(x, 0, 1000)
    y = np.clip(y, 0, 1000)
    
    x_pos.append(x)
    y_pos.append(y)
    
    particle.center = (x, y)
    line.set_xdata(x_pos)
    line.set_ydata(y_pos)
    
    return particle, line

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=10)

# Save BEFORE showing
plt.title('Random Walk Animation')
plt.grid(True, alpha=0.3)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
pillow_writer = animation.PillowWriter(fps=30)
anim.save("single_photon_400_400.gif", writer=pillow_writer)
print("Animation saved as 'single_photon_400_400.gif'!")

plt.show()




mfp = 150 #m mean free path

frames = 1800
x_pos = [0]
y_pos = [0]
fig = plt.figure(figsize=(6,6))
ax = plt.axes(xlim=(0,1000), ylim=(0,1000))
particle = plt.Circle((0,0), radius=7, fc='c')  # Made radius bigger so you can see it
ax.add_patch(particle)
line, = ax.plot(x_pos, y_pos, c='r', alpha=0.5)  # Fixed syntax

def animate(i):
    pos = particle.center 
    x = pos[0]
    y = pos[1]
    
    # Use full range for random walk in all directions
    step_size = np.random.exponential(mfp)  #Different mfp for Sun 
    angle = np.random.uniform(0, 2*np.pi)
    
    # Step size (adjust as needed)
    x += step_size * np.cos(angle)
    y += step_size * np.sin(angle)
    
    # Keep within bounds
    x = np.clip(x, 0, 1000)
    y = np.clip(y, 0, 1000)
    
    x_pos.append(x)
    y_pos.append(y)
    
    particle.center = (x, y)
    line.set_xdata(x_pos)
    line.set_ydata(y_pos)
    
    return particle, line

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=10)

# Save BEFORE showing
plt.title('Random Walk Animation')
plt.grid(True, alpha=0.3)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
pillow_writer = animation.PillowWriter(fps=30)
anim.save("single_photon_origin_wcutoff.gif", writer=pillow_writer)
print("Animation saved as 'single_photon_origin_wcutoff.gif'!")


plt.show()




mfp = 150 #m mean free path

frames = 1800
x_pos = [0]
y_pos = [0]
fig = plt.figure(figsize=(6,6))
ax = plt.axes(xlim=(-1000,1000), ylim=(-1000,1000))
particle = plt.Circle((0,0), radius=9, fc='c')  # Made radius bigger so you can see it
ax.add_patch(particle)
line, = ax.plot(x_pos, y_pos, c='k', alpha=0.6)  # Fixed syntax

def animate(i):
    pos = particle.center 
    x = pos[0]
    y = pos[1]

    # Random angle for random walk
    step_size = np.random.exponential(mfp)  #Different mfp for Sun 
    angle = np.random.uniform(0, 2*np.pi)
    
    # Step size (adjust as needed)
    x += step_size * np.cos(angle)
    y += step_size * np.sin(angle)
    
    # Keep within bounds
    x = np.clip(x, -1000, 1000)
    y = np.clip(y, -1000, 1000)
    
    x_pos.append(x)
    y_pos.append(y)
    
    particle.center = (x, y)
    line.set_xdata(x_pos)
    line.set_ydata(y_pos)
    
    return particle, line

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=10)

# Save BEFORE showing
plt.title('Random Walk Animation')
plt.grid(True, alpha=0.3)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
pillow_writer = animation.PillowWriter(fps=30)
anim.save("single_photon_origin.gif", writer=pillow_writer)
print("Animation saved as 'single_photon_origin.gif'!")

plt.show()




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mfp = 150 #m 


# Parameters
frames = 1800
n_photons = 3


# Initialize figure
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(xlim=(-1000, 1000), ylim=(-1000, 1000))
ax.set_title('Multiple Photon Random Walk')
ax.grid(True, alpha=0.3)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')

# Store positions for each photon
all_x_pos = [[0] for _ in range(n_photons)]
all_y_pos = [[0] for _ in range(n_photons)]

# Create particles and trails for each photon
colors = ['cyan', 'red', 'blue', 'green', 'orange', 'purple', 'yellow', 'magenta']
particles = []
lines = []

for i in range(n_photons):
    color = colors[i % len(colors)]
    particle = plt.Circle((0, 0), radius=5, fc=color)
    ax.add_patch(particle)
    particles.append(particle)
    
    line, = ax.plot([0], [0], c=color, alpha=0.5, linewidth=1)
    lines.append(line)
    
def animate(frame):
    for i in range(n_photons):
        # Get current position
        pos = particles[i].center
        x = pos[0]
        y = pos[1]
        
        # Random angle for random walk
        step_size = np.random.exponential(mfp) #Different mfp for Sun 
        angle = np.random.uniform(0, 2*np.pi)
        
        # Update position
        x += step_size * np.cos(angle)
        y += step_size * np.sin(angle)
        
        # Keep within bounds
        x = np.clip(x, -1000, 1000) 
        y = np.clip(y, -1000, 1000) 
        
        # Store new position
        all_x_pos[i].append(x)
        all_y_pos[i].append(y)
        
        # Update particle and trail
        particles[i].center = (x, y)
        lines[i].set_xdata(all_x_pos[i])
        lines[i].set_ydata(all_y_pos[i])
    
    return particles + lines

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1, blit=True)

# Save BEFORE showing
print("Saving animation... this may take a moment...")
pillow_writer = animation.PillowWriter(fps=30)
anim.save("multiple_photons_3.gif", writer=pillow_writer)
print("Animation saved as 'multiple_photons_3.gif'!")

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mfp = 150 #m 


# Parameters
frames = 1800
n_photons = 5


# Initialize figure
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(xlim=(-1000, 1000), ylim=(-1000, 1000))
ax.set_title('Multiple Photon Random Walk')
ax.grid(True, alpha=0.3)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')

# Store positions for each photon
all_x_pos = [[0] for _ in range(n_photons)]
all_y_pos = [[0] for _ in range(n_photons)]

# Create particles and trails for each photon
colors = ['cyan', 'red', 'blue', 'green', 'orange', 'purple', 'yellow', 'magenta']
particles = []
lines = []

for i in range(n_photons):
    color = colors[i % len(colors)]
    particle = plt.Circle((0, 0), radius=5, fc=color)
    ax.add_patch(particle)
    particles.append(particle)
    
    line, = ax.plot([0], [0], c=color, alpha=0.5, linewidth=1)
    lines.append(line)

def animate(frame):
    for i in range(n_photons):
        # Get current position
        pos = particles[i].center
        x = pos[0]
        y = pos[1]
        
        # Random angle for random walk
        step_size = np.random.exponential(mfp) #Different mfp for Sun 
        angle = np.random.uniform(0, 2*np.pi)
        
        # Update position
        x += step_size * np.cos(angle)
        y += step_size * np.sin(angle)
        
        # Keep within bounds
        x = np.clip(x, -1000, 1000) #change to show more 
        y = np.clip(y, -1000, 1000) #change as well 
        
        # Store new position
        all_x_pos[i].append(x)
        all_y_pos[i].append(y)
        
        # Update particle and trail
        particles[i].center = (x, y)
        lines[i].set_xdata(all_x_pos[i])
        lines[i].set_ydata(all_y_pos[i])
    
    return particles + lines

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=10, blit=True)

# Save BEFORE showing
print("Saving animation... this may take a moment...")
pillow_writer = animation.PillowWriter(fps=30)
anim.save("multiple_photons_5.gif", writer=pillow_writer)
print("Animation saved as 'multiple_photons_5.gif'!")

plt.show()




#Attempting the Sun portion 
#define constants 
sigma_T = 6.652e-25 #cm**2
n_e = 2.5e26*np.exp(-((0)/(0.096*696340))) #r = 0 
l = (1/(n_e*sigma_T)) #mean free path equation
print(l)

#for the Sun case I neeed to change the mfp 
mfp = 0.006013229104028863 #mean free path for the Sun 

frames = 1800
x_pos = [100]
y_pos = [100]
fig = plt.figure(figsize=(6,6))
ax = plt.axes(xlim=(-1000,1000), ylim=(-1000,1000))
particle = plt.Circle((0,0), radius=15, fc='c')  # Made radius bigger so you can see it
ax.add_patch(particle)
line, = ax.plot(x_pos, y_pos, c='g', alpha=0.6)  # Fixed syntax

def animate(i):
    pos = particle.center 
    x = pos[0]
    y = pos[1]

    # Random angle for random walk
    step_size = np.random.exponential(mfp) #Needed to CHANGE THIS #Different mfp for Sun 
    angle = np.random.uniform(0, np.pi) #0 to 180 degree 
    
    # Step size (adjust as needed)
    x += step_size * np.cos(angle)
    y += step_size * np.sin(angle)
    
    # Keep within bounds
    x = np.clip(x, -1000, 1000)
    y = np.clip(y, -1000, 1000)
    
    x_pos.append(x)
    y_pos.append(y)
    
    particle.center = (x, y)
    line.set_xdata(x_pos)
    line.set_ydata(y_pos)
    
    return particle, line

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=10)

# Save BEFORE showing
plt.title('Random Walk Animation')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.grid(True, alpha=0.3)
pillow_writer = animation.PillowWriter(fps=30)
anim.save("single_photon_origin_Sun.gif", writer=pillow_writer)
print("Animation saved as 'single_photon_origin_Sun.gif'!")

plt.show()



# Sun simulation parameters
mfp = 0.006013229104028863  # Very small MFP!
frames = 1800

# Start at center of Sun 
start_x, start_y = 0, 0
x_pos = [start_x]
y_pos = [start_y]

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50))  # Adjusted for small MFP

particle = plt.Circle((start_x, start_y), radius=2, fc='yellow', edgecolor='orange', linewidth=2)
ax.add_patch(particle)
line, = ax.plot(x_pos, y_pos, c='orange', alpha=0.6, linewidth=1)

# Add diagnostic text
distance_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Track total distance
total_distance = 0

def animate(i):
    global total_distance
    
    pos = particle.center 
    x = pos[0]
    y = pos[1]
    
    # Random step from exponential distribution
    step_size = np.random.exponential(mfp)
    total_distance += step_size
    
    # Random angle - use full 2π for true random walk
    angle = np.random.uniform(0, np.pi)
    
    # Update position
    x += step_size * np.cos(angle)
    y += step_size * np.sin(angle)
    
    # Keep within bounds (optional - remove if you want photon to escape)
    x = np.clip(x, -50, 50)
    y = np.clip(y, -50, 50)
    
    x_pos.append(x)
    y_pos.append(y)
    
    particle.center = (x, y)
    line.set_xdata(x_pos)
    line.set_ydata(y_pos)
    
    # Update diagnostics
    distance_from_origin = np.sqrt(x**2 + y**2)
    distance_text.set_text(f'Step: {i}/{frames}\n'
                          f'Total distance: {total_distance:.2f} m\n'
                          f'Current radius: {distance_from_origin:.2f} m\n'
                          f'Steps/scatter: {i}')
    
    return particle, line, distance_text

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=10, blit=True)

plt.title('Photon Random Walk in Sun\n(Very small MFP = 0.006 m)')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.grid(True, alpha=0.3)

# Add circle to show scale
sun_core = plt.Circle((0, 0), 25, fill=False, edgecolor='red', 
                      linestyle='--', linewidth=2, alpha=0.3, label='Reference circle')
ax.add_patch(sun_core)
ax.legend()

print("Saving animation...")
pillow_writer = animation.PillowWriter(fps=30)
anim.save("single_photon_Sun_test.gif", writer=pillow_writer)
print("Animation saved!")
plt.show()


#Trying Different Mean Free Paths Values 
mfp = 300 #m mean free path

frames = 1800
x_pos = [0]
y_pos = [0]
fig = plt.figure(figsize=(6,6))
ax = plt.axes(xlim=(-1000,1000), ylim=(-1000,1000))
particle = plt.Circle((0,0), radius=9, fc='c')  # Made radius bigger so you can see it
ax.add_patch(particle)
line, = ax.plot(x_pos, y_pos, c='b', alpha=0.6)  # Fixed syntax

def animate(i):
    pos = particle.center 
    x = pos[0]
    y = pos[1]

    # Random angle for random walk
    step_size = np.random.exponential(mfp)  #Different mfp for Sun 
    angle = np.random.uniform(0, 2*np.pi)
    
    # Step size (adjust as needed)
    x += step_size * np.cos(angle)
    y += step_size * np.sin(angle)
    
    # Keep within bounds
    x = np.clip(x, -1000, 1000)
    y = np.clip(y, -1000, 1000)
    
    x_pos.append(x)
    y_pos.append(y)
    
    particle.center = (x, y)
    line.set_xdata(x_pos)
    line.set_ydata(y_pos)
    
    return particle, line

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=10)

# Save BEFORE showing
plt.title('Random Walk Animation')
plt.grid(True, alpha=0.3)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
pillow_writer = animation.PillowWriter(fps=30)
anim.save("single_photon_mfp_new.gif", writer=pillow_writer)
print("Animation saved as 'single_photon_mfp_new.gif'!")

plt.show()


mfp = 20 #m mean free path

frames = 1800
x_pos = [0]
y_pos = [0]
fig = plt.figure(figsize=(6,6))
ax = plt.axes(xlim=(-1000,1000), ylim=(-1000,1000))
particle = plt.Circle((0,0), radius=9, fc='c')  # Made radius bigger so you can see it
ax.add_patch(particle)
line, = ax.plot(x_pos, y_pos, c='y', alpha=0.6)  # Fixed syntax

def animate(i):
    pos = particle.center 
    x = pos[0]
    y = pos[1]

    # Random angle for random walk
    step_size = np.random.exponential(mfp)  #Different mfp for Sun 
    angle = np.random.uniform(0, 2*np.pi)
    
    # Step size (adjust as needed)
    x += step_size * np.cos(angle)
    y += step_size * np.sin(angle)
    
    # Keep within bounds
    x = np.clip(x, -1000, 1000)
    y = np.clip(y, -1000, 1000)
    
    x_pos.append(x)
    y_pos.append(y)
    
    particle.center = (x, y)
    line.set_xdata(x_pos)
    line.set_ydata(y_pos)
    
    return particle, line

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=10)

# Save BEFORE showing
plt.title('Random Walk Animation')
plt.grid(True, alpha=0.3)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
pillow_writer = animation.PillowWriter(fps=30)
anim.save("single_photon_mfp_new_small.gif", writer=pillow_writer)
print("Animation saved as 'single_photon_mfp_new_small.gif'!")

plt.show()



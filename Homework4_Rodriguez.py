#Homework 4 - Fourier Analysis Time Series Analysis 
#Analyzing eclipsing binaries and their orbital period. 

#importing packages
from astropy.io import fits 
import numpy as np
import matplotlib.pyplot as plt


#Star: TIC 3921749 

#Reading the FITS file 
filename = 'tic0003921749.fits'
hdul = fits.open(filename) 
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

plt.figure(figsize=(12,5))
plt.scatter(times,fluxes,s=1.9)
plt.xlabel('Time (BJD)') #BJD - Barycentric Julia Date 
plt.ylabel('Flux') 
plt.title('TIC 3921749')
plt.show()

#This displays the light curve data of the time vs flux NOT normalized. 
#This matches to the plot given by the TESS website! 


plt.figure(figsize=(12,5))
plt.plot(times,fluxes)
plt.xlabel('Time (BJD)') #BJD - Barycentric Julia Date 
plt.ylabel('Flux') 
plt.title('TIC 3921749')
plt.show()
#Same graph as above just as lines and not points. 

#Let's normalize our flux 
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

#Normalizing flux 
flux_normalized = fluxes / np.median(fluxes)

plt.figure(figsize=(12,5))
plt.scatter(times,flux_normalized,s=1.9)
plt.xlabel('Time (BJD)') #BJD - Barycentric Julia Date 
plt.ylabel('Normalized Flux')
plt.title('TIC 3921749-Normalized')
plt.show()


#Discrete Fourier Transfrom on Flux (y) thus allowing us to retrieve the power spectrum. 
def dft_star(y):
    N = len(y)
    N_real = N//2+1
    count = np.arange(N)
    c = np.zeros(N_real,dtype='complex')
    for k in range(N_real):
        for n in range(N):
            c[k] += y[n]*np.exp(-2j*np.pi*k*n/N) #fourier coefficients 
    return c


mask = (times>=2400)&(times<=2550) #Mask hides (or replaces) the data that meets specific conditions. 
mask

#new variables
times_new = times[mask]
fluxes_new = fluxes[mask]
ferrs_new = ferrs[mask]

c = dft_star(fluxes_new)

plt.figure(figsize=(12,5))
plt.plot(times_new,fluxes_new, linewidth=1)
plt.xlabel('Time (BJD)') #BJD - Barycentric Julia Date 
plt.ylabel('Flux') 
plt.title('TIC 3921749 Fourier Transform')
plt.grid(True, alpha=0.3)
plt.show()

#to get power spectrum you plot the magnitude squared of ck 
abs_c = abs(c)**2 #the plot of absolute values of coefficients is called the power spectrum 
print(abs_c[0:2]) 
k = np.arange(len(c))
abs_c[0]=0

plt.figure(figsize=(10,5))
plt.ylabel('$|c|^2$')
plt.xlabel('k')
plt.title('TIC 3921749 Power Spectrum')
plt.plot(k,abs_c)
plt.show()

mask = (times>=2400)&(times<=2550)
times_new = times[mask]
fluxes_new = fluxes[mask]
ferrs_new = ferrs[mask]

# Compute DFT on the NEW data
c = dft_star(fluxes_new)  # This should give N//2 + 1 coefficients


# Making sure N matches 
N = len(fluxes_new)
N_real = len(c)


c_full = np.zeros(N, dtype='complex')
c_full[:N_real] = c
c_full[N_real:] = np.conj(c[1:N_real-1][::-1])

# Inverse DFT
def inverse_dft(c_full, N):
    y = np.zeros(N, dtype='complex')
    for k in range(N):
        for n in range(N):
            y[n] += c_full[k] * np.exp(2j * np.pi * k * n / N)
    return (y / N).real

inverse_flux = inverse_dft(c_full, N)


plt.figure(figsize=(14, 5))
plt.plot(times_new, inverse_flux, 'r-', linewidth=1)
plt.xlabel('Time (BJD)')
plt.ylabel('Flux')
plt.title('TIC 3921749 Inverse Transform')
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(14, 5))
plt.plot(times_new, fluxes_new, 'b.', markersize=2, alpha=0.5, label='Original Data')
plt.plot(times_new, inverse_flux, 'r-', linewidth=1, label='Inverse DFT')
plt.xlabel('Time (BJD)')
plt.ylabel('Flux')
plt.title('TIC 3921749 Inverse Transform Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

from scipy.interpolate import interp1d

mask = (times>=2400)&(times<=2550)
times_new = times[mask]
fluxes_new = fluxes[mask]

# Find the typical time step
dt = np.diff(times_new)
median_dt = np.median(dt)

# Create evenly spaced time grid
times_uniform = np.arange(times_new[0], times_new[-1], median_dt)

# Interpolate flux
f_interp = interp1d(times_new, fluxes_new, kind='linear', fill_value='extrapolate')
fluxes_uniform = f_interp(times_uniform)

# Plot before and after
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

ax1.scatter(times_new, fluxes_new, s=2, alpha=0.6, label='Original (with gaps)')
ax1.set_ylabel('Flux')
ax1.set_title('Original Data with Missing Points')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.scatter(times_new, fluxes_new, s=2, alpha=0.6, label='Original data')
ax2.plot(times_uniform, fluxes_uniform, 'r-', linewidth=0.5, label='Interpolated')
ax2.set_xlabel('Time (BJD)')
ax2.set_ylabel('Flux')
ax2.set_title('After Linear Interpolation')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#TESTING NEW REGION

#Discrete Fourier Transfrom on Flux (y) thus allowing us to retrieve the power spectrum. 
def dft_star(y):
    N = len(y)
    N_real = N//2+1
    count = np.arange(N)
    c = np.zeros(N_real,dtype='complex')
    for k in range(N_real):
        for n in range(N):
            c[k] += y[n]*np.exp(-2j*np.pi*k*n/N) #fourier coefficients 
    return c
mask = (times>=2490)&(times<=2500) #Mask hides (or replaces) the data that meets specific conditions. 
mask

#new variables
times_new = times[mask]
fluxes_new = fluxes[mask]
ferrs_new = ferrs[mask]

c = dft_star(fluxes_new)

plt.figure(figsize=(12,5))
plt.plot(times_new,fluxes_new, linewidth=1)
plt.xlabel('Time (BJD)') #BJD - Barycentric Julia Date 
plt.ylabel('Flux') 
plt.title('TIC 3921749 Fourier Transform')
plt.grid(True, alpha=0.3)
plt.show()


#to get power spectrum you plot the magnitude squared of ck 
abs_c = abs(c)**2 #the plot of absolute values of coefficients is called the power spectrum 
print(abs_c[0:2]) 
k = np.arange(len(c))
abs_c[0]=0

plt.figure(figsize=(10,5))
plt.ylabel('$|c|^2$')
plt.xlabel('k')
plt.title('TIC 3921749 Power Spectrum')
plt.plot(k,abs_c)
plt.show()

mask = (times>=2490)&(times<=2500)
times_new = times[mask]
fluxes_new = fluxes[mask]
ferrs_new = ferrs[mask]

# Compute DFT on the NEW data
c = dft_star(fluxes_new)  # This should give N//2 + 1 coefficients


# Making sure N matches 
N = len(fluxes_new)
N_real = len(c)


c_full = np.zeros(N, dtype='complex')
c_full[:N_real] = c
c_full[N_real:] = np.conj(c[1:N_real-1][::-1])

# Inverse DFT
def inverse_dft(c_full, N):
    y = np.zeros(N, dtype='complex')
    for k in range(N):
        for n in range(N):
            y[n] += c_full[k] * np.exp(2j * np.pi * k * n / N)
    return (y / N).real

inverse_flux = inverse_dft(c_full, N)

plt.figure(figsize=(14, 5))
plt.plot(times_new, inverse_flux, 'r-', linewidth=1)
plt.xlabel('Time (BJD)')
plt.ylabel('Flux')
plt.title('TIC 3921749 Inverse Transform')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(times_new, fluxes_new, 'b.', markersize=2, alpha=0.5, label='Original Data')
plt.plot(times_new, inverse_flux, 'r-', linewidth=1, label='Inverse DFT')
plt.xlabel('Time (BJD)')
plt.ylabel('Flux')
plt.title('TIC 3921749 Inverse Transform Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#IT WORKS! 


#TESTING NEW REGION PART 2

#Discrete Fourier Transfrom on Flux (y) thus allowing us to retrieve the power spectrum. 
def dft_star(y):
    N = len(y)
    N_real = N//2+1
    count = np.arange(N)
    c = np.zeros(N_real,dtype='complex')
    for k in range(N_real):
        for n in range(N):
            c[k] += y[n]*np.exp(-2j*np.pi*k*n/N) #fourier coefficients 
    return c
mask = (times>=2300)&(times<=2460) #Mask hides (or replaces) the data that meets specific conditions. 
mask

#new variables
times_new = times[mask]
fluxes_new = fluxes[mask]
ferrs_new = ferrs[mask]

c = dft_star(fluxes_new)

plt.figure(figsize=(12,5))
plt.plot(times_new,fluxes_new, linewidth=1)
plt.xlabel('Time (BJD)') #BJD - Barycentric Julia Date 
plt.ylabel('Flux') 
plt.title('TIC 3921749 Fourier Transform')
plt.grid(True, alpha=0.3)
plt.show()

#to get power spectrum you plot the magnitude squared of ck 
abs_c = abs(c)**2 #the plot of absolute values of coefficients is called the power spectrum 
print(abs_c[0:2]) 
k = np.arange(len(c))
abs_c[0]=0

plt.figure(figsize=(10,5))
plt.ylabel('$|c|^2$')
plt.xlabel('k')
plt.title('TIC 3921749 Power Spectrum')
plt.plot(k,abs_c)
plt.show()

mask = (times>=2300)&(times<=2460)
times_new = times[mask]
fluxes_new = fluxes[mask]
ferrs_new = ferrs[mask]

# Compute DFT on the NEW data
c = dft_star(fluxes_new)  # This should give N//2 + 1 coefficients


# Making sure N matches 
N = len(fluxes_new)
N_real = len(c)

c_full = np.zeros(N, dtype='complex')
c_full[:N_real] = c

if N % 2 == 0:
    c_full[N_real:] = np.conj(c[-2:0:-1])
else: 
    c_full[N_real:] = np.conj(c[-1:0:-1])

# Inverse DFT
def inverse_dft(c_full, N):
    y = np.zeros(N, dtype='complex')
    for k in range(N):
        for n in range(N):
            y[n] += c_full[k] * np.exp(2j * np.pi * k * n / N)
    return (y / N).real

inverse_flux = inverse_dft(c_full, N)

plt.figure(figsize=(14, 5))
plt.plot(times_new, inverse_flux, 'r-', linewidth=1)
plt.xlabel('Time (BJD)')
plt.ylabel('Flux')
plt.title('TIC 3921749 Inverse Transform')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(times_new, fluxes_new, 'b.', markersize=2, alpha=0.5, label='Original Data')
plt.plot(times_new, inverse_flux, 'r-', linewidth=1, label='Inverse DFT')
plt.xlabel('Time (BJD)')
plt.ylabel('Flux')
plt.title('TIC 3921749 Inverse Transform Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#IT WORKS! 
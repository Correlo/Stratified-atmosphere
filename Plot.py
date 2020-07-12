import numpy as np
import matplotlib.pyplot as plt

# Read data
Data = np.loadtxt('Amplitudes.txt')

plt.close()
plt.figure(figsize = (12,6))
plt.plot(Data[:,3], Data[:,0], 'r-', label = 'Pressure perturbation')
plt.plot(Data[:,3], Data[:,1], 'g-', label = r'$v/c_s$ perturbation')
plt.plot(Data[:,3], Data[:,2], 'b-', label = 'Density perturbation')
plt.xlim(min(Data[:,3]), max(Data[:,3]))
plt.xlabel('Temperature (ADU)', fontsize = 15)
plt.ylabel('Amplitude', fontsize = 15)
plt.legend(frameon = False, fontsize = 13)
plt.tick_params(axis='both',direction='inout',which='minor',length=3,width=.5,labelsize=12)
plt.tick_params(axis='both',direction='inout',which='major',length=8,width=1,labelsize=12)
plt.savefig('Amplitude.png')
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import csv

data_load = np.loadtxt("ex1d", delimiter=" ")
nx = 64
fft_ex_interval = 100
num_samples = data_load.shape[0]/nx
num_step = (num_samples-1)*fft_ex_interval
dt = 0.0175009
time = np.linspace(0.0,num_step*dt,int(num_samples))
exf  = np.zeros((int(num_samples),int(nx/2)+1))
exf[:,0] = time

xf = data_load[0:nx,0]

ss = (xf[nx-1]-xf[0])/nx #sample spacing

for si in range(0,int(num_samples)):
    yf = scipy.fftpack.fft(data_load[si*nx:(si+1)*nx,1])
    exf[si,1:int(nx/2)+1] = np.abs(yf[0:int(nx/2)])

freq = np.linspace(0.0, 1.0/(2.0*ss), nx/2)
fig, ax = plt.subplots()
#ax.plot(freq, 2.0/nx * exf[si,:])
ax.plot(time,2.0/nx*exf[:,2],label="vpic")
lf = np.exp(0.2/np.sqrt(20)*time)
ax.plot(time,lf*0.012*2.0/nx,label="linear theory",linestyle='--')
plt.yscale('log')
plt.ylim(1e-4,0.2)
plt.xlim(left=0)
plt.xlabel(r'$\omega_pt$')
plt.ylabel(r'$|E_x|(m=1)$')
plt.legend(loc='lower right',title='nx=ny=64,nppc=40')
#plt.show()
plt.savefig('Diocotron-comp.png')
np.savetxt('ex-fft.txt',exf)

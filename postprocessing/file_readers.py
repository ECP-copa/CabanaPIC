import numpy as np
import matplotlib.pyplot as plt

# Returns a numpy array xv[i,j], i indexes particle number
# in the file.  j=0 -> x
# 					 j=1 -> y
# 					 j=2 -> z
# 					 j=3 -> vx
# 					 j=4 -> vy
# 					 j=5 -> vz

def load_particle_data(location,step,numpars):
	filename = location + 'particles_step' + str(step) + '.vtk'
	xv = np.loadtxt(filename,skiprows=6,max_rows=numpars)

	return xv

def get_kinetic_energy(location,step,numpars):
	xv = load_particle_data(location,step,numpars)
	KE = 0.5*np.sum(xv[:,3]**2 + xv[:,4]**2 + xv[:,5]**2)
	return KE

def get_all_kinetic_energies(location,numsteps,numpars):
	KEs = np.zeros(numsteps)
	for i in range(1,numsteps+1):
		KEs[i-1] = get_kinetic_energy(location,i,numpars)
	
	return KEs

def load_field_data(location,step):
	filename = location + 'efield_step' + str(step) + '.vtk'
	E = np.loadtxt(filename,skiprows=6)
	return E

def get_E_energy(location,step):
	E = load_field_data(location,step)
	PE = np.sum(E[:,0]**2 + E[:,1]**2 + E[:,2]**2)
	return PE

def get_all_E_energies(location,numsteps):
	PEs = np.zeros(numsteps)
	for i in range(1,numsteps+1):
		PEs[i-1] = get_E_energy(location,i)
	return PEs

def read_energy_files(location):
	PE_filename = location + 'energies.txt'
	KE_filename = location + 'kenergies.txt'
	PE = np.loadtxt(PE_filename)
	E_energy = PE[:,2]
	t = PE[:,1]
	KE = np.loadtxt(KE_filename)
	KE = KE[:,2]
	return t, E_energy, KE
	
	

#xv = load_particle_data('../RUNS/vis/',1,9600)
#E = load_field_data('../RUNS/vis/',1)

#print(xv.shape)
#print(xv[-1,:])

#print(E.shape)

num_cells = 32
num_par = 3200
dx = 0.5*np.pi/num_cells

t, PE, KE = read_energy_files('../../RUNS/')
PE = PE*dx
#KE = 1.570796*KE/num_par
#KE = 4.908739e-4*KE

TE = KE+PE

e_change = ( np.amax(TE) - np.amin(TE) )/np.mean(TE)
print('Fractional energy change = ' + str(e_change))

plt.figure(1)
plt.semilogy(t,PE)

plt.figure(2)
plt.plot(t,PE,label='Potential')
plt.plot(t,KE,label='Kinetic')
plt.plot(t,KE+PE,label='Total')

plt.legend()

plt.show()

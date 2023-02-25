# write a LAMMPS data file with liquid configuration atoms and tether atoms at lattice sites 

import numpy as np

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

class Atom:
	def __init__(self,atom_id,atom_type,charge,x,y,z):
		self.atom_id = atom_id
		self.atom_type = atom_type
		self.charge = charge
		self.position = np.array([x,y,z])

# tether atom types: {NaSite: 3, ClSite: 4}

atoms = []
boundsLat = [[0,0],[0,0],[0,0]] # box dims, xlo xhi ylo yhi zlo zhiLat
boundsLiq = [[0,0],[0,0],[0,0]]

print("Reading in lattice sites...")
with open('rocksalt_lattice.data','r') as f: lines = f.readlines()
lines = [l.strip('\n') for l in lines]
atomstart = lines.index('Atoms # charge')
atomend = lines.index('Velocities') if 'Velocities' in lines else len(lines)

# read in lattice box bounds 
for line in lines[:atomstart]:
	tokens = purge(line.split(' '))
	if len(tokens) > 0:
		if tokens[-1] == 'xhi':	
			boundsLat[0][0] = float(tokens[0])
			boundsLat[0][1] = float(tokens[1])
		elif tokens[-1] == 'yhi':
			boundsLat[1][0] = float(tokens[0])
			boundsLat[1][1] = float(tokens[1])			
		elif tokens[-1] == 'zhi':
			boundsLat[2][0] = float(tokens[0])
			boundsLat[2][1] = float(tokens[1])
# read in lattice site atoms
for line in lines[atomstart:atomend]:
	tokens = purge(line.split(' '))
	if len(tokens) > 3:
		if int(tokens[1]) == 1:
			thisType = 3 # NaSite
		elif int(tokens[1]) == 2:
			thisType = 4 # ClSite
		atoms.append(Atom(atom_id=int(tokens[0]),
						  atom_type=thisType,
						  charge=float(tokens[2]),
						  x=float(tokens[3]),
						  y=float(tokens[4]), 
						  z=float(tokens[5]))) 

nSites = len(atoms)
print("Reading in scaled-down liquid configuration...")
with open('1.0.ft2short.data','r') as f: lines = f.readlines()
lines = [l.strip('\n') for l in lines]
atomstart = lines.index('Atoms # charge')
atomend = lines.index('Velocities') if 'Velocities' in lines else len(lines)

# read in box bounds 
for line in lines[:atomstart]:
	tokens = purge(line.split(' '))
	if len(tokens) > 0:
		if tokens[-1] == 'xhi':	
			boundsLiq[0][0] = float(tokens[0])
			boundsLiq[0][1] = float(tokens[1])
		elif tokens[-1] == 'yhi':
			boundsLiq[1][0] = float(tokens[0])
			boundsLiq[1][1] = float(tokens[1])			
		elif tokens[-1] == 'zhi':
			boundsLiq[2][0] = float(tokens[0])
			boundsLiq[2][1] = float(tokens[1])
# read in atoms, need to shift positions of liquid atoms to match lattice sites
for line in lines[atomstart:atomend]:
	tokens = purge(line.split(' '))
	if len(tokens) > 3:
		atoms.append(Atom(atom_id=int(tokens[0]) + nSites,
						  atom_type=int(tokens[1]),
						  charge=float(tokens[2]),
						  x=float(tokens[3]) - boundsLiq[0][0], 
						  y=float(tokens[4]) - boundsLiq[1][0],  
						  z=float(tokens[5]) - boundsLiq[2][0])) # shift positions by difference between boundsLat and boundsLiq origins

with open("sq-wells.data",'w') as f:
	f.write("LAMMPS data file with scaled down liquid configuration and tethering sites\n\n")
	f.write(str(len(atoms)) + " atoms\n")
	f.write('4 atom types\n')
	f.write(f"{boundsLat[0][0]} {boundsLat[0][1]} xlo xhi\n")
	f.write(f"{boundsLat[1][0]} {boundsLat[1][1]} ylo yhi\n")
	f.write(f"{boundsLat[2][0]} {boundsLat[2][1]} zlo zhi\n\n")
	f.write('Masses\n\n')
	f.write('1 22.9898\n')
	f.write('2 35.4527\n')
	f.write('3 1\n')
	f.write('4 1\n')
	f.write('\n')
	f.write('Atoms # charge\n\n')
	for a in atoms: f.write(f'{a.atom_id} {a.atom_type} {a.charge} {a.position[0]} {a.position[1]} {a.position[2]}\n')
	f.write('\n')
print('written to sq-wells.data')




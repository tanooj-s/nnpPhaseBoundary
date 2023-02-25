
# PATHWAY
# 1. Deform liquid to solid volume
# 2. original interactions -> scaled down interactions (TI)
# 3. scaled down -> scaled down + tethers (TI)
# 4. scaled down + tethers -> original potential (TI)

import os
import time
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

curr_path = os.getcwd()
lmp = '/home/tanooj/lammps-23Jun2022/build/lmp' # location of lammps executable

plt.rcParams['figure.figsize'] = (7,18)

# short ranged scaling parameter
eta = 0.1
print(f"Short ranged liquid scaling parameter η = {eta} (0 = no interactions between particles)")

# tethering potential parameters (A and B in LAMMPS pair_style gauss)
wellDepth = 2.
invWidth = 1.1
print(f"Tether well depth: {wellDepth} eV")
def gaussWidth(B): return np.sqrt(1/(2*float(B))) if B > 0 else 'Inf'
print(f"Tether well width (estimated from Gaussian inflection point): {gaussWidth(invWidth)} A")

# system size - make sure there are at least ~500 atoms in the system for satisfactorily accurate bulk properties
boxsize = 2 

# temperatures to run pathway at - this window should be estimated from an initial coarse scan using direct interface sims
temps = np.arange(1130,1141,10) # Truns
print("Temperatures ∆G_sl will be evaluated at:")
print(temps)


# how frequently to sample along alchemical pathway
dlambda = 0.02 
# use at least 50 points (dlambda = 0.02) for production calculations and smooth curves
# densely sample points near 0 and 1 (in case of pathological behavior near endpoints)
points0 = [0.,0.005,0.01,0.02]
points1 = [0.98,0.99,0.995,1.]
pointsMid = np.arange(np.max(points0),np.min(points1),dlambda)
lambdas = sorted(list(set(points0) | set(pointsMid) | set(points1)))
lambdas = [round(l,4) for l in lambdas]
print("Sampled λ points:")
print(lambdas)

# choose which steps to run along pathway (all True by default - switch individual ones on/off for debuggins)
initConfigs = True 
step1 = True 
step2 = True      
step3 = True 
step4 = True 



with open("delA-T.csv","w") as f:
	f.write("Trun,dA_deform,dA_2,dA_3,dA_4,dA_TI,dApath,PdV,dGpath,dH,dT,Tmelt\n")
	
	for T in temps:
		print(f" ----- T = {T}K ----- ")
		os.system("rm fumi*id.data deform*data *.ft2short.* *.tetherOn.* *.fumiOnTetherOff.*") # delete files from runs at previous temperatures

		if initConfigs:
			print(f"Creating ambient pressure crystalline configurations at {T} K...")
			seed = np.random.randint(0,1000)
			os.system(f"sbatch create_solid.job {seed} {T} {boxsize}")
			while not os.path.exists(curr_path+'/fumi_solid.data'):
				time.sleep(1)

			print(f"Creating ambient pressure molten configurations at {T} K...")
			seed = np.random.randint(0,1000)
			os.system(f"sbatch create_liquid.job {seed} {T} {boxsize}")
			while not os.path.exists(curr_path+'/fumi_liquid.data'):
				time.sleep(1)

			os.system('python parseLammpsLog.py -i fumi_solid.log -o csv')
			os.system('python parseLammpsLog.py -i fumi_liquid.log -o csv')

		# get equilibrium box dimensions for deformation step
		dfl = pd.read_csv('fumi_liquid.csv')
		dfs = pd.read_csv('fumi_solid.csv')
		nUse = int(0.5*dfl.shape[0]) # same for both
		VSol = np.mean(dfs['Volume'][nUse:])
		VLiq = np.mean(dfl['Volume'][nUse:])

		delX = np.mean(dfs['Lx'][nUse:]) - np.mean(dfl['Lx'][nUse:])
		delY = np.mean(dfs['Ly'][nUse:]) - np.mean(dfl['Ly'][nUse:])
		delZ = np.mean(dfs['Lz'][nUse:]) - np.mean(dfl['Lz'][nUse:])

		print("SOLID MINUS LIQUID DIMENSIONS")
		print(f"delX: {delX}")
		print(f"delY: {delY}")
		print(f"delZ: {delZ}")

		# also log enthalpy delH for debugging
		HLiq = np.mean(dfl['Enthalpy'][nUse:])
		HSol = np.mean(dfs['Enthalpy'][nUse:])
		delH = HSol - HLiq
		print("∆H_sl: " + str(delH))
		print("\n")

		print("Starting tethering potential pathway...")

		# ======= deform liquid to solid volume (dA = -∫PdV) ======== 
		if step1:
			print("1.")
			print("---- LIQUID DEFORMATION TO SOLID VOLUME ----")
			seed = np.random.randint(0,1000)
			os.system(f"sbatch deform_liquid.job {seed} {T} {delX} {delY} {delZ}")
			while not os.path.exists(curr_path+'/deformed_liquid.data'):
				time.sleep(1)

			os.system("python parseLammpsLog.py -i deform_liquid.log -o csv")
			df = pd.read_csv('deform_liquid.csv')

			nC = int(df.shape[0]/3)
			volumes = np.array(df['Volume'][nC:-nC])
			pressures = np.array(df['Press'][nC:-nC])
			nAtoms = df['Atoms'][0]
			P0 = np.mean(df['Press'][:nC])
			P1 = np.mean(df['Press'][-nC:])
			print(f"Pressure at solid volume: {P0} bars")
			print(f"Pressure at liquid volume: {P1} bars")
			
			# P(V) curve for -∫PdV
			coeffs = np.polyfit(x=volumes,y=pressures,deg=2) # quadratic fit, this may be need to be tuned for different materials
			Pfit = np.polyval(coeffs,volumes)
			PdV = np.trapz(x=volumes,y=Pfit)
			PdV *= 6.241509e-7 # conversion factor for [bars][cA] --> [eV]
			PdV /= df['Atoms'][0] # per atom energies
			dA_deformation = -PdV
			print(f"dA = {dA_deformation} eV/atom\n")

			plt.subplot(411)
			datalabel = str(round(dA_deformation,4)) + " eV/atom"
			plt.scatter(x=volumes,y=pressures,label=datalabel)
			plt.plot(volumes,Pfit,lw=5)
			plt.xlabel('Volume (cA)')
			plt.ylabel('Pressure (bars)')
			plt.title("Step 1: Liquid Deformation")
			plt.grid()
			plt.legend()


		# ====== scale down interaction potential ========
		if step2:
			plt.rcParams['figure.figsize'] = (15,24)
			print("2.")
			print("---- SCALING DOWN IONIC INTERACTIONS ----")
			print("NVT")

			# submit jobs
			for l in lambdas:
				if l == 0.:
					startfile = 'deformed_liquid.data'
				else:
					prev_l = lambdas[lambdas.index(l)-1]
					startfile = str(prev_l)+'.ft2short.data'
				seed = np.random.randint(0,1000)
				os.system(f"sbatch ft2short.job {seed} {startfile} {l} {T} {eta}") 
				while not os.path.exists(curr_path+'/'+str(l)+'.ft2short.data'):
					time.sleep(1)

			# pull relevant data from logs
			tdfs = []
			ljs = []
			fumis = []
			gausses = []
			pes = []
			Ps = []
			for l in lambdas:
				os.system(f"python parseLammpsLog.py -i {l}.ft2short.log -o csv")
				df = pd.read_csv(f"{l}.ft2short.csv")
				nAtoms = int(df['Atoms'][0])
				nUse = int(df.shape[0]/2)
				tdf = np.mean(df['v_tdf'][nUse:])
				P = np.mean(df['Press'][nUse:])
				pe = np.mean(df['PotEng'][nUse:])/df['Atoms'][0]
				print(f"STEP 2: Lambda = {l} | TDF = {tdf} | PE = {pe} | Press = {P}")
				tdfs.append(tdf)
				Ps.append(P)
				pes.append(np.mean(df['PotEng'][nUse:])/df['Atoms'][0])
			tdfs = np.array(tdfs)
			dA = np.trapz(x=lambdas,y=tdfs)
			print(f"dA = {dA} eV/atom")
			dA_fumi2short = dA
			print(f"dA stage 2 = {dA_fumi2short} eV/atom\n")
			
			plt.subplot(412)
			datalabel = str(round(dA_fumi2short,4)) + " eV/atom"
			plt.scatter(x=lambdas,y=tdfs,s=10,label=datalabel)
			plt.xlabel('λ')
			plt.ylabel('dU/dλ (eV/atom)')
			plt.title("Step 2: Original potential -> Scaled down potential")
			plt.grid()
			plt.legend()

	
		# ====== switch on tethering potential ======
		if step3:
			# first generate lattice positions for tether sites
			df = pd.read_csv('fumi_solid.csv')
			latConst = np.mean(df['Lx'][nUse:])/boxsize # equilibrium lattice constant at this temperature
			print(f"Rocksalt lattice constant: {latConst} A")
			print("Generating crystal potential sites...")
			os.system(f"{lmp} -i rocksalt_lattice.in -v lc {latConst} -v s {boxsize} -sc none") 
			os.system("python addWellAtoms.py") # write datafile

			print("3.")
			print("---- SWITCHING ON TETHERING POTENTIAL ----")
			print("NVT")

			# submit jobs
			for l in lambdas:
				if l == 0.:
					startfile = 'sq-wells.data'
				else:
					prev_l = lambdas[lambdas.index(l)-1]
					startfile = str(prev_l)+'.tetherOn.data'
				seed = np.random.randint(0,1000)
				os.system(f"sbatch tetherOn.job {seed} {startfile} {l} {T} {eta} {wellDepth} {invWidth}") 	
				while not os.path.exists(curr_path+'/'+str(l)+'.tetherOn.data'):
					time.sleep(2)

			# pull relevant data from logs
			tdfs = []
			ljs = []
			fumis = []
			gausses = []
			pes = []
			Ps = []
			for l in lambdas:
				os.system(f"python parseLammpsLog.py -i {l}.tetherOn.log -o csv")
				df = pd.read_csv(f"{l}.tetherOn.csv")
				nUse = int(df.shape[0]/2)
				fumi = np.mean(df['c_fumi'][nUse:])
				tether = np.mean(df['c_gauss'][nUse:])
				tdf = np.mean(df['v_tdf'][nUse:])
				P = np.mean(df['Press'][nUse:])
				pe = np.mean(df['PotEng'][nUse:])/df['v_nReal'][0]
				print(f"STEP 3: Lambda = {l} | TDF = {tdf} | PE = {pe} | Fumi = {fumi} | Tether = {tether} | Press = {P}")
				tdfs.append(tdf)
				Ps.append(P)
				pes.append(np.mean(df['PotEng'][nUse:])/df['v_nReal'][0])
			tdfs = np.array(tdfs)
			dA = np.trapz(x=lambdas,y=tdfs)
			print(f"dA = {dA} eV/atom")
			dA_short2tethers = dA 
			print(f"dA stage 3 = {dA_short2tethers} eV/atom\n")

			plt.subplot(413)
			datalabel = str(round(dA_short2tethers,4)) + " eV/atom"
			plt.scatter(x=lambdas,y=tdfs,s=10,label=datalabel)
			plt.xlabel('λ')
			plt.ylabel('dU/dλ (eV/atom)')
			plt.title("Step 3: Tethering potential switched on")
			plt.grid()
			plt.legend()


		# ====== restore original potential, scale down tethering potential ======
		if step4:
			print("4.")
			print("---- RESTORING ORIGINAL INTERACTIONS, SWITCHING TETHERS OFF ----")
			print("NVT")

			# submit jobs
			for l in lambdas:
				if l == 0.:
					startfile = '1.0.tetherOn.data'
				else:
					prev_l = lambdas[lambdas.index(l)-1]
					startfile = str(prev_l)+'.fumiOnTetherOff.data'
				seed = np.random.randint(0,1000)
				os.system(f"sbatch fumiOnTetherOff.job {seed} {startfile} {l} {T} {eta} {wellDepth} {invWidth}")
				while not os.path.exists(curr_path+'/'+str(l)+'.fumiOnTetherOff.data'):
					time.sleep(2)

			# pull relevant data from logs
			tdfs = []
			ljs = []
			fumis = []
			gausses = []
			pes = []
			Ps = []
			for l in lambdas:
				os.system(f"python parseLammpsLog.py -i {l}.fumiOnTetherOff.log -o csv")
				time.sleep(2)
				df = pd.read_csv(f"{l}.fumiOnTetherOff.csv")
				nUse = int(df.shape[0]/2)
				fumi = np.mean(df['c_fumi'][nUse:])
				tether = np.mean(df['c_gauss'][nUse:])
				tdf = np.mean(df['v_tdf'][nUse:])
				P = np.mean(df['Press'][nUse:])
				pe = np.mean(df['PotEng'][nUse:])/df['v_nReal'][0]
				print(f"STEP 4: Lambda = {l} | TDF = {tdf} | PE = {pe} | Fumi = {fumi} | Tether = {tether} | Press = {P}")
				tdfs.append(tdf)
				Ps.append(P)
				pes.append(np.mean(df['PotEng'][nUse:])/df['v_nReal'][0])
			tdfs = np.array(tdfs)
			dA = np.trapz(x=lambdas,y=tdfs)
			print(f"dA = {dA} eV/atom")
			dA_fumiOn = dA 
			print(f"dA stage 4 = {dA_fumiOn} eV/atom\n")
			

			plt.subplot(414)
			datalabel = str(round(dA_fumiOn,4)) + " eV/atom"
			plt.scatter(x=lambdas,y=tdfs,s=10,label=datalabel)
			plt.xlabel('λ')
			plt.ylabel('dU/dλ (eV/atom)')
			plt.title("Step 4: Tethers switched off, ionic interactions scaled back up")
			plt.grid()
			plt.legend()
			

			# ===== total Helmholtz free energy difference along pathway ==== 
			dA_pathway = dA_fumiOn + dA_short2tethers + dA_fumi2short + dA_deformation
			dA_TI = dA_fumiOn + dA_short2tethers + dA_fumi2short


		print(f"------ ∆A along pathway: {dA_pathway} eV/atom !!!!")
		print(f"TI ∆A: {dA_TI} eV/atom")
		print(f"Deformation ∆A: {dA_deformation} eV/atom")	

		df = pd.read_csv('fumi_solid.csv')
		pdv = (1e5*((VSol-VLiq)*1e-30))*6.241509e18/np.mean(df['Atoms'][0])
		print(f"P∆V between phases at ambient pressure: {pdv} eV/atom ")

		# net Gibbs free energy difference along pathway
		# ∆G = ∆A + ∆(PV) = ∆A + V∆P + P∆V = ∆A + P∆V (ambient pressure, ∆P = 0)
		dG_pathway = dA_pathway + pdv
		print(f"=> ∆G = ∆A + P∆V = {dG_pathway} eV/atom ")
		f.write(f"{T},{dA_deformation},{dA_fumi2short},{dA_short2tethers},{dA_fumiOn},{dA_TI},{dA_pathway},{pdv},{dG_pathway},{delH}\n")

		figname = "T-"+str(T)+"K_curves.png"
		plt.savefig(figname)
		plt.clf()
		time.sleep(10)


exit()



# script to numerically integrate Clausius-Clapeyron equation using MD
# start from a single known coexistence point, iteratively run simulations to converge upon coexistence points at higher temperatures and pressures
# for each point 
# - initially predict a straight line from the known point
# - run the thermodynamic cycle, calculate ∆G_sl
# - iteratively correct until predicted melting temperature converges

import numpy as np
import pandas as pd
import os
import glob
import argparse
import time

parser = argparse.ArgumentParser(description="predict coexisting phase points from known point")
parser.add_argument('-T0', action="store", dest="T0") # known coexistence temp 
parser.add_argument('-P0', action="store", dest="P0") # known coexistence pressure, should be in bars
parser.add_argument('-s', action="store", dest="size") # system size, in terms of number of unit cells along each axis
parser.add_argument('-t', action="store", dest="threshold") # temperature convergence threshold
parser.add_argument('-dP', action="store", dest="dP") # pressure step size, interval of phase boundary predictions
parser.add_argument('-c', action="store", dest="n_calc") # number of points to calculate along phase boundary  
parser.add_argument('-pot', action="store", dest="pot") # location of NNP file
args = parser.parse_args()


T0 = float(args.T0)
P0 = float(args.P0)
size = int(args.size)
dP = float(args.dP)
threshold = float(args.threshold)
n_calc = float(args.n_calc)
pot = args.pot
curr_path = os.getcwd()

def moving_average(x, w): return np.convolve(x, np.ones(w), 'valid')/w

point_counter = 1
with open('clausius-predictions.csv','w') as f:
    f.write("T (K),P (bar),n_iter\n") # include how many iterations it took to converge at each pressure
    

    print(f"Predicting melting temperature every {dP} bars")

    curr_T = int(args.T0) # kelvin
    curr_P = int(args.P0) # bars

    while point_counter <= n_calc:
        f.write(f"{curr_T},{curr_P},{point_counter}\n")
        print(f"Creating solid and liquid configurations at {curr_T} K, {curr_P} bars...")
        os.system(f"sbatch create_solid.job {pot} {curr_P} {curr_T} {size} {np.random.randint(1,1000)}")
        os.system(f"sbatch create_liquid.job {pot} {curr_P} {curr_T} {size} {np.random.randint(1,1000)}")
        
        while not (os.path.exists(curr_path+'/create_solid.data') and os.path.exists(curr_path+'/create_liquid.data')): 
            time.sleep(1)

        os.system("python parseLammpsLog.py -i create_solid.log -o csv")
        os.system("python parseLammpsLog.py -i create_liquid.log -o csv")

        print(f"Coexisting state point: ({curr_T} K, {curr_P} bars)")

        solid, liquid = pd.read_csv("create_solid.csv"), pd.read_csv("create_liquid.csv")
        delH = np.mean(solid['Enthalpy']) - np.mean(liquid['Enthalpy'])
        delV = np.mean(solid['Volume']) - np.mean(liquid['Volume'])
        print(f"ΔH_sl (eV) = {delH} eV")
        print(f"ΔV_sl (cA) = {delV} cubic angstroms") # needed for initial straight line prediction
        # convert relevant quantities to SI for slope
        delH = delH*1.602177e-19 # eV to joule
        delV = delV*1e-30 # cubic angstrom to m3

        slope = curr_T*delV/delH
        print(f"dT/dP = TΔH/ΔV = {slope} K/bar\n") 
        print("(T0, P0) and dT/dP => straight line approximation to phase boundary")
        delH = delH/1.602177e-19 # convert back to eV for later temperature correction
        delP = dP # from args
        pred_P = curr_P + delP # next P
        pred_T = curr_T + slope*delP*1e5 # intiial straight line prediction for next melting point
        print(f"Initial prediction for next coexisting state point: ({pred_T} K, {pred_P} bars)")



        # ======== compression simulations ========
        # these only need to be run once for each dP
        print(f"Compressing liquid and solid phases")
        print(f"{curr_P} bars --> {pred_P} bars, at {curr_T} K")

        os.system(f"sbatch liquid_compress.job {pot} {curr_P} {pred_P} {curr_T} {np.random.randint(1,1000)}")
        os.system(f"sbatch solid_compress.job {pot} {curr_P} {pred_P} {curr_T} {np.random.randint(1,1000)}")

        while not (os.path.exists(curr_path+'/solid_compress.data') and os.path.exists(curr_path+'/liquid_compress.data')): 
            time.sleep(1)

        os.system("python parseLammpsLog.py -i liquid_compress.log -o csv")
        os.system("python parseLammpsLog.py -i solid_compress.log -o csv")
        scdf, lcdf = pd.read_csv('solid_compress.csv'), pd.read_csv('liquid_compress.csv')
        print("NOTE: ∆G = ∫VdP is being approximated as V∆P here with Vbar over the course of the compression")
        delG_sc = (np.mean(scdf['Volume'])*1e-30*delP*1e5)/(1.602177e-19)
        delG_lc = (np.mean(lcdf['Volume'])*1e-30*delP*1e5)/(1.602177e-19)
        print(f"∆G solid compression = {delG_sc} eV")
        print(f"∆G liquid compression = {delG_lc} eV")
        delG_comp = delG_sc - delG_lc
        print(f" => ∆G_sl_comp = {delG_comp} eV\n")


        # ======== heating simulations ========
        # these need to be run till convergence
        delT = 1000 # initially set arbitrarily high 
        heat_counter = 0
        while np.abs(delT) > threshold:
            if heat_counter >= 10: continue # don't get stuck oscillating about some value
            if delT > 1000: break # diverging predictions

            print(f"Heating iteration: {heat_counter}")
            print(f"Heating liquid and solid phases")
            print(f"{curr_T} K --> {pred_T} K, at {pred_P} bars")
            os.system(f"sbatch liquid_heatup.job {pot} {pred_P} {curr_T} {pred_T} {np.random.randint(1,1000)}")
            os.system(f"sbatch solid_heatup.job {pot} {pred_P} {curr_T} {pred_T} {np.random.randint(1,1000)}")
            
            while not (os.path.exists(curr_path+'/solid_heatup.data') and os.path.exists(curr_path+'/liquid_heatup.data')): 
                time.sleep(1)

            os.system("python parseLammpsLog.py -i liquid_heatup.log -o csv")
            os.system("python parseLammpsLog.py -i solid_heatup.log -o csv")

            # -(H/T)dt integration
            shdf, lhdf = pd.read_csv('solid_heatup.csv'), pd.read_csv('liquid_heatup.csv') 
            h_sl = shdf['Enthalpy'] - lhdf['Enthalpy'] # delH between phases across the heating sims
            T_int = np.linspace(curr_T,pred_T,h_sl.shape[0])
            delG_heat = np.trapz(y=(-h_sl/T_int),x=T_int)
            print(f" => ∆G_sl_heat = -∫(Hs-Hl)dT/T = {delG_heat} eV\n")

            delG = delG_comp + delG_heat
            print("*** ∆G_SL: " + str(delG) + " eV ***\n\n")

            # make correction for temperature, same approximation based on latent heat as NNP cycle 
            delS = delH/pred_T
            delT = -delG/delS
            print(f"∆T = {delT} K")
            corrected_T = pred_T + 0.75*delT # with mixing fraction factor 
            print(f"Updated prediction for coexistence point: ({corrected_T} K, {pred_P} bars)\n")
            
            pred_T = corrected_T # now go back and calculate ∆G_sl at this point again
            heat_counter += 1

            # delete generated files to not mix up with next iteration
            os.system("rm *heatup.data *heatup.log *heatup.csv")

        print(f"Converged prediction for coexistence point: ({corrected_T} K, {pred_P} bars)\n")

        curr_P = pred_P
        curr_T = corrected_T
        point_counter += 1

        # delete generated files to not mix up with next iteration
        os.system("rm *csv *data *log")
        

# script that predicts a single solid-liquid coexistence point of an NNP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import glob
import argparse

parser = argparse.ArgumentParser(description="predict coexisting phase points from known point")
parser.add_argument('-T0', action="store", dest="T0") # known Fumi-Tosi melting temperature at ambient pressure
parser.add_argument('-Tg', action="store", dest="Tg") # initial guess for NNP melting temperature at same pressure
parser.add_argument('-pot', action="store", dest="pot") # location of NNP file 
parser.add_argument('-t', action="store", dest="threshold") # temperature convergence threshold
parser.add_argument('-s', action="store", dest="size") # system size

args = parser.parse_args()

T0 = float(args.T0)
P0 = 1 # bars, assuming we want the melting point at ambient pressure
Tg = float(args.Tg)
threshold = float(args.threshold)
pot = args.pot
size = int(args.size) 

curr_path = os.getcwd()

def moving_average(x, w): return np.convolve(x, np.ones(w), 'valid')/w

os.system("mkdir solid liquid") # directories to hold separate TI sim log files for each phase

# first initialize Fumi-Tosi configurations at that potential's known melting point
print(f"Creating Fumi solid and liquid configurations at {T0} K, {P0} bars...")
os.system(f"sbatch create_fumi_solid.job {T0} {P0} {size} {np.random.randint(1,1000)}")
os.system(f"sbatch create_fumi_liquid.job {T0} {P0} {size} {np.random.randint(1,1000)}")

while not (os.path.exists(curr_path+'/fumi_solid.data') and os.path.exists(curr_path+'/fumi_liquid.data')):
    time.sleep(1)

os.system("python parseLammpsLog.py -i fumi_solid.log -o csv")
os.system("python parseLammpsLog.py -i fumi_liquid.log -o csv")
solid_fumi, liquid_fumi = pd.read_csv("fumi_solid.csv"), pd.read_csv("fumi_liquid.csv")

prediction_filename = "nnp-predictions.csv"
with open(prediction_filename,'w') as f: f.write("Temperature (K)\n")
delT = 1000 # set arbitrarily high at first
counter = 0


# predictor-corrector run
while np.abs(delT) > threshold:

    # create NNP configurations at guessed point
    print(f"Creating NNP solid and liquid configurations at {Tg} K, {P0} bars...")
    os.system(f"sbatch create_nn_solid.job {Tg} {P0} {size} {pot} {np.random.randint(1,1000)}")
    os.system(f"sbatch create_nn_liquid.job {Tg} {P0} {size} {pot} {np.random.randint(1,1000)}")

    while not (os.path.exists(curr_path+'/nn_solid.data') and os.path.exists(curr_path+'/nn_liquid.data')):
        time.sleep(1)

    os.system("python parseLammpsLog.py -i nn_solid.log -o csv")
    os.system("python parseLammpsLog.py -i nn_liquid.log -o csv")
    solid_nn, liquid_nn = pd.read_csv("nn_solid.csv"), pd.read_csv("nn_liquid.csv")


    # ========= THERMODYNAMIC INTEGRATION ========
    # NVT simulations - alchemical transformation at constant volume takes system to another state point
    # (Tg, P0) ---> (Tg, ???)
    print("\n=x=x=x=x= THERMODYNAMIC INTEGRATION SIMULATIONS =x=x=x=x=")
    print(f"NNP --> Fumi-Tosi at NNP {Tg} K, {P0} bars")
    lambdas = np.arange(0.,1.1,0.5) 
    # only need 3 lambda points since these dU/dlambda curves are linear (so long as NNP and pairwise structures are similar)
    # these can be launched in parallel
    for l in lambdas: 
        os.system(f"sbatch ti_solid.job {l} {Tg} {pot} {np.random.randint(1,1000)}")
        os.system(f"sbatch ti_liquid.job {l} {Tg} {pot} {np.random.randint(1,1000)}")
    
    # sleep till TI sims are done
    while not (os.path.exists(curr_path+'/solid/ti-0.0.data') and os.path.exists(curr_path+'/liquid/ti-0.0.data') and os.path.exists(curr_path+'/solid/ti-0.5.data') and os.path.exists(curr_path+'/liquid/ti-0.5.data') and os.path.exists(curr_path+'/solid/ti-1.0.data') and os.path.exists(curr_path+'/liquid/ti-1.0.data')):
        time.sleep(1) 
    
    # parse log files
    for filename in glob.iglob("solid/*log"): os.system(f"python parseLammpsLog.py -i {filename} -o csv")
    for filename in glob.iglob("liquid/*log"): os.system(f"python parseLammpsLog.py -i {filename} -o csv")

    # read in relevant data from each TI file
    solid_tdfs, liquid_tdfs = [], []
    for l in lambdas:
        s_df, l_df = pd.read_csv(f'solid/ti-{l}.csv'), pd.read_csv(f'liquid/ti-{l}.csv')
        nUse = int(0.5*len(s_df['Step'])) # use last half for equilibrated values
        solid_tdfs.append(np.mean(s_df['v_dUdl'][nUse:]))
        liquid_tdfs.append(np.mean(l_df['v_dUdl'][nUse:]))

    idx = lambdas.argsort() # overly verbose but ensure values are sorted
    solid_tdfs, liquid_tdfs = np.array(solid_tdfs)[idx], np.array(liquid_tdfs)[idx]
    # integrate under each curve to get ∆A for each phase 
    delA_TI_S, delA_TI_L = np.trapz(y=solid_tdfs, x=lambdas), np.trapz(y=liquid_tdfs, x=lambdas)
    print("Solid ∆A (TI) = " + str(delA_TI_S) + " eV")
    print("Liquid ∆A (TI) = " + str(delA_TI_L) + " eV")


    # --- V∆P terms for net ∆G_TI ----
    # solid V∆P
    df = pd.read_csv('solid/ti-1.0.csv')
    nUse = int(0.5*len(df['Press'])) 
    P1 = np.mean(df['Press'][nUse:])
    print(f"Solid: {P0} ----> {P1}")
    delP = P1 - P0
    solid_comp_P1 = P1 # needed for next phase of (de)compression sims
    VdelP = (df['Volume'][0]*1e-30)*(delP*1e5)/(1.602177e-19)
    delG = delA_TI_S + VdelP
    print(f"Solid V∆P = {VdelP} eV")
    print(f"Solid ∆G_TI = ∆A + V∆P = {delG} eV")
    delG1 = delG 
    # liquid V∆P 
    df = pd.read_csv('liquid/ti-1.0.csv')
    nUse = int(0.5*len(df['Press'])) 
    P1 = np.mean(df['Press'][nUse:])
    print(f"Liquid: {P0} ----> {P1}")
    delP = P1 - P0
    liquid_comp_P1 = P1 # needed for next phase of (de)compression sims
    VdelP = (df['Volume'][0]*1e-30)*(delP*1e5)/(1.602177e-19)
    delG = delA_TI_L + VdelP
    print(f"Liquid V∆P = {VdelP} eV")
    print(f"Liquid ∆G_TI = ∆A + V∆P = {delG} eV")
    delG4 = delG 

    delG_TI = delG1 - delG4 # solid delG_TI minus liquid delG_TI
    print("--∆G_TI = " + str(delG_TI) + " eV--\n")




    # ========= (DE)COMPRESSION SIMS FOR ∫VdP =========
    # take (now wholly Fumi-Tosi) system back to ambient pressure
    # (Tg, ??? bars) --> (Tg, P0 bars)
    # read in ??? pressures from the λ = 1 simulations of previous step 
    print("\n=x=x=x=x= (DE)COMPRESSION SIMULATIONS =x=x=x=x=")
    print("Confirm starting pressures for (de)compression simulations are the same as ending pressures from TI run")
    os.system(f"sbatch solid_press.job {Tg} {solid_comp_P1} {P0} {np.random.randint(1,1000)}")
    os.system(f"sbatch liquid_press.job {Tg} {liquid_comp_P1} {P0} {np.random.randint(1,1000)}")

    while not (os.path.exists(curr_path+'/solid_press.data') and os.path.exists(curr_path+'/liquid_press.data')):
        time.sleep(2)

    os.system("python parseLammpsLog.py -i solid_press.log -o csv")
    os.system("python parseLammpsLog.py -i liquid_press.log -o csv")
    time.sleep(2)
    scdf, lcdf = pd.read_csv('solid_press.csv'), pd.read_csv('liquid_press.csv')

    print("NOTE: ∆G = ∫VdP is being approximated as V∆P here with Vbar over the course of the compression")
    # solid
    print(f"Solid: {solid_comp_P1} bars --> {P0} bars, at {Tg}K")
    delP = P0 - solid_comp_P1
    delG = np.mean(scdf['Volume'])*1e-30*delP*1e5/(1.602177e-19)
    print(f"Solid ∆G_C = V∆P = {delG} eV")
    delG2 = delG
    # liquid
    print(f"Liquid: {liquid_comp_P1} bars --> {P0} bars, at {Tg}K")
    delP = P0 - liquid_comp_P1
    delG = np.mean(lcdf['Volume'])*1e-30*delP*1e5/(1.602177e-19)
    print(f"Liquid ∆G_C = V∆P = {delG} eV")
    delG5 = delG

    delG_C = delG2 - delG5 # solid delG_C minus liquid delG_C
    print(f"--∆G_C = {delG_C} eV--\n")




    # ========= HEATING (COOLING) SIMS FOR ∫SdT =========
    # (P0, Tg) --> (P0, T0) 
    # take Fumi-Tosi NaCl back to its known melting point where ∆Gsl=0 by definition
    
    print("\n=x=x=x=x= HEATING (COOLING) SIMULATIONS =x=x=x=x=")
    print("∆G = -∫SdT")
    print("Using difference in enthalpies between phases to approximate difference in entropy")
    print("∆S = (Hs - Hl)/T\n")
    print(f"{Tg}K --> {T0}K for each phase, at {P0} bars")
    os.system(f"sbatch solid_heat.job {Tg} {T0} {P0} {np.random.randint(1,1000)}")
    os.system(f"sbatch liquid_heat.job {Tg} {T0} {P0} {np.random.randint(1,1000)}")

    # sleep till sims are done
    while not (os.path.exists(curr_path+'/solid_heat.data') and os.path.exists(curr_path+'/liquid_heat.data')):
        time.sleep(1)

    os.system("python parseLammpsLog.py -i solid_heat.log -o csv")
    os.system("python parseLammpsLog.py -i liquid_heat.log -o csv")
    shdf, lhdf = pd.read_csv('solid_heat.csv'), pd.read_csv('liquid_heat.csv')

    # = integrate -(Hs-Hl)/T dT
    delH = shdf['Enthalpy'] - lhdf['Enthalpy']
    delH = moving_average(delH, 100) # moving average to smooth out noise
    T_int = np.linspace(Tg, T0, delH.shape[0]) # x points for integral
    delG_H = np.trapz(y=(-delH/T_int),x=T_int)
    print(f"--∆G_H = -S∆T = {delG_H} eV--\n")


    # ==== net free energy difference between phases along pathway ====
    delG_sl = delG_TI + delG_C + delG_H
    print(f"---∆G = ∆G_TI + ∆G_C + ∆G_H = {delG_sl} eV---\n")


    # ==== obtain ∆T correction factor, update Tg ====
    print("Using enthalpy difference between phases for Fumi at known melting point (i.e. latent heat) to obtain ∆T correction at Tg")
    print("∆T correction = ∆G_sl/∆S at Tg")
    print("∆S = ∆H0/Tg (latent heat divided by guessed temperature)")
    delH = np.mean(solid_fumi['Enthalpy']) - np.mean(liquid_fumi['Enthalpy'])
    delS = delH/Tg 
    delT = -delG_sl/delS
    print(f"Fumi-Tosi ∆H0 at ({T0}, {P0}) = {delH} eV => ∆S = {delS} eV/K for NNP at {Tg} K")
    print(f"=> ∆T correction: {delT} K\n")
    predicted_T = Tg + 0.75*delT # with mixing fraction factor
    print(f"Predicted NNP coexistence point: {P0} bars, {predicted_T} K")
    Tg = predicted_T
    counter += 1

    with open(prediction_filename,'a') as f: f.write(f"{predicted_T}\n")

    if counter >= 10: exit() # don't get stuck oscillating around some value; use a higher threshold
    if delT > 1000: exit() # diverging predictions

    # delete all generated files here so that different iterations don't get mixed up 
    os.system('rm fumi_solid* fumi_liquid*')
    os.system('rm nn_solid* nn_liquid*')
    os.system('rm solid/ti* liquid/ti*')
    os.system('rm *id_press.log *id_press.data *id_press.csv')
    os.system('rm *id_heat.log *id_heat.data *id_heat.csv')


# pass converged Tmelt into Clausius-Clapeyron script 






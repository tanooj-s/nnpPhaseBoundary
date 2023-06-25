This repo contains relevant scripts for [First-principles molten salt phase diagrams through thermodynamic integration](https://arxiv.org/abs/2306.02406).


These scripts enable a workflow to predict the solid-liquid phase boundary of arbitrary interatomic force fields (including machine-learned ones) in the pressure-temperature space using molecular dynamics simulations.
With machine-learned force fields, this can allow one to extrapolate the effects of the approximations made by electronic structure methods (such as choice of exchange-correlation functional) on bulk properties. 
The methodology here was prototyped and tested using NaCl as a model system, however it should be generalizable to other systems.  


The EikePathway/ folder contains scripts which adapt the pathway originally proposed by Eike et al in 2004 to predict NaCl's melting point for the additive pairwise Fumi-Tosi interatomic potential.
The solid-liquid Gibbs free energy difference is calculated as a function of temperature by running a 4-step reversible pathway from the liquid to the solid phase at multiple temperatures at ambient pressure, from which the zero crossing of ΔG_sl(T) can be used to obtain the melting point.  
The appropriate pair_style and other relevant lines would need to be edited in each LAMMPS input script to test other materials/force fields. 


The NNPMeltingPoint/ folder contains scripts to obtain the melting point of a machine-learned potential "tethered" to the known melting point of a classical potential, using a thermodynamic cycle involving a predictor-corrector sort of algorithm.
This algorithm is necessary because machine-learned potentials are typically not able to sample an appropriate amount of repulsion from AIMD training data (whereas for classical potentials this is built into the functional form) - this means that the pathway mentioned above, or another pathway involving an alchemical transformation from each phase to its reference state, would not necessarily be stable for machine-learned potentials.
These scripts were developed to work with the SimpleNN flavor of NNPs (which use atom-centered symmetry functions to describe the neighborhood around each atom), however they should be applicable to any other kind of machine-learned potential implemented in LAMMPS.


The ClausiusClapeyron/ folder contains scripts to extend the solid-liquid phase boundary in the (P, T) space by numerical integration of the Clausius-Clapeyron equation, using data from MD simulations.
Given an initial point of phase coexistence, the scripts in this folder should work for any sort of interatomic potential.


(within each folder is a masterScript.py Python script which implements the logic of each algorithm and invokes various Slurm .job files to launch relevant simulations along the way - the Slurm scripts are not included here and should be customized for whatever cluster simulations are being run on, however the format of each job file can be inferred by looking at the LAMMPS .in scripts).

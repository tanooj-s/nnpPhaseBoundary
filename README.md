This repo contains relevant scripts for 'First-principles molten salt phase diagrams through thermodynamic integration' (manuscript in progress).


These scripts enable a workflow to predict the solid-liquid phase boundary of different kinds of interatomic potentials in the (P, T) space using molecular dynamics simulations.
Specifically, they allow for the prediction of phase boundaries of machine-learned potentials trained to electronic structure data such as DFT.
This can allow one to extrapolate the effects of the approximations made by those methods (such as choice of exchange-correlation functional) on bulk properties. 
The methodology here was prototyped and tested using NaCl as a model system, however it should be generalizable to other systems.  


The EikePathway/ folder contains scripts which adapt the pathway originally proposed by Eike et al in 2004 to predict NaCl's melting point for the additive pairwise BMH interatomic potential.
The solid-liquid Gibbs free energy difference is calculated as a function of temperature by running a 4-step reversible pathway from the liquid to the solid phase at multiple temperatures at ambient pressure, from which the zero crossing of ΔG_sl(T) can be used to obtain the melting point.  
In principle this same pathway should be applicable to obtain the melting point of any other classical potential in MD (although it has not been rigourously tested on other systems yet), so long as the appropriate pair_style and other lines are edited in each relevant LAMMPS input script.


The NNPMeltingPoint/ folder contains scripts to predict the melting point of a machine-learned potential (specifically a neural network potential, or NNP) "tethered" to the known melting point of a classical potential.
Here a predictor-corrector sort of algorithm is used to obtain the NNP's melting point, where an initial guess is made for that potential's melting temperature, and a sequence of alchemical, compression and heating simulations are run to obtain ΔG_sl at that guess. This is used to update the guess for the NNP's melting point until convergence.  
This algorithm is necessary because machine-learned potentials are typically not able to learn the requisite interatomic repulsion at every interatomic distance (whereas for classical potentials this is built into the functional form) - this means that the pathway mentioned above, or any other pathway which could involve an alchemical transformation from each phase to its reference state, are not necessarily stable for machine-learned potentials.
These scripts were developed to work with the SimpleNN flavor of NNPs (which use atom-centered symmetry functions to describe the neighborhood around each atom), however they should be applicable to any other kind of machine-learned potential implemented in LAMMPS, so long as the relevant pair_style and pair_coeff lines in each .in script are edited for whichever flavor of machine-learned potential.


The ClausiusClapeyron/ folder contains scripts to extend the solid-liquid phase boundary in the (P, T) space by numerically integrating the Clausius-Clapeyron equation from a single known melting point, using data from MD simulations.
Note that the method here is not Gibbs-Duhem integration, but closer to coexistence line free-energy difference integration originally proposed by Meijer et al in 1997. 
The scripts in this folder should work for any sort of interatomic potential.


(within each folder is a masterScript.py script which implements the logic of each algorithm and invokes various Slurm .job files to launch relevant simulations along the way - the Slurm scripts are not included here and should be customized for whatever cluster simulations are being run on, however the format of each jobfile can be inferred by looking at the LAMMPS .in files).

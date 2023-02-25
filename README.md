This repo contains relevant scripts for 'First-principles molten salt phase diagrams through thermodynamic integration'.

The EikePathway/ folder contains scripts which adapt the pathway originally proposed by Eike et al in 2004 to predict NaCl's melting point for the additive pairwise Fumi-Tosi interatomic potential. 
In principle this same pathway should be applicable to obtain the melting point of any other classical potential in MD (although it has not been rigourously tested on other systems yet), so long as the appropriate pair_style and other lines are edited in each relevant LAMMPS input script.
(ΔGsl is calculated as a function of temperature, from which the zero crossing can be used to obtain the melting point)

The NNPMeltingPoint/ folder contains scripts to predict the melting point of a machine-learned potential "tethered" to the known melting point of a classical potential.
This is necessary because machine-learned potentials are typically not able to learn the requisite interatomic repulsion at every interatomic distance (whereas for classical potentials this is built into the functional form) - this means that the pathway mentioned above, or any other pathway which could involve an alchemical transformation from each phase to its reference state, are not necessarily stable for machine-learned potentials.
These scripts were developed to work with the SimpleNN flavor of NNPs (which use atom-centered symmetry functions to describe the neighborhood around each atom), however they should be applicable to any other kind of machine-learned potential implemented in LAMMPS, so long as the relevant pair_style and pair_coeff lines in each .in script are edited for whichever flavor of machine-learned potential.

The ClausiusClapeyron/ folder contains scripts to extend the solid-liquid phase boundary in the (P, T) space from a single known coexistence point by numericall integration of the Clausius-Clapeyron equation using data from MD simulations.
The scripts in this folder should work for any sort of interatomic potential.

(note that within each folder is a masterScript.py script which invokes various Slurm .job files - the Slurm scripts are not included here and should be customized for whatever cluster simulations are being run on, however the format of each jobfile can be inferred by looking at the LAMMPS .in files).

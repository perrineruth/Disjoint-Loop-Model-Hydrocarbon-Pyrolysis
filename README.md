# Disjoint Loop Model

This Github repository contains the code used for the Disjoint Loop Model with Assortativity Correction used to study hydrocarbon pyrolysis. All code is located in `src/Random_Graphs`. 
Summary data and random graph sampling data are located in `Data`. To load summary statistics from MD data run the `Load_MD_Data.py` file, which can be added to the header of a notebook as follows

``from Load_MD_Data import *``

Code for sampling all random graph models (Configuration Model, Disjoint Loop Model, and Disjoint Loop Model with Assortativity Correction) is located in `HydrocarbonSimulationHelpers.py`. 
This includes an implementation of random graph sampling and the associated generating function formalisms. To load this data into a notebook add

``from HydrocarbonSimulationHelpers import *`` 

to the header.

Code for validating the Disjoint Loop Model is organized into Jupyter notebooks:
- `DegreeDistributionAnalysis.ipynb`: Code for fitting the equilibrium constants associated with the degree distribution parameters $\\bar{p}\_3$ and $p\_{\rm HH}$ to an Arrhenius law.
  This also includes validation of the parametric model for the degree distribution of the carbon skeleton.
- `LoopAnalysis.ipynb`: Code for learning the loop length distribution $\\{\phi_k\\}$ and the loop rate per carbon atom $\lambda$ and comparison to MD data.
- `LoopSampling.ipynb`: Sampling of the Disjoint Loop Model to verify that it (1) correctly recreates the loop length distribution and (2) induces assortative mixing by degree
  motivating Assortativity Correction.
- `ComponentSizes.ipynb`: Measurements of the giant component size and small component size distribution using both sampling and generating functions.

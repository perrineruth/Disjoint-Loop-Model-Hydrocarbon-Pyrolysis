# Disjoint Loop Model for Hydrocarbon Pyrolysis

This Github repository contains code for simulating hydrocarbon pyrolysis using the Disjoint Loop Model with Assortativity Correction. See [_Ruth, Dufour-Decieux, Moakler, Cameron,  "Cyclic random graph models predicting giant molecules in hydrocarbon pyrolysis", 2024_ [arXiv:2409.19141](https://arxiv.org/abs/2409.19141)]. This repository is implemented using Python and Jupyter notebooks. The main directories in this repository are 
 - `Data`: time-averaged summary statistics from ReaxFF MD simulations and sampled random graph data. 
 - `src`: source code.
 - `Figures`: output figures used for publication.

In broad strokes, the source code has two objectives which are broken into sections below:

 1. `src/Parameter_Regression`: Analysis of time-averaged summary statistics from ReaxFF data, which serve as input for the random graph models used here.
 2. `src/Random_Graph_Analysis`: Study of random graph models using sampling or generating functions validated against ReaxFF MD data.



## 1. Analysis of Input Parameters

Here, we give a description of the workflow used to obtain the inputs for random graph models from MD data and fit them Arrhenius laws. A more detailed description of the files located in `src/Parameter_Regression` is given in blocks at the end of this section. The summary statistics from MD data are contained in csv files located in `Data/processed_MD`. The entries in these csv files contain multiple variable types, so we include a file, `Load_MD_Data.py`, for parsing this data. This file can be run at the start of a Jupyter notebook as follows

  ``from Load_MD_Data import *``

This line assumes the relative file structure is the same as in this repository. After running this line, 4 pandas dataframes are loaded into the Python environment (`global_data`, `skeleton_data`, `global_data_avg`, and `skeleton_data_avg`) which contain summary statistics for the global hydrocarbon network and its carbon skeleton. The "avg" suffix denotes that the entries are averaged over runs corresponding to the same initial conditions, e.g. the samples `C4H10_3600K_1` and `C4H10_3600K_2` would be averaged into a common index `C4H10_3600K`. The file `Load_MD_Data.py` also loads a list of indices `Indices` to the Python environment as they appear in run-averaged data.

We then perform regression of the input parameters using Arrhenius laws. This is split by bond and loop parameters:
* `src/Parameter_Regression/DegreeDistributionAnalysis.ipynb`: Estimation of two parameters $p_{\rm HH}$ the fraction of hydrogen atoms that are bonded to another hydrogen and $\bar{p}_3$ the fraction of carbon atoms bonded to only 3 atoms. These are used to estimate the degree distribution, i.e. $p_k$ the probability a carbon is bonded to $k$ carbons. The equilibrium constants used to fit these parameters correspond to bond swapping and dissociation reactions, as discussed in Sec. IV B-C.
* `src/Parameter_Regression/LoopAnalysis.ipynb`: Estimation of loop parameters $\lambda$ the loop rate per carbon atom and $\phi_k$ the fraction of loops with k atoms. These parameters are obtained by ring expansion reactions which lead to a geometric tale, as discussed in Appendix D.

These fits are performed using only $\rm C_4 H_{10}$ data while the remaining data ($\rm CH_4, C_2H_6, C_8H_{18}$) is used as training data.

> #### <u>Load_MD_Data.py</u>
>
> Load `GlobalData.csv` and `SkeletonData.csv` from `Data/Processed_MD`. The ReaxFF MD data used here is the same as in [_Dufour-Decieux, Moakler, Cameron, Reed, 2022,_ [arXiv:2205.13664](https://arxiv.org/abs/2205.13664)]. The MD samples are (two runs per)
> * $\rm C_4H_{10}$: 3200K, 3200K, 3300K, 3400K, 3500K, 3600K, 3600K large ($N_{\rm C}=320$), 4000K, 4500K, 5000K
> * $\rm C_8H_{18}$: 3300K, 4000K
> * $\rm C_2H_6$: 3300K, 4000K, 4000K large ($N_{\rm C}=320$)
> * $\rm CH_4$: 3300K (only 2nd run included, not equilibriated), 3600K, 4500K
> * mix (H/C ratio 3.05): 3500K
>
> This load the csv files as pandas dataframes, whose rows correspond to datasets and columns summary statistics. For instance, `global_data.loc['C4H10_3200K_1','Nc]` gives the number of carbon atoms in the first run initialized with $\rm C_4H_{10}$ at 3200K. The summary statistics typically correspond to subgraph counts. To obtain the actual input parameters it is easiest to perform vectorized operators, e.g. the fraction of hydrogens bonded to hydrogen is computed via bond counts as
>
>  ``phh_vec = 2*global_data['Nhh']/(2*global_data['Nhh']+global_data['Nch'])``
>
> The summary statistics for the **global hydrocarbon network** are
> * `InitComp`: Initial Composition (C8H18, C4H10, C2H6, CH4, or mix).
> * `Temp`: Tempurature.
> * `Nc`: Number of carbon atoms.
> * `Nh`: Number of hydrogen atoms.	
> * `Ncc`: Number of carbon-carbon bonds.
> * `Nch`: Number of carbon-hydrogen bonds.
> * `Nhh`: Number of hydrogen-hydrogen bonds.
> * `CDegDist`: Degree distribution (in terms of total atoms) of carbons as a np.array, where index k=0,1,...,4 is the fraction	of carbons bonded to k atoms. Index 5 includes all overcoordinated carbons.
> * `HDegDist`: Degree distribution (in terms of total atoms) of hydrogens as a np.array, where index k=0,1 is the fraction	of hydrogens bonded to k atoms. Index 2 includes all overcoordinated carbons.
> * `r`: Degree assortativity coefficient of the global hydrocarbon network.	
> * `Nc3h`: Number of bonds between (global) degree 3 carbons and hydrogens.
> * `Nc4h`: Number of bonds between (global) degree 4 carbons and hydrogens.
>
> The summary statistics for the **carbon skeleton** are
> * `DegDist`: degree distribution as a np.array, where index k=0,1,...,4 is the fraction of carbons bonded to k carbons. Index 5 is the fraction of carbons bonded to 5 or more carbons by overcoordination.	
> * `r`: degree assortativity coefficient of the carbon skeleton.	
> * `CompSizeDist`: Size distribution of small molecules in terms of number of carbons.	
> * `CompSizeStdev`: Standard deviation for the size distribution of small molecules. This is not the standard deviation of the mean, i.e. it is not scaled by sample size due to time correlations.	
> * `MaxMol`: Average size of the largest molecule in terms of number of carbons.
> * `MaxMolStdev`: Standard deviation of the size of the largest molecule in terms of number of carbons.
> * `LoopSizeDist`: Size distribution of loops as a np.array, where entry $k$ is $\{\phi_k\}$, the fraction of loops that have k atoms.	
> * `LoopCountDist`: Distribution for the number of loops $N_L$. This is well approximated by a Poisson distribution, and the expected number of loops $\lambda N_{\rm C}=\sum_k k\phi_k$ is given by
>   ``skeleton_data['LoopCountDist'].apply(lambda x: x@np.arange(len(x)))``.
>
> This file also loads the dataframes `global_data_avg` and `skeleton_data_avg` which are the dataframes `global_data` and `skeleton_data` averaged over pairs of runs. The rows (indices) of these dataframes are loaded into the Python environment as `Indices`.

> #### <u>DegreeDistributionAnalysis.ipynb</u>
>
> Analysis and regression of then degree distribution parameters $p_{\rm HH}$ and $\bar{p}_3$. In order, this notebook provides code for
> * Visualizing the parametric model of the degree distribution against the time-averaged degree distribution of the carbon skeleton (Figure 8).
> * Fitting a regression for the parameter $p_{\rm HH}$ using an Arrhenius law corresponding to a bond-swapping reaction.
> * Generate a table corresponding to the accuracy of Assumption 2: carbons are bonded to 3 or 4 atoms and hydrogens are bonded to exactly 1 atom.
> * Plotting $\bar{p}_3$ as a function of temperature for $\rm C_4H_{10}$ to show it increases with temperature (Figure 7), and to visualize that $\bar{p}_3+\bar{p}_4\approx1$ for Assumption 2.
> * Visualizing the fraction of bonds from degree 3 and degree 4 carbons that are bonded to hydrogens to show they are not equal (Figure 17, right).
> * Fitting a regression for the parameter $\bar{p}_3$ using an Arrhenius law corresponding to a bond-dissociation reaction.
> * Plotting the parameters $p_{\rm HH}$ and $\bar{p}_3$ time-averaged from MD vs their values obtained from equilibrium constants fit to Arrhenius laws (Figure 9). Then the corresponding equilbrium constants $K_{\rm HH}^{\rm eff}$ and $K_{\rm C=C}^{\rm eff}$ are plotted against their fits to Arrhenius laws (Figure 18).
> * Plot of the $\ell_1$ error of the degree distribution obtained from various models (Figure 17, left).

> #### <u>LoopAnalysis.ipynb</u>
>
> Analysis and regression for the loop rate per carbon atom $\lambda$ and the loop length distribution $\{\phi_k\}$. First, this notebook is used to fit the equilbrium constants $K_{\ell}^{\rm eff}$ for loops of length $\ell=3,4,...,8$ and the tail equilibrium constant $K_L^{\rm eff}$, associated with the reactions defined in Appendix D. These reaction rates are also plotted against their Arrhenius fits as in Figure 19. Following this, the loop length distribution is plotted against its fit values and saved in `Figures/Loops/ReaxResults`. Last, code is provided for a summary figure (Figure 20) for (a) comparing the loop length distribution to MD data initialized $\rm C_4H_{10}$ at 3600K and $\rm C_8H_{18}$ at 4000K, (b) Wasserstein error of the loop length distribution, and (c) comparison of the loop rate per carbon $\lambda$ from Arrhenius fits vs. MD data.


## 2. Random Graph Analysis With the Disjoint Loop Model

Next, we include code for simulating hydrocarbon pyrolysis using random graphs. As before, the files in `src\Random_Graph_Analysis` are summarized in blocks below.

## What's needed to reproduce these results

time-averaging:
$$N_{\rm HH} = \sum_{j=j_0}^{j_1} N_{\rm HH}^{(j)} (t_{j+1}-t_j)$$
is the time-averaged number of hydrogen-hydrogen bonds. $j_0$ is a cutoff time where the system is effectively




This Github repository contains the code used for the Disjoint Loop Model with Assortativity Correction used to study hydrocarbon pyrolysis. All code is located in `src/Random_Graphs`. 
Summary data and random graph sampling data are located in `Data`. To load summary statistics from MD data run the `Load_MD_Data.py` file, which can be added to the header of a notebook as follows

``from Load_MD_Data import *``

Code for sampling all random graph models (Configuration Model, Disjoint Loop Model, and Disjoint Loop Model with Assortativity Correction) is located in `HydrocarbonSimulationHelpers.py`. 
This includes an implementation of random graph sampling and the associated generating function formalisms. To load this data into a notebook add

``from HydrocarbonSimulationHelpers import *`` 

to the header.

Code for validating the Disjoint Loop Model is organized into Jupyter notebooks:
- `DegreeDistributionAnalysis.ipynb`: Code for fitting the equilibrium constants associated with the degree distribution parameters $\bar{p}\_3$ and $p\_{\rm HH}$ to an Arrhenius law.
  This also includes validation of the parametric model for the degree distribution of the carbon skeleton.
- `LoopAnalysis.ipynb`: Code for learning the loop length distribution $\\{\phi_k\\}$ and the loop rate per carbon atom $\lambda$ and comparison to MD data.
- `LoopSampling.ipynb`: Sampling of the Disjoint Loop Model to verify that it (1) correctly recreates the loop length distribution and (2) induces assortative mixing by degree
  motivating Assortativity Correction.
- `ComponentSizes.ipynb`: Measurements of the giant component size and small component size distribution using both sampling and generating functions.

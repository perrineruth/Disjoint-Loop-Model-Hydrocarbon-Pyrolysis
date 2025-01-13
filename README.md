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
> * Visualizing the parametric model of the degree distribution against the time-averaged degree distribution of the carbon skeleton (Figure 9).
> * Fitting a regression for the parameter $p_{\rm HH}$ using an Arrhenius law corresponding to a bond-swapping reaction.
> * Generate a table corresponding to the accuracy of Assumption 2: carbons are bonded to 3 or 4 atoms and hydrogens are bonded to exactly 1 atom.
> * Plotting $\bar{p}_3$ as a function of temperature for $\rm C_4H_{10}$ to show it increases with temperature (Figure 8), and to visualize that $\bar{p}_3+\bar{p}_4\approx1$ for Assumption 2.
> * Visualizing the fraction of bonds from degree 3 and degree 4 carbons that are bonded to hydrogens to show they are not equal (Figure 18, right).
> * Fitting a regression for the parameter $\bar{p}_3$ using an Arrhenius law corresponding to a bond-dissociation reaction.
> * Plotting the parameters $p_{\rm HH}$ and $\bar{p}_3$ time-averaged from MD vs their values obtained from equilibrium constants fit to Arrhenius laws (Figure 10). Then the corresponding equilbrium constants $K_{\rm HH}^{\rm eff}$ and $K_{\rm C=C}^{\rm eff}$ are plotted against their fits to Arrhenius laws (Figure 19).
> * Plot of the $\ell_1$ error of the degree distribution obtained from various models (Figure 18, left).

> #### <u>LoopAnalysis.ipynb</u>
>
> Analysis and regression for the loop rate per carbon atom $\lambda$ and the loop length distribution $\{\phi_k\}$. First, this notebook is used to fit the equilbrium constants $K_{\ell}^{\rm eff}$ for loops of length $\ell=3,4,...,8$ and the tail equilibrium constant $K_L^{\rm eff}$, associated with the reactions defined in Appendix D. These reaction rates are also plotted against their Arrhenius fits as in Figure 20. Following this, the loop length distribution is plotted against its fit values and saved in `Figures/Loops/ReaxResults`. Last, code is provided for a summary figure (Figure 21) for (a) comparing the loop length distribution to MD data initialized $\rm C_4H_{10}$ at 3600K and $\rm C_8H_{18}$ at 4000K, (b) Wasserstein error of the loop length distribution, and (c) comparison of the loop rate per carbon $\lambda$ from Arrhenius fits vs. MD data.


## 2. Random Graph Analysis With the Disjoint Loop Model

Next, we include code for simulating hydrocarbon pyrolysis using random graphs. Most of the code for this section is contained in the `HydrocarbonPyrolysisHelpers.py` file, which contains code for sampling random graphs, analyzing them with generating function, and obtaining their estimated parameters for hydrocarbon pyrolysis using Arrhenius fits. A more detailed outline of this file is given in the first block below. The functions from this file are then loaded at the start of a Jupyter notebook. We provide two Jupyter notebooks for the following purpose
* `LoopSampling.ipynb`: Sampling loop counts from random graphs to validate they match with MD data. Additionally, measure the impact of loops in random graphs on assortative mixing by degree.
* `ComponentSizes.ipynb`: Measurement and visualization of the giant component and small component size distribution obtained from random graphs.


> #### <u>HydrocarbonPyrolysisHelpers.py</u>
>
> Helper functions for simulating hyrdocarbon pyrolysis using random graphs. This file has three main components:
> 1. #### Code for sampling random graphs for simulating hydrocarbon pyrolysis. 
>    The class `randHCNet` can be used to generate a random sample of the Disjoint Loop Model (or configuration model) for simulating a hydrocarbon network.
>    > ``randHCnet(Nc, Nh, phh, p3=0, p4=None, LoopLengths=[])``
>    > 
>    > Random hydrocarbon network class. This is used to implement the configuration model and Disjoint Loop Model (with or without Assortativity Correction). If there are not enough degree 2 nodes to construct all the desired loops then the error flag (self.eFlag) will be set to 1, and the random algorithm will need to be rerun. The graph is stored as edge list separated by type.\
>    > Inputs:
>    >  * Nc  = # C atoms
>    >  * Nh  = # H atoms
>    >  * phh = prob an H atom is bonded to another H atom
>    >  * p3  = prob a C atom is bonded to 3 atoms, default 0
>    >  * p4  = prob a C atom, default 1-p3, p3 and p4 are normalized to sum to 1
>    >  * LoopLengths = List of Lengths of each Loop, default [] corresponding to configuration model
>    
>    In practice it is easier to provide the loop rate per carbon atom $\lambda$ and the loop length distribution $\{\phi_k\}$ instead of the randomly sampled loop lengths. The following function is a wrapper that outputs a random hydrocarbon network after sampling loop lengths
>    > ``randHCnet_from_params(Nc, Nh, phh, p3=0, Lam=None, LoopLenDist=None)``
>    >
>    > Wrapper for randHCnet where the loop rate per C atom (Lam) and loop length distribution (LoopLenDist) are used as input instead of sampled loop lengths. 
>
>    If there are not enough nodes of degree 2 or higher to sample for loops, then the error flag will be set to 1. If this is the case, a new random network should be sampled. This will be rare if the expected number of nodes with degree 2 or more is greater than the number of nodes needed for loops $$\sum_{k\ge 2}p_k \ge \lambda \sum_{k\ge3}k\phi_k$$
>
>
>    The `randHCnet` class has the `assortativity_correction` method for rewiring carbon-carbon bonds according to the assortativity correction algorithm. This can be implemented as follows
>    ```
>      HCN = randHCnet(*params*)
>      HCN.assortativity_correction(*Num_steps*)
>    ```
>    To validate that this removes assortative mixing by degree one can validate the degree assortativity coefficient of the carbon skeleton is small. This is given by `HCN.assort_C`. More atoms and longer assortativity helps to remove assortative mixing by degree.
>
>    The methods `HCN.G()` and `HCN.Gc()` output the global hydrocarbon network and carbon skeleton as NetworkX graphs. Other methods are included for computing summary statistics for these graphs which are primarily wrappers of NetworkX functions: 
>    * `HCN.assort_C` the degree assortativity coefficient of the carbon skeleton
>    * `HCN.ConnComp_C` connected components of the carbon skeleton
>    * `HCN.GCC_C` number of nodes in the giant component of the carbon skeleton
>    * `HCN.CircuitRank` number of independent cycles in the hydrocarbon network
>    * `HCN.MCB` minimum cycle basis of the carbon skeleton
>    
>    The random hydrocarbon simulation class is a wrapper for the more general Disjoint Loop Model function
>
>    > ``DisjointLoopModel(degrees,LoopLengths=[],NRewire=0,verbose=False)``
>    > Random graph model where nodes may participate in a single loop, i.e. loops are disjoint. Remaining edges follow the configuration model.\
>    > Inputs:
>    > * degrees = numpy array degree sequence
>    > * LoopLengths = list lengths of each loop
>    >
>    > Output:
>    > * LEdges = NLx2 numpy array of loop edges,            NL = sum(LoopLengths) = # loop nodes
>    > * REdges = (m-NL)x2 numpy array of regular edges,     m = sum(degrees)/2 = # edges
>    > * eFlag  = error flag: 1 if too many loop nodes, 0 otherwise
>    
>    The Assortativity Correction algorithm is a wrapper for the more general algorithm for rewiring a set of nodes
>   
>    > ``rewire(Edges,degrees,EEdges,NRewire,verbose=False)``    
>    > Rewire a set of edges to have assortive mixing. Averages near a desired fraction of edges connect nodes of certain degrees. This is simply the algorithm in [Newman, M. E. J. (2002). "Assortative mixing in networks". Physical review letters](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.89.208701).\
>    > Inputs:\
>    > * Edges   = list of edges to be rewired (need not be all edges)
>    > * degrees = degree sequence (or other arbitrary integer property of nodes, if desired)
>    > * EEdges  = matrix E where E_{ij} = frac. of Edges between node ex. deg i - j (up to a constant)
>    > * NRewire = # rewire steps
>    > 
>    > Output
>    > * Edges (but rewired to be like EEdges)
>    
>    To implement Assortativity Correction, `Edges` should be the regular edges (`REdges`) from the disjoint loop model, `degrees` should be the total degrees of each node (not just from regular edges), and the entries ``EEdges[j,k]`` should match the values $\hat{e}_{j+1,k+1}$ (+1 to convert from excess degree to total degree) from Sec. III C 1.
>
> 2. #### Generating function analysis of random graph models. 
>    Once again, there is a main function for implementing this using the parameters for hydrocarbon pyrolysis
>    > ``randHCnet_GF(HCR,phh,p3,Lam=None,LoopDist=None,Rewire=False,tol=1e-8)``\
>    > Analysis of random hydrocarbon graphs using generating functions. To study the configuration model leave the loop parameters at there default values (Lam=None, LoopDist=None). To account for assortativity correction set Rewire=True. \
>    > Inputs:
>    > * HCR = hydrogen to carbon ratio (Nc / Nh)
>    > * phh = prob. H bonds to H
>    > * p3  = frac. of C nodes deg. 3
>    > * Lam = loop rate per C atom (#loops/Nc), None by default (configuration model)
>    > * LoopDist = Loop Length distribution, None by default (configuration model)
>    > * Rewire   = if assortativity correction is included, False by default
>    > * tol = tolerance for computing generating functions, default = 1e-8
>    
>    This allows us to analyze the carbon skeleton using generating functions. This a wrapper for the more general class for analyzing the Disjoint Loop Model with generating functions.
>    > ``DisjointLoop_GF(DegDist,f=None,LoopDist=None,rewire=False,tol=1e-8)``
>    > Class for generating function analysis of the disjoint loop model. Initialization constructs a method for computing for computing the generating function H(x) and degree distribution. Properties of the associated random graph model are obtained using methods of this class. In particular, the small component size distribution is obtained via self.SmallComponents(), the fraction of nodes in the giant component is obtained via self.Giant(), and the degree assortativity coefficient (only non-zero for the Disjoint Loop Model without Assortativity Correction) is given by self.assortativity_coefficient().\
>    > Inputs:
>    > * DegDist  = Degree distribution (as a numpy array)
>    > * LoopDist = Loop length distribution (as a numpy array)
>    > * rewire   = whether to add assortativity correction
>    > * tol = numerical tolerance of GF evaluations, default 1e-8
>
>    At a base level this class creates code for evaluating the function $H(x)$ which generates the size distribution $\{P_s\}$, where $P_s$ is the probabilty a node belongs to a component with $s$ nodes (carbon atoms). This class is valid for computing properties of three random graph models: the configuration model and the Disjoint Loop Model with and without Assortativity Correction. This analysis is valid in the limit of large graphs. If we set `HCN_GF = randHCnet_GF(...)` then the following properties are available
>    * `HCN_GF.Giant()`: Fraction of nodes in the giant component $S=1-H(1)$.
>    * `HCN_GF.SmallComponents(N)`: Small component size distribution $\{\pi_s\}$, where $\pi_s = \frac{P_s/s}{\sum_{s'}P_{s'}/s'}, s=1,...,N$. Values $P_s$ are obtained via the Fast Fourier Transform applied to $H(x)$ on the unit disk. A natural choice is setting $N$ as a power of 2, default $N=2^{10}$. $N$ should not be too small to give a fine enough mesh to compute FFT integrals.
>    * `HCN_GF.assortativity_coefficient()`: Degree assortativity coefficient (see Sec III C 1). Only nonzero for the Disjoint Loop Model without Assortativity Correction.
>    * `HCN_GF.threshold_function()`: Threshold function for the giant connected component. Below 1 when the giant component is missing, above 1 when the giant component is present, and equal at the critical threshold.
> 3. #### Parameter Estimation
>    Estimated input parameters via Arrhenius fits
>    * `phh_from_fit(hcr,temp)`: Fraction of hydrogens bonded to hydrogen $p_{\rm HH}$.
>    * `p3_from_fit(hcr,temp)`: Fraction of carbons bonded to 3 atoms $\bar{p}_3$.
>    * `loop_from_fit(HCR,Temp,K_max=20)`: Loop rate distribution $\lambda\phi_k, k=0,1,...,K_{\max}$ as a numpy array (zero for $k=0,1,2$ for convenient indexing). To get the total loop rate $\lambda$ one should sum these values over $k$. To get the loop length distribution one should normalize the array by $\lambda$.


> `LoopSampling.ipynb`\
> Follow-up analysis of behavior of generating functions with loop. This is done by computing the following data
> * Loop rate distribution sampled from MD, Configuration Model (Model 1), and Proposed Model
> * Degree assortativity coefficient obtained from Model 2 (Disjoint Loop Model without Assortativity Correction). This is also sample from Model 2 and Proposed Model. This is sample averages are printed out to validate that the Disjoint Loop Model has assortative mixing by degree and that Assortativity Correction removes this.
>
> This data is saved to a csv `Data/Random_Graphs/Loop_Counts.csv` which can be loaded using code in this notebook. Then code is given for the following plots
> * Visualizing the loop rate distribution from MD data compared to the configuration model and Proposed Model (Figure 11).
> * Degree assortativity coefficient given by MD data compared to the Disjoint Loop Model (obtained via generating functions).

> `ComponentSizes.ipynb`\
> Analysis of component sizes computed by sampling random graphs and by generating functions. This code in this notebook is used to do the following:
> * Obtain statistics about the giant component via sampling and generating functions.
> * Visualizing results about the giant component
>   * Compare the size distribution of the giant from MD data, Proposed Model, and Previous Work (Figure 1)
>   * (a) More detailed comparison of the size distribution of the giant from MD compared to Proposed Model. (b) Comparison of the expected size of the giant component from all models using both methods (Figure 12).
>   * Additional histograms comparing the size of the giant from MD to Proposed Model (Figure 22 and 23).
> * Predicted Phase diagram of hydrocarbon pyrolysis from Proposed Model (Figure 15).
> * Visualizing results about small components
>   * First, estimate the small molecule size distribution for each model. Obtain the Wasserstein $W_1$ error of these distributions to the true distribution from MD data
>   * Plot of the small component size distribution of Proposed Model and Previous work compared to MD data. Additionally plot the $W_1$ error of these models (Figure 13).
>   * Additional plots of the small component size distribution compared to MD data (Figures 23 and 24).

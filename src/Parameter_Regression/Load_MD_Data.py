###################################
#####     Load_MD_Data.py     #####
###################################

# loads time-averaged summary data from MD simulations consisting of four pd.dataframes
# 1 - global_data = summary dataframe of the global hydrocarbon network
#     vals: 'InitComp': initial molecular composition as a string, 
#           'Temp':     temperature in Kelvin 
#           'Nc':       number of carbon atoms 
#           'Nh':       number of hydrogen atoms 
#           'Ncc':      number of carbon-carbon bonds
#           'Nch':      number of carbon-hydrogen bonds
#           'Nhh':      number of hydrogen-hydrogen bonds
#           'CDegDist': degree distribution of carbon atoms in terms of total number of atoms bonded to
#           'r':        degree assortativity coefficient of global hydrocarbon network
#           'Nc4h':     number of bonds between a carbon bonded to 4 atoms and a hydrogen
#           'Nc4h':     number of bonds between a carbon bonded to 3 atoms and a hydrogen
# 2 - global_data_avg = global_data averaged over two runs for each set of initial conditions
# 3 - skeleton_data = summary dataframe of the carbon skeleton
#     vals: 'DegDist':          degree distribution of the carbon skeleton (i.e. number of carbons a carbon is bonded to)
#           'r':                degree assortativity coefficient of the carbon skeleton
#           'CompSizeDist':     small molecule / component size distribution in terms of number of C atoms
#           'CompSizeStdev':    standard deviation of measurements of CompSizeDist
#           'MaxMol':           size of the largest molecule in terms of number of C atoms
#           'MaxMolStdev':      single standard deviation of 'MaxMol'
#           'LoopSizeDist':     loop length distribution, i.e. probabilities \phi_k a loop is length k
#           'LoopCountDist':    distribution for the number of indendent loops M-Nc+comps, m=# CC bonds, comps=# connected components
# 4 - skeleton_data_avg = skeleton_data averaged over two runs
#
# also loads Indices of this data (datasets of format "InitComp_TemperatureK_number-of-atoms")

import pandas as pd
import numpy as np
# what to include in import *
__all__ = ['global_data','skeleton_data','global_data_avg','skeleton_data_avg','Indices']

# Summary data
global_data = pd.read_csv('../../Data/processed_MD/GlobalData.csv',index_col=0)
skeleton_data = pd.read_csv('../../Data/processed_MD/SkeletonData.csv',index_col=0)
# convert strings to np arrays as needed
global_data.loc[:,['CDegDist','HDegDist']] = global_data.loc[:,['CDegDist','HDegDist']].map(lambda string: 
                         np.array([float(word) for word in string.strip('[]').replace('\n','').split()]))
skeleton_data.loc[:,['DegDist','CompSizeDist','CompSizeStdev','LoopSizeDist','LoopCountDist']] = \
                        skeleton_data.loc[:,['DegDist','CompSizeDist','CompSizeStdev','LoopSizeDist','LoopCountDist']].map(lambda string: 
                        np.array([float(word) for word in string.strip('[]').replace('\n','').split()]))

# Summary data average between runs
Indices = sorted(list(set([i[:-2] for i in skeleton_data.index])))
# not converged but useful comparison for $\bar{p}_3$ and $p_{\rm HH}$ (both increasing, i.e. underestimates)
Indices.remove('CH4_3300K')     
global_data_avg = pd.DataFrame(columns=global_data.columns)
skeleton_data_avg = pd.DataFrame(columns=skeleton_data.columns)
def aux_f(x):
    if type(x) == str: return x[:len(x)//2]
    else: return x/2
for idx in Indices:
    skeleton_data_avg.loc[idx] = (skeleton_data.loc[idx+'_1']+skeleton_data.loc[idx+'_2'])/2
    global_data_avg.loc[idx] = (global_data.loc[idx+'_1']+global_data.loc[idx+'_2']).apply(aux_f)
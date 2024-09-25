####################################################################
### Code for simulating hydrocarbon networks using random graphs ###
####################################################################
# Last updated: 8/5/2024
#
# 3 primary sections:
#   1. Wrapper for global hydrocarbon network
#   2. Disjoint loop model + rewire code
#   3. Parameter functions fit from Arrhenius law
# Methods include sampling and generating function approach

import numpy as np
import networkx as nx
from scipy.optimize import fsolve

# minimum cycle basis helper function - refined pre-conditioning
from networkx.algorithms.cycles import _min_cycle_basis
def min_cycle_basis(G,weight = None):
    """
    Minimum cycle basis using NetworkX code partitioned via biconnected components. This is typically much
    faster than nx.minimum_cycle_basis (which preconditions using connected components) and gives the same
    result.
    """
    return sum((_min_cycle_basis(nx.edge_subgraph(G,c).copy(),weight) for c in\
                 nx.biconnected_component_edges(G) if len(c)>2),[])

##########################################
### global hydrocarbon network wrapper ###
##########################################

class randHCnet:
    """
    Random hydrocarbon network class. This is used to implement the configuration model and Disjoint Loop
    Model (with or without Assortativity Correction).
    Inputs:
        Nc  = # C atoms
        Nh  = # H atoms
        phh = prob an H atom is bonded to another H atom
        p3  = prob a C atom is bonded to 3 atoms, default 0
        p4  = prob a C atom, default 1-p3, p3 and p4 are normalized to sum to 1
        LoopLengths = List of Lengths of each Loop, default [] corresponding to configuration model
    """
    def __init__(self, Nc, Nh, phh, p3=0, p4=None, LoopLengths=[]):
        # initialize atoms
        self.Nc = int(Nc)
        self.Nh = int(Nh)
        self.N  = self.Nc+self.Nh
        self.CNodes = np.arange(self.Nc,dtype=int)
        self.HNodes = np.arange(self.Nc,self.N,dtype=int)
        self.Nodes  = np.arange(self.N,dtype=int)

        # initialize probabilities
        if p4 is None: p4 = 1-p3
        self.phh = phh
        self.p3 = p3/(p3+p4)
        Nc3 = int(self.p3*Nc)
        self.p4 = 1-self.p3
        Nc4 = Nc-Nc3

        # some bond counts
        Nhh = int(self.phh * self.Nh / 2)
        Nch = self.Nh - 2*Nhh

        # HH molecules bond them together
        HStubs = self.HNodes
        HHBonds = HStubs[:2*Nhh].reshape(Nhh,2)
        edgeList = HHBonds

        # pair CH bonds
        CStubs = np.concatenate((np.tile(np.arange(Nc4,dtype=int),4), np.tile(np.arange(Nc4,self.Nc,dtype=int),3)))
        np.random.shuffle(CStubs)
        CHBonds = np.column_stack((CStubs[-Nch:],HStubs[2*Nhh:]))
        edgeList = np.vstack((CHBonds,edgeList))
        CStubs = CStubs[:-Nch]

        # Carbon Subgraph Degrees
        degrees = np.zeros(self.Nc,dtype=int)
        for x in CStubs: degrees[x]+=1
        self.degrees_C = degrees

        # draw remaining via single loop model
        LoopBonds,RegCCBonds,self.eFlag = DisjointLoopModel(self.degrees_C,LoopLengths)

        # final edge list
        self.HHBonds    = HHBonds
        self.CHBonds    = CHBonds
        self.LoopBonds  = LoopBonds
        self.RegCCBonds = RegCCBonds

    # rewiring scheme to remove assortativity
    def assort_correct(self,steps):
        """
        Assortitivity Correction. Perform rewiring to remove assortative mixing by
        degree. Instead of computing \hat{e}_{jk}^E directly this just subtracts
        the number of loop edges from nodes deg. j - k from desired amount.
        """
        # obtain excess degree distribution
        DegDist = np.zeros(5)
        for d in self.degrees_C: DegDist[d] += 1 # here DegDist[k] = Nc * pk
        exDegDist = DegDist[1:] * np.arange(1,5)
        exDegDist /= sum(exDegDist)
        # expected number of edges between nodes by ex. degree
        EEdges = np.outer(exDegDist,exDegDist)*(DegDist @ np.arange(5))/2
        # desired # of reg edges
        for (u,v) in self.LoopBonds:
            EEdges[self.degrees_C[u]-1,self.degrees_C[v]-1] -= 1
        EEdges = (EEdges+EEdges.T)/2 # make symmetric
        # run rewiring
        self.RegCCBonds = rewire(self.RegCCBonds,self.degrees_C,EEdges,steps)
        
    # most functions are wrappers of NetworkX functions

    def G(self):
        """
        Global hydrocarbon as a NetworkX graph.
        """
        nxGraph = nx.Graph()
        # for some reason MCB does not work with nodes as np.int32
        nxGraph.add_nodes_from([int(u) for u in self.Nodes])
        for Edges in [self.HHBonds,self.CHBonds,self.LoopBonds,self.RegCCBonds]:
            nxGraph.add_edges_from([(int(u),int(v)) for u,v in Edges])
        # remove self loops here - kept previously for rewiring scheme
        nxGraph.remove_edges_from(nx.selfloop_edges(nxGraph)) 
        return nxGraph
    
    def Gc(self):
        """
        Carbon skeleton as a NetworkX graph.
        """
        nxGraph = nx.Graph()
        # for some reason MCB does not work with nodes as np.int32
        nxGraph.add_nodes_from([int(u) for u in self.CNodes])
        nxGraph.add_edges_from([(int(u),int(v)) for u,v in self.LoopBonds])
        nxGraph.add_edges_from([(int(u),int(v)) for u,v in self.RegCCBonds])
        # remove self loops here - kept previously for rewiring scheme
        nxGraph.remove_edges_from(nx.selfloop_edges(nxGraph)) 
        return nxGraph
    
    def assort_C(self):
        """
        Degree assortativity coefficient of the carbon skeleton.
        """
        return nx.degree_assortativity_coefficient(self.Gc())
    
    def ConnComp_C(self):
        """
        Connected components of the carbon skeleton.
        """
        return nx.connected_components(self.Gc())

    def GCC_C(self):
        """
        Number of C atoms in the giant connected component.
        """
        return max([len(Comp) for Comp in self.ConnComp_C()])

    def CircuitRank(self):
        """
        Number of independent cycles in the carbon skeleton.
        """
        return self.Gc().number_of_edges() - self.Nc + len(self.ConnComp_C())
    
    def MCB(self):
        """
        Minimum cycle basis of the carbon skeleton.
        """
        return min_cycle_basis(self.Gc())
    
    def draw(self):
        """
        Plot global hydrocarbon skeleton. This tends not to look amazing.
        """
        nx.draw(self.G(),node_size=20,node_color=['b']*self.Nc + ['r']*self.Nh)

    def draw_C(self):
        """
        Plot carbon skeleton.
        """
        nx.draw(self.Gc(),node_size=20,node_color='b')



def randHCnet_from_params(Nc, Nh, phh, p3=0, Lam=None, LoopLenDist=None):
    """
    Wrapper for randHCnet where the loop rate per C atom (Lam) and loop length distribution
    (LoopLenDist) are used as input instead of sampled loop lengths. 
    """
    if Lam is None:
        return randHCnet(Nc=Nc,Nh=Nh,phh=phh,p3=p3)
    else:
        # sample loop lengths, Poisson(Lam*Nc) then i.i.d. sampling.
        LoopLens = np.cumsum(LoopLenDist).searchsorted(np.random.rand(np.random.poisson(Lam*Nc)))
        return randHCnet(Nc=Nc,Nh=Nh,phh=phh,p3=p3,LoopLengths=LoopLens)



def randHCnet_GF(HCR,phh,p3,Lam=None,LoopDist=None,Rewire=False,tol=1e-8):
    """
    Analysis of random hydrocarbon graphs using generating functions. As before, this model
    includes the configuration model and Disjoint Loop Models.
    Inputs:
        HCR = hydrogen to carbon ratio (Nc / Nh)
        phh = prob. H bonds to H
        p3  = frac. of C nodes deg. 3
        Lam = loop rate per C atom (#loops/Nc), None by default (configuration model)
        LoopDist = Loop Length distribution, None by default (configuration model)
        Rewire   = if assortativity correction is included, False by default
        tol = tolerance for computing generating functions
    """
    # compute degree distribution
    pcc = 1-HCR*(1-phh)/(4-p3)
    aux = [1-pcc,pcc]
    DegDist      = (1-p3)*np.convolve(aux,np.convolve(aux,np.convolve(aux,aux))) # Degree Dist from C atoms deg 4
    DegDist[:4] += p3*np.convolve(aux,np.convolve(aux,aux))                      # Degree Dist from C atoms deg 3
    # no loops -> Configuration model
    if Lam is None:
        return  DisjointLoop_GF(DegDist,tol=tol) # generic Disjoint Loop Model
    # compute f function: f(d) = prob node deg. d in a loop (Eq. (14))
    MLen = sum(LoopDist*np.arange(len(LoopDist)))   # mean loop length
    fq = lambda q,j: 1-q**(j*(j-1)/2)
    q = fsolve(lambda q: DegDist.dot(fq(q,np.arange(5))) - Lam*MLen,.5)[0] # appropriate corner probabilities
    f = lambda j: fq(q,j)
    return DisjointLoop_GF(DegDist,f,LoopDist,Rewire,tol)



##########################################
###         Disjoint Loop Model        ###
##########################################

def DisjointLoopModel(degrees,LoopLengths=[],NRewire=0,verbose=False):
    """
    Random graph model where nodes may participate in a single loop, i.e. loops are disjoint. 
    Remaining edges follow the configuration model.
    Inputs:
        degrees = numpy array degree sequence
        LoopLengths = list lengths of each loop
    Output:
        LEdges = NLx2 numpy array of loop edges,            NL = sum(LoopLengths) = # loop nodes
        REdges = (m-NL)x2 numpy array of regular edges,     m = sum(degrees)/2 = # edges
        eFlag  = error flag: 1 if too many loop nodes, 0 otherwise
    """
    # graph params
    N = len(degrees)
    NLoopNodes = sum(LoopLengths)

    # draw loop nodes with the following weights
    if NLoopNodes == 0: 
        LoopNodes = np.zeros(0,dtype=int) # empty
    else:
        weights = degrees*(degrees-1)/2
        weights /= sum(weights)
        # if impossible to draw enough nodes return an error
        if NLoopNodes > sum(weights>0): 
            return None,None,1
        # draw without replacement
        LoopNodes = np.random.choice(np.arange(N),NLoopNodes,replace=False,p=weights)
        np.random.shuffle(LoopNodes) # first nodes are biased to be higher degree, unbias edges

    # draw loop nodes
    idx = 0
    LoopEdges = np.zeros((0,2),dtype=int)
    for Length in LoopLengths:
        newEdges = np.vstack((LoopNodes[idx:(idx+Length)],np.roll(LoopNodes[idx:(idx+Length)],1))).T
        LoopEdges = np.vstack((LoopEdges,newEdges))
        idx+=Length

    # create remaining stubs
    stubCounts = degrees.copy().astype(int)
    stubCounts[LoopNodes] -= 2                  # remove 2 stubs loop nodes
    Stubs = np.repeat(np.arange(N,dtype=int),stubCounts)
    np.random.shuffle(Stubs)
    # draw as edges
    NRegEdges = len(Stubs)//2
    RegEdges = np.vstack((Stubs[:NRegEdges],Stubs[NRegEdges:2*NRegEdges])).T

    # if adding assortativity correction - Use the one in above wrapper for larger model
    if NRewire:
        dmax = max(degrees)
        # expected number of edges between an edge of node of degree d1,d2
        # excess degree distribution
        degdist = np.zeros(dmax+1)
        for d in degrees: degdist[d]+=1
        degdist    = sum(degdist)
        exdegdist  = degdist[1:]*np.arange(1,dmax+1)
        exdegdist /= sum(exdegdist)
        # expected number of edges between nodes by degree
        EEdges = np.outer(exdegdist,exdegdist)*sum(degrees)//2
        # remove loop edges
        for (u,v) in RegEdges:
            EEdges[degrees[u]-1,degrees[v]-1] -= 1
        EEdges = (EEdges+EEdges.T)/2
        RegEdges = rewire(RegEdges,degrees,EEdges,NRewire)

    return LoopEdges,RegEdges,0



def confModel(degrees):
    """
    Wrapper for Disjoint Loop Model when there are no loops. In this case there are no loop 
    edges and the error flag won't trigger. Only return classical edges.
    """
    Edges,_,__ = DisjointLoopModel(degrees)
    return Edges



def rewire(Edges,degrees,EEdges,NRewire,verbose=False):
    """
    Rewire a set of edges to have assortive mixing. Averages near a desired fraction of edges
    connect nodes of certain degrees. This is simply the algorithm in:
        Newman, M. E. J. (2002). "Assortative mixing in networks". Physical review letters.
    Inputs:
        Edges   = list of edges to be rewired (need not be all edges)
        degrees = degree sequence
        EEdges  = matrix E where E_{ij} = frac. of Edges between node ex. deg i - j (up to a constant)
        NRewire = # rewire steps
    Output
        Edges (but rewired to be like EEdges)
    """
    # sampled edges
    if verbose:
        SEdges = np.zeros_like(EEdges)
        for xxx in range(Edges):
            d1,d2 = degrees[Edges[xxx,:]]
            SEdges[d1-1,d2-1] += 1
        SEdges = (SEdges+SEdges.T)/2
        print('Initial Rewire Error:', np.sum(np.abs(EEdges-SEdges)))

    # Rewiring scheme
    for _ in range(NRewire):
        # two random edge indices
        i1,i2 = np.random.randint(0,len(Edges),2)
        if i1 == i2: continue
        p1,p2 = np.random.randint(0,2,2)                                 # boolean to flip edge order
        u1,v1   = Edges[i1,p1],  Edges[i1,1-p1]                          # nodes in edge 1
        du1,dv1 = degrees[u1]-1, degrees[v1]-1                           # excess degrees
        u2,v2   = Edges[i2,p2],  Edges[i2,1-p2]
        du2,dv2 = degrees[u2]-1, degrees[v2]-1
        # Metropolis Hastings probability - Add epsilon to denominator for divide by zero error
        if np.random.random() < EEdges[du1,du2]*EEdges[dv1,dv2]/(EEdges[du1,dv1]*EEdges[du2,dv2]+1e-14): 
            Edges[i1,:] = np.array([u1,u2])
            Edges[i2,:] = np.array([v1,v2])
            
    # print final error
    if verbose:
        SEdges = np.zeros_like(EEdges)
        for xxx in range(Edges):
            d1,d2 = degrees[Edges[xxx,:]]
            SEdges[d1,d2] += 1
        SEdges = (SEdges+SEdges.T)/2
        print('Final Rewire Error:', np.sum(np.abs(EEdges-SEdges)))
    
    return Edges



class DisjointLoop_GF:
    """
    Class for generating function analysis of the disjoint loop model. Initialization constructs 
    a method for computing for computing the generating function H(x) and degree distribution.
    Properties of the associated random graph model are obtained using methods of this class. In
    particular, the small component size distribution is obtained via *.SmallComponents(), the fraction
    of nodes in the giant component is obtained via *.Giant(), and the degree assortativity coefficient
    (only non-zero for the Disjoint Loop Model without Assortativity Correction) is given by
    *.assortativity_coefficient().
    Inputs:
        DegDist  = Degree distribution (as a numpy array)
        LoopDist = Loop length distribution (as a numpy array)
        rewire   = whether to add assortativity correction
        tol = numerical tolerance of GF evaluations, default 1e-8 is much smaller than model
    """
    def __init__(self,DegDist,f=None,LoopDist=None,rewire=False,tol=1e-8):
        self.degree_distribution = DegDist
        self.LoopDist = LoopDist
        self.rewire = rewire
        self.f = f
        self.tol = tol
        # Main goal GF H for small + large CCs

        dmax = len(DegDist) # max degree + 1
        # no loops -- configuration model
        if f is None:
            # deg dist gen functions
            G0 = lambda x: DegDist @ x**np.arange(dmax)
            dG0 = lambda x: (DegDist[1:]*np.arange(1,dmax)) @ x**np.arange(dmax-1)
            G1 = lambda x: dG0(x)/dG0(1)

            # recurrence H1 = xG1(H1)
            def H1(x):
                h1 = 1/2
                for _ in range(10000):
                    aux = x*G1(h1)
                    if abs(aux-h1)<=tol:
                        break
                    h1 = aux
                return h1
            
            H0 = lambda x: x*G0(H1(x))
            self.H = H0
            return
        
        # loop excess degree distribution
        dPhi0 = lambda x: LoopDist[1:] @ (np.arange(1,len(LoopDist))*x**np.arange(len(LoopDist)-1))
        Phi1 = lambda x: dPhi0(x)/(dPhi0(1)+1e-14) # add to denominator to avoid overflow

        # loop model without rewire
        if not rewire:
            # general info for both models
            # update degree distribution
            aux = DegDist
            DegDist = np.vstack((aux*(1-f(np.arange(len(aux)))),
                                np.concatenate((aux[2:]*f(np.arange(2,len(aux))), np.zeros(2)))))

            # degree distributions
            G = lambda x,z: z**np.arange(2) @ DegDist @ x**np.arange(dmax)

            # excess degree distribution
            dGdx = lambda x,z: z**np.arange(2) @ DegDist[:,1:] @ (np.arange(1,dmax) * x**np.arange(dmax-1))
            GE = lambda x,z: dGdx(x,z)/dGdx(1,1)
            dGdz = lambda x: DegDist[1,:] @ x**np.arange(dmax)
            GL = lambda x: dGdz(x)/dGdz(1)
            
            # excess component size
            phiHL = lambda He,x: Phi1(x*GL(He))

            def HE(x):
                HE = 1/2
                # recurrence He = x*Ge(He,phiHL)
                for _ in range(10000):
                    aux = x*GE(HE,phiHL(HE,x))
                    if abs(aux-HE) <= tol:
                        break
                    HE = aux
                return HE
            
            def H(x):
                He = HE(x)
                return x*G(He,phiHL(He,x))
            
            self.H = H
            return
        
        # loop model with rewire
        else:
            # excess degree distribution
            exDegDist = DegDist[1:]*np.arange(1,dmax)
            exDegDist/= sum(exDegDist)
            # for loop edges
            exDegL = DegDist[2:]*f(np.arange(2,dmax))
            exDegL/= sum(exDegL)
            # regular edges
            jvals = np.arange(1,dmax)
            exDegR = np.tile(exDegDist,(dmax-1,1)).T * jvals
            exDegR[1:,:] -= 2*np.tile(exDegL,(dmax-1,1)).T * f(jvals)
            exDegR /= (jvals-2*f(jvals))

            def PhiHL(He,x):
                return Phi1(x*exDegL.dot(He[1:]**np.arange(dmax-2)))
            
            def HE(x):
                He = 1/2*np.ones(dmax-1)
                # iteration in section 2.4
                c = jvals*(1-f(jvals))/(jvals-2*f(jvals))
                for _ in range(10000):
                    aux1 = c*He**np.arange(dmax-1)
                    aux2 = (1-c)*He**(np.arange(dmax-1)-2)*PhiHL(He,x)
                    aux = x*((aux1+aux2)@exDegR)
                    if max(np.abs(aux-He)) < tol:
                        break
                    He = aux
                return He
            
            def H(x):
                He = HE(x)
                PHL = PhiHL(He,x)
                aux = (1-f(jvals))*He**jvals + f(jvals)*He**(jvals-2)*PHL
                return x*DegDist[0] + x*DegDist[1:].dot(aux)

            self.H = H
            return
        
    def Giant(self):
        """
        Fraction of nodes in the giant component S=1-H(1)
        """
        return 1-self.H(1)
    
    def SmallComponents(self,N=2**10):
        """
        Small component size distribution \{\pi_s\}
        \pi_s = P_s/s / \sum P_{s'}/s'
          P_s = coeff. of H(x)
        """
        # double points to avoid error from FFT
        thvals = np.linspace(0,2*np.pi,2*N,endpoint=False)
        xvals = np.exp(1j * thvals)
        Hvals = np.array([self.H(x) for x in xvals])

        coefficients = np.real(np.fft.fft(Hvals))               # coefficients of H up to scaling
        pi_s = coefficients[1:]/np.arange(1,len(coefficients))  # $\pi_s$ up to scaling
        pi_s = pi_s[:N]/sum(pi_s[:N])                           # normalize
        return pi_s

    def assortivity_coefficient(self):
        """
        Estimate degree assortativity coefficient of Disjoint Loop Model. 0 for the configuration
        model and the Disjoint Loop Model with assortativity correction.
        """
        # only nonzero assortativity if disjoint loop model w/o rewiring
        if self.LoopDist is not None and not self.rewire:
            # degree distributions
            DD = self.degree_distribution
            # generic excess degree
            exDD = DD[1:]*np.arange(1,len(DD))
            exDD/= sum(exDD)
            # regular edges j - 2f(j)
            exDDR = DD[1:]*(np.arange(1,len(DD))-2*self.f(np.arange(1,len(DD))))
            exDDR/= sum(exDDR)
            # loop edges
            exDDL = DD[2:]*self.f(np.arange(2,len(DD)))
            exDDL/= sum(exDDL)

            Mean = exDD@np.arange(len(exDD))
            Var = exDD@(np.arange(len(exDD))**2)-Mean**2

            # fraction of edges that are regular edges
            jvals = np.arange(1,len(DD))
            c = sum((jvals-2*self.f(jvals))*DD[1:])/sum(jvals*DD[1:])

            E = c*np.outer(exDDR,exDDR)
            E[1:,1:] += (1-c)*np.outer(exDDL,exDDL)

            return (np.arange(len(exDD))@E@np.arange(len(exDD)) - Mean**2)/Var
        else:
            return 0





##########################################
###           Parameter Fits           ###
##########################################
        

# Degree Distribution from Arrhenius fit
# Arrhenius parameters
Ahh,Chh = 4.4056, 18704
A3,C3   = 6.46e-4, 30526
# functions for p3/phh
def phh_from_fit(HCR,Temp):
    """
    Estimate the parameter phh as a function of the hydrogen/carbon ratio (HCR) and temperature (Temp) using
    the equilibrium constant (Khh). 
    """
    K = Ahh*np.exp(-Chh/Temp)
    return 1 - (4/HCR + 1 - np.sqrt((4/HCR-1)**2+16*K/HCR))/2/(1-K)
def p3_from_fit(HCR,Temp):
    """
    Estimate the parameter bar{p}3 as a function of the hydrogen/carbon ratio (HCR) and temperature (Temp) using
    the equilibrium constant (Kc=c). 
    """
    K = A3*np.exp(-C3/Temp)
    phh = phh_from_fit(HCR,Temp)
    aux = (12.011+HCR)*Temp
    return K*aux/(HCR*phh + K*aux)

# Loop length distribution as a vector Lambda * phi_k, default length 20
# Arrhenius parameters
A_vec = [0.12389968008853605, 0.00015152476936858725, 6.773381093603916e-05, 4.832975633194727, 
         5.219656228043643, 4.431445997072439, 4.420962213664369]
C_vec = [5451.064196288146, -28110.14661095346, -36237.86751560563, 18948.792552310457, 
         22167.948046709855, 20251.109758230716, 18253.55550832747]
# obtain loop distribution K_max = max loop length
def loop_from_fit(HCR,Temp,K_max=20):
    """
    Obtain the loop rate distribution \{\lambda\phi_k\} as a function of the hydrogen/carbon ratio (HCR) and 
    temperature (Temp). One can also set a maximum loop length, default K_max = 20.
    """
    p3  = p3_from_fit(HCR,Temp)
    phh = phh_from_fit(HCR,Temp)

    kc = 4-p3                   # mean degree
    pch = HCR/kc * (1-phh)
    rHCH = 3*(2-p3)*pch**2      # number H-C-H triples / C atom

    if np.isscalar(p3):     LamPhi_dist_arrh = np.zeros(K_max)
    else:                   LamPhi_dist_arrh = np.zeros((K_max,len(p3))) # if vectorized HCR or temp, match format other params
    # small loops
    LamPhi_dist_arrh[3] = A_vec[0]*np.exp(-C_vec[0]/Temp) * p3
    LamPhi_dist_arrh[4] = A_vec[1]*np.exp(-C_vec[1]/Temp) * p3**2 
    LamPhi_dist_arrh[5] = A_vec[2]*np.exp(-C_vec[2]/Temp) * p3**2 
    # ring expansion reactions
    for L in range(6,len(LamPhi_dist_arrh)):
        LamPhi_dist_arrh[L] = A_vec[min(L-3,6)]*np.exp(-C_vec[min(L-3,6)]/Temp) * LamPhi_dist_arrh[L-1]*rHCH / (phh*HCR)
    return LamPhi_dist_arrh

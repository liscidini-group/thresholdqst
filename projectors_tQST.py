#!/usr/bin/env python3

#
# A general class to generate projectors
# for the threshold quantum state tomography
# all classes MUST have the method
# --- projector_from_matrix_element(self,i,j,comp='r'):
#     returning the projector that measures the 'r'eal or 'i'maginary part of the (i,j) element of
#     the density matrix
#
# --- projector_name_from_matrix_element(self,i,j,comp='r')
#     returns the name of the projector
#
# --- all_projectors(self)
#     that returns all the projectors that have been generated and the corresponding list of (i,j,"i|r")
#
# 1. n qubits using HVDR
# --- Projectors_tQST_qubit
#
# 2. n qudits using the generalization of the above
#
# 3. a system of arbitrary qudits, calculating the projectors that minimize the distance to the
# operators that would measure the real and imaginary part of the density matrix.
# --- OptimizedProjectors_tQST()
#
# This class extends the qudit class.
#

import sys
import os
# set these variables BEFORE importing numpy
# multithreaded blas/lapack are not needed unless the matrices are _really_ large
#os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
#os.environ["OMP_NUM_THREADS"]      = "1" # export OMP_NUM_THREADS=1

import random
import pickle
import functools as ft
import itertools as it

import numpy as np
import scipy as sp
import scipy.sparse

import matplotlib.pyplot as plt
from tqdm import tqdm

import psutil

import qudits_class as qd

def get_rank(V):
    """
    Returns the rank of the matrix that checks if a set of projectors are independent
    On input, V is a matrix that has the projector |p> as rows
    """
    M = V @ np.conj(V.T)
    M = np.real(M * M.conj())
    max_rank = M.shape[0]

    try:
        L = np.linalg.cholesky(M)
        diag = np.sort(np.diag(L))
        # checks whether cholesky worked, but the matrix is singular
        # because of small positive eigenvalues
        if diag[0] < 1e-4*diag[1]:
            r = max_rank-1
        else:
            r = max_rank
    except np.linalg.LinAlgError as err:
        r = max_rank-1


    return(r, r==max_rank)

# Projectors_tQST_qubit

b1_list = [('H','O'), ('D','R'),('O','O'),('V','O')]
b1 = np.array(b1_list, dtype='U20,U20').reshape(2,2)
z_list = [('O','O'), ('O','O'),('O','O'),('O','O')]
z = np.array(z_list, dtype='U20,U20').reshape(2,2)

def my_transpose(b):
    bf = b.T.flatten() # flatten the transpose
    bfi = np.copy(bf)
    for i in range(bf.size):
        x,y = bf[i]
        bfi[i] = (y,x)
    return bfi.reshape(b.shape)

def prepend(c,b):
    """
    prepends the character to all the elements of b
    """
    bf = b.flatten()
    ret = np.copy(bf)
    for i in range(bf.size):
        x,y = bf[i]
        if x != 'O':
            x = c+x
        if y != 'O':
            y = c+y
        ret[i] = (x,y)
        #print(ret[i])
        
    return ret.reshape(b.shape)


def add(m1,m2):
    """
    add the tuples
    """
    m1f = m1.flatten()
    m2f = m2.flatten()
    assert m1f.size == m2f.size
    
    ret = np.copy(m1f)
    
    for i in range(m1f.size):
        x1,y1 = m1f[i]
        x2,y2 = m2f[i]
        x = x1+x2
        x = x.replace('O','')
        if len(x) == 0:
            x = 'O'
        y = y1+y2
        y = y.replace('O','')
        if len(y) == 0:
            y = 'O'
        
        ret[i] = (x,y)
        
    return ret.reshape(m1.shape)

def one_more_qubit(b):
    H = prepend('H',b)
    V = prepend('V',b)
    D = prepend('D',b)
    R = prepend('R',my_transpose(b))
    Z = np.array( [('O','O')]*b.size, dtype='U20,U20').reshape(b.shape)
    
    #print(Z.shape)
    
    row1 = np.hstack((H,add(R,D)))
    row2 = np.hstack((Z,V))
    
    return np.vstack((row1,row2))

def bn(n):
    "recursively add one qubit"
    ret = b1
    for i in range(n-1):
        ret = one_more_qubit(ret)        
    return(ret)

class Projectors_tQST_qubit():
    
    def __init__(self, n):
        self.dim = 2**n
        
        filename="tQST_projectors_"+str(n)+"qubit.npy"
        try:
            self.b = np.load(filename)
            print("# loaded from",filename)            
        except:
            print("# generating tQST projectors for",n,"qubit")
            self.b = bn(n)
            print("# saving to",filename)
            np.save(filename, self.b)

        #print(self.b)
        
    def projector_from_string(self,S):
        """
        Creates a projector from a string in the bn array
        """
        D = {}
        D['H'] = np.array([1,0],  dtype=np.complex128)
        D['V'] = np.array([0,1],  dtype=np.complex128)
        D['D'] = np.array([1,1],  dtype=np.complex128) / np.sqrt(2)
        D['R'] = np.array([1,1j], dtype=np.complex128) / np.sqrt(2)
        
        # creates a list of vectors
        vec = [D[x] for x in S]        
        proj = ft.reduce(np.kron,vec)        
        return(proj)

    
    def projector_name_from_matrix_element(self,i,j,comp='r'):
        assert j >= i        

        x = self.b[i,j]
        name = x[0]
        
        if comp == 'i' and i != j:
            name = x[1]

        return name

    def projector_from_matrix_element(self,i,j,comp='r'):
        name = self.projector_name_from_matrix_element(i,j,comp)
        return self.projector_from_string(name)
    
    def all_projectors(self):
        # first the diagonal ones
        ret = []
        mat_el_of_proj = []
        for i in range(self.dim):
            ret.append( self.projector_from_matrix_element(i,i) )
            mat_el_of_proj.append( (i,i,'r') )
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                t = (i,j,'r')
                ret.append( self.projector_from_matrix_element(*t) )
                mat_el_of_proj.append( t )
                t = (i,j,'i')
                ret.append( self.projector_from_matrix_element(*t) )
                mat_el_of_proj.append( t )

        return np.array(ret), mat_el_of_proj
    
#
# OptimizedProjectors_tQST()
#

def SU2_basis_reduced(d,b_vec="DR"):
    # number of vectors
    N = 2*d*d-d
    if len(b_vec) == 2:
        N = d*d

    proj = np.zeros((N,d),dtype=np.complex128)
    
    su2_names = ['H','V']
    # diagonal
    for i in range(d):
        proj[i,i]=1
    # real part and imaginary parts
    # we loop over all the pair of distinct indices
    isq = 1/np.sqrt(2)

    match b_vec:        
        case "DR":
            # combinations = without repetitions
            for idx,c in zip(it.count(start=d,step=2), it.combinations(range(d),2)):
                i,j = c
                #D
                proj[idx,i] = isq
                proj[idx,j] = isq
                #R
                proj[idx+1,i] = isq        
                proj[idx+1,j] = 1j*isq
                su2_names.append('D')
                su2_names.append('R')                

        case "DL":
            # combinations = without repetitions
            for idx,c in zip(it.count(start=d,step=2), it.combinations(range(d),2)):
                i,j = c
                #D
                proj[idx,i] = isq
                proj[idx,j] = isq
                #L
                proj[idx+1,i] = isq        
                proj[idx+1,j] = -1j*isq
                su2_names.append('D')
                su2_names.append('L')                

        case "AR":
            # combinations = without repetitions
            for idx,c in zip(it.count(start=d,step=2), it.combinations(range(d),2)):
                i,j = c
                #A
                proj[idx,i] = isq
                proj[idx,j] = -isq
                #R
                proj[idx+1,i] = isq        
                proj[idx+1,j] = 1j*isq
                su2_names.append('A')
                su2_names.append('R')                

        case "AL":
            # combinations = without repetitions
            for idx,c in zip(it.count(start=d,step=2), it.combinations(range(d),2)):
                i,j = c
                #A
                proj[idx,i] = isq
                proj[idx,j] = -isq
                #L
                proj[idx+1,i] = isq        
                proj[idx+1,j] = -1j*isq
                su2_names.append('A')
                su2_names.append('L')                

        case "DARL":
            # combinations = without repetitions
            for idx,c in zip(it.count(start=d,step=4), it.combinations(range(d),2)):
                i,j = c
                #D
                proj[idx,i] = isq
                proj[idx,j] = isq
                #A
                proj[idx+1,i] = isq
                proj[idx+1,j] = -isq
                #R
                proj[idx+2,i] = isq        
                proj[idx+2,j] = 1j*isq
                #L
                proj[idx+3,i] = isq                
                proj[idx+3,j] = -1j*isq
                su2_names.append('D')
                su2_names.append('A')                
                su2_names.append('R')
                su2_names.append('L')                

        case _:
            print("Basis vectors can be: DR(default), DL, AR, AL and DARL")

    #proj = proj.tocsr()
    numbers = list(range(proj.shape[0]))

    ll = len(str(numbers[-1]))    
    if d == 2:
        names = su2_names
    else:
        names = ['S'+str(d)+"-"+str(x).zfill(ll) for x in numbers]
    
    diag_names = [names[i] for i in range(d)]
    
    return(names, proj, diag_names)

def SUd_basis_reduced(d):
    # number of vectors
    N = d*d
    #proj = sp.sparse.lil_array((N,d),dtype=np.complex128)
    proj = np.zeros((N,d), dtype=np.complex128)
    names = [] # Sd-i.j-[xyz]+ 
    ll=len(str(d))
    
    # diagonal
    for i in range(d):
        proj[i,i]=1
        names.append('S'+str(d)+"-"+str(i).zfill(ll)+"."+str(i).zfill(ll)+'-z+')
    # real part and imaginary parts
    # we loop over all the pair of distinct indices
    isq = 1/np.sqrt(2)
    # combinations = without repetitions
    for idx,c in zip(it.count(start=d,step=2), it.combinations(range(d),2)):
        i,j = c
        proj[idx,i] = isq
        proj[idx,j] = isq
        names.append('S'+str(d)+"-"+str(i).zfill(ll)+"."+str(j).zfill(ll)+'-x+')        

        proj[idx+1,i] = isq        
        proj[idx+1,j] = 1j*isq
        names.append('S'+str(d)+"-"+str(i).zfill(ll)+"."+str(j).zfill(ll)+'-y+')
        
    #proj = proj.tocsr()
    
    diag_names = [names[i] for i in range(d)]
    
    return(names, proj, diag_names)

def Density_Matrix_Observables_Elements(d):
    """
    Returns the indices of the nonzero elements of the observables
    """
    retn = [] # matrix indices
    reti = [] # rows
    retj = [] # cols
    retval = [] # values
    mat_el_of_proj = []
    
    n = 0 # number of the matrix element to be considered
    
    # diagonal
    for i in range(0,d):
        retn.append(n)
        reti.append(i)
        retj.append(i)
        retval.append(1)
        n = n+1
        mat_el_of_proj.append( (i,i,'r') )
        
    # 2. the off diagonal ones    
    for i in range(0,d):
        for j in range(i+1,d):
            retn.append(n)
            reti.append(i)
            retj.append(j)
            retval.append(1/2)
            
            retn.append(n)
            reti.append(j)
            retj.append(i)
            retval.append(1/2)
            
            n = n+1
            mat_el_of_proj.append( (i,j,'r') )
            
            retn.append(n)
            reti.append(i)
            retj.append(j)
            retval.append(-1j/2)
            
            retn.append(n)
            reti.append(j)
            retj.append(i)
            retval.append(1j/2)           
            
            n = n+1
            mat_el_of_proj.append( (i,j,'i') )

    idx = (np.array(retn), np.array(reti), np.array(retj))
    n = np.array(retn)
    i = np.array(reti)
    j = np.array(retj)
    val = np.array(retval)
    return(n,i,j, val, mat_el_of_proj)


def has_duplicates(L):
    # returns True the list L has duplicates
    # unless the list is None
    if L is None:
        return(False)
    
    l = len(L)
    s = len(set(L))
    return ( not (l==s) )
    
class Explorer():
    indices = () # tuple of indices to explore
    N = 0 # length of indices
    state = []   # the state, that is list of at most len(indices) whose
                 # entry n is such that  indices[n][state[n]] = current_list[n]
    current_list = [] # the current list of indices
    
    def __init__(self, indices):
        """
        Initialize a class to explore the various possible sets of
        tomographic projectors.

        indices: is a list of indices of projectors corresponding to the
        observables that measure an entry of the density matrix
        
        REQUIREMENT: indices must be sorted according to the length
        """
        self.indices = tuple(map(tuple,indices))
        self.state = [0 for x in indices if len(x)==1]
        self.current_list = [indices[i][0] for i in range(len(self.state))]
        self.N = len(self.indices)

        print("initial list of size",len(self.current_list))
        print("len indices",len(indices))
        
    def get_list(self):
        return self.current_list

    def set_next(self):
        n = len(self.current_list)-1 # the slot we are considering
        N = len(self.indices[n])-1   # the max number of indices in this slot

        if self.state[n] < N:
            # we have some other indices from this slot
            # get the next on from the current list
            self.state[n] = self.state[n] + 1
            self.current_list[n] = self.indices[n][self.state[n]]
        else:
            # remove one (backtrack) and get the next
            if len(self.current_list) > 1:
                self.current_list.pop()
                self.state.pop()
                self.set_next()
            else:
                # cannot backtrack anymore... we have exhausted the possibilities
                self.current_list = None
                #return(None)
	
    def set_another(self):
        if len(self.state) == len(self.indices):
            # we cannot add another state, so we call next() that does the backtracking
            self.set_next()
        else:
            n = len(self.state)
            # add the first index from the next set of projectors
            self.state = self.state + [0]
            self.current_list = self.current_list + [self.indices[n][0]]
            while has_duplicates( self.current_list ):
                self.set_next()
                
    def next(self):
        self.set_next()        
        while has_duplicates(self.current_list):
            self.set_next()        
        return(self.get_list())
        
    def another(self):
        if len(self.state) == len(self.indices):
            # we cannot add another state, so we call next() that does the backtracking
            return(self.next())
        else:
            n = len(self.state)
            # add the first index from the next set of projectors
            self.state = self.state + [0]
            self.current_list = self.current_list + [self.indices[n][0]]
            if has_duplicates( self.current_list ):
                return(self.next())
            else:
                return(self.current_list)
        
class OptimizedProjectors_tQST():
    qudit_list = []
    dim        = 0       # dimension of the Hilbert space of the qudits
    single_basis = []    # a list of single basis vectors

    proj_components = [] # list of the components of the various projectors
    psi = []             # the vectors |psi> so that P = |psi><psi|

    proj = []            # projectors assigned to matrix elements
    mat_el_of_proj = []  # the matrix elements corresponding to the various |proj>
    
    def __init__(self,qudit_list, generate=SUd_basis_reduced, verbose=False):

        self.qudit_list = qudit_list
        self.dim = np.array(qudit_list).prod()
        self.verbose = verbose

        single_basis_all = tuple( generate(d) for d in qudit_list )
        self.single_basis_names = tuple(x[0] for x in single_basis_all)
        self.single_basis = tuple(x[1] for x in single_basis_all)
                
        filename='Optimized_projectors'+str(qudit_list)
        if os.path.exists(filename):
            self.load(filename)
        else:        
            self.find_optimized_projectors()
            self.save(filename)


    def load(self,filename):
        with open(filename,'rb') as f:
            print("# loading from",filename)
            self.proj = np.load(f)
            self.mat_el_of_proj = pickle.load(f)            
            self.proj_components = pickle.load(f)
            
    def save(self,filename):
        """
        saves the projectors and their matrix elements to file
        """
        with open(filename,'wb') as f:
            print("# saving to",filename)
            np.save(f, self.proj)
            pickle.dump(self.mat_el_of_proj,f)            
            pickle.dump(self.proj_components,f)
            
    def product_state(self, V):
        """
        Generate the product state from the vectors in the list V
        """
         #ret = V[0]
         #for i in range(1,len(V)):
         #    ret = sp.sparse.kron(ret,V[i])

        ret = ft.reduce(np.kron,V)        
        return(ret)

    def components(self, idx_list):
        """
        Generate a list of components using idx_list to get from the basis of single qudits
        """
        n = len(self.qudit_list)
        #phi = tuple( self.single_basis[i][[j],:] for (i,j) in zip(range(n),idx_list) )
        phi = tuple( self.single_basis[i][j,:] for (i,j) in zip(range(n),idx_list) )        
        return(phi)


    def prune_bigidx(self, sorted_idx):
        initial_values = sum([len(x) for x in sorted_idx])

        assigned = [x[0] for x in sorted_idx if len(x) == 1]

        for i in range(len(sorted_idx)):
            if(len(sorted_idx[i]) > 1):
                l = sorted_idx[i]
                [ l.remove(x) for x in assigned if x in l]

        final_values = sum([len(x) for x in sorted_idx])

        print("pruning from",initial_values,"to",final_values)
        
        return initial_values != final_values
        
    
    def extract_projectors_explore(self, sorted_idx):
        """
        Extracts a tomographically complete set of vectors from a sorted bigidx
        Assumes that bigidx is already sorted, and so is mat_el_of_proj
        """
        while(self.prune_bigidx(sorted_idx)):
            pass
                    
        ee = Explorer(sorted_idx)
        idx = ee.get_list()

        pbar = tqdm(position=0, leave=True) # so that it does not do funny things
        max_rank = len(sorted_idx)
        r = 0
        while idx is not None:
            V = self.psi[idx]
            r, is_max_rank = get_rank(V)
            if r == max_rank:
                # found complete tomographic set
                pbar.set_description("rank = %d/%d" % (r,max_rank))
                break
            else:
                if is_max_rank:
                    # max rank, but not enough vectors
                    idx = ee.another()
                else:
                    # not max rank, consider the next one
                    # and backtrack if needed
                    idx = ee.next()
            pbar.set_description("rank = %d/%d" % (r,max_rank))
            pbar.update()

        pbar.close()
        if idx is not None:
            print("\nFound tomographic complete set")
            if self.verbose is True:
                print(idx)
            #self.mat_el_of_proj = mat_el_of_proj
        else:
            raise Exception("Tomographic complete set not found")

        return(idx)

    def projector_bigidx(self):
        """
        Returns a big-index list of projectors
        That is a list of length number of observables
        containing a list of all the projectors that minimize the distance
        """
        On, Oi, Oj, Oval, mat_el_of_proj = Density_Matrix_Observables_Elements(self.dim)
        print("Number of observables matrix elements",len(On))
        self.mat_el_of_proj = mat_el_of_proj
        
        bigidx = []

        N = 1 # number of possible projector vectors
        l = []
        for x in self.single_basis:
            N = N * x.shape[0]
            l.append(range(x.shape[0]))
        
        self.psi = []
        self.proj_components = []
        all_dist = []

        # let's do some memory calculations
        MAXMEM = 64 # Gb
        psi_mem = N * self.dim * 16 / (1024**3) # memory for the psi vectors
        print("Memory needed for the projector vectors",psi_mem,"Gb")

        for x in tqdm(it.product(*l), total=N): # loop on all the possible projectors
        #for x in it.product(*l):
            phi = self.components(x)
            psi = self.product_state(phi).flatten() #.toarray().flatten()
            self.psi.append(psi)
            self.proj_components.append(x)

            # compute the distances to all the observables
            d1 = psi[Oi] * np.conj(psi[Oj]) # P_ij
            d2 = Oval - d1                  # O_ij - P_ij
            d = np.real(d2 * np.conj(d2))   # |O_ij - P_ij|^2
            
            #d = np.real(d2 * np.conj(d2) - d1 * np.conj(d1)) # not sure why this
            
            # now we have to add the terms that have the same index in On            
            S = np.zeros(self.dim*self.dim)

            # diagonal
            S[:self.dim] = d[:self.dim]
            # two terms each out of the diagonal
            x = d[self.dim:].reshape(-1,2)
            S[self.dim:] = np.sum(x,axis=1)
            all_dist.append(np.array(S).flatten() )

        self.psi = np.array(self.psi)
        M = np.array(all_dist).T        
        bigidx = [ np.where(S==S.min())[0].tolist() for S in M ]

        tmp_idx = sorted(bigidx,key=len)
        mat_el_of_proj = [x for _,x in sorted(zip(bigidx,self.mat_el_of_proj), key=lambda x: len(x[0])) ]

        bigidx = tmp_idx
        self.mat_el_of_proj = mat_el_of_proj

        return(bigidx)
    
    def find_optimized_projectors(self):
        """
        for each rho_observable, find a complete set of projectors
        """

        bigidx = self.projector_bigidx()

        idx = self.extract_projectors_explore(bigidx)  
        #idx = self.extract_projectors_random(bigidx)        
        
        #if self.verbose == True:
            #print("Found",len(idx),"unique projectors out of",self.dim*self.dim)
        x = [ self.proj_components[i] for i in idx]
        self.proj_components = x.copy()
        self.proj = self.psi[idx]
        
        #return(self.psi[idx].reshape(len(idx),-1), idx)

    def projector_from_matrix_element(self,i,j,comp='r'):
        idx = self.mat_el_of_proj.index( (i,j,comp) )
        return self.proj[idx]

    def projector_name_from_matrix_element(self,i,j,comp='r'):
        idx = self.mat_el_of_proj.index( (i,j,comp) )
        l = self.proj_components[idx]
        name = ''.join([self.single_basis_names[i][j] for i,j in zip(it.count(),l)])
        return name

    def all_projectors(self):
        return self.proj, self.mat_el_of_proj
    


####################################################################################################
if __name__ == '__main__':
    P1 = OptimizedProjectors_tQST([2]*7, SU2_basis_reduced)    

    row=1
    for i in range(1,P1.dim):
        print("real",row,i,"->",P1.projector_name_from_matrix_element(row,i) )
    print("")    
    for i in range(2,P1.dim):
        print("imag",row,i,"->",P1.projector_name_from_matrix_element(row,i,'i') )        

    # for x,y in zip(*P1.all_projectors()):
    #     print(y,x)
    # print("**********************************************************************")
    # for x,y in zip(*P2.all_projectors()):
    #     print(y,x)
    # exit()
    
   # print("DIAGONAL")
   # for i in range(4):
   #     v1 = P1.projector_from_matrix_element(i,i,'r')
   #     v2 = P2.projector_from_matrix_element(i,i,'r')
   #     print(v1,v2)
   # 
   # print("----------------------------------------------------------------------")
   # for i in range(P1.dim):
   #     for j in range(i+1,P1.dim):
   #         v1 = P1.projector_from_matrix_element(i,j,'r')
   #         v2 = P2.projector_from_matrix_element(i,j,'r')
   #         diff = np.sum(np.abs(v1-v2))
   #         if diff > 1e-12:
   #             print("Difference at",i,j,"real:",diff,end=' -> ')
   #             print(P1.projector_name_from_matrix_element(i,j,'r'),
   #                   P2.projector_name_from_matrix_element(i,j,'r'))
   #         v1 = P1.projector_from_matrix_element(i,j,'i')
   #         v2 = P2.projector_from_matrix_element(i,j,'i')
   #         diff = np.sum(np.abs(v1-v2))
   #         if diff > 1e-12:
   #             print("Difference at",i,j,"imag:",diff,end=' -> ')
   #             print(P1.projector_name_from_matrix_element(i,j,'i'),
   #                   P2.projector_name_from_matrix_element(i,j,'i'))
            

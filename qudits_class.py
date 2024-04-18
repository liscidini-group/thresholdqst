#!/usr/bin/env python3

# 
# A class for qudits (that is quantum systems with d states; d=2 are
# qubits)
# by Giovanni Garberoglio - garberoglio@ectstar.eu
#
# version 1.0 - 2018/08/29
# version 1.1 - 2020/07/02 (renamed qunit to qudit)
# version 1.2 - 2023/01/18 (stripped down for quantum tomography)

# Here we develop the Qudit class, which is a general class to deal with
# many-body states. It is used to generate and manipulate pure states of
# the form |n1 n2 n3 ... nN>, where ni is the dimension of the i-th system
# (2 for a qubit, 3 for a qutrit, 64 if we are dealing with an oscillator
# with a certain cutoff and so on...)
#


import numpy as np
import scipy as sp
import scipy.linalg
from scipy.linalg import sqrtm

import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import dsolve
from scipy.stats import unitary_group

import matplotlib.pyplot as plt

import density_matrix_tool as dmt

class Qudit:
    """
    A class representing a system of qudits
    We only represent the state by a statistical operator (density matrix)
    even if the state is pure
    """
    qdit_list = None # the list used for initialization
    qudit_arr = None # same as above, but as a numpy array
    nqudits   = None # the total number of qudits (len of qudit_list)
    dim       = None # the dimension of the Hilbert space of all the qudits
    states  = None   # list of dimensions used when printing

    rho     = None # the statistical operator

    def __init__(self, qudit_list):
        """
        Initialize using a list of qudits
        e.g. [2, 3] will create a system with a qubit and a qutrit
        """
        #
        # the helper function above helps creating N qudit systems
        #
        ## size of the Hilbert space of each qudit
        self.qudit_list = qudit_list
        self.qudit_arr = np.array(qudit_list,dtype=int)
        ## number of systemss
        self.nqudits=self.qudit_arr.size
        ## dimension of the total Hilbert space
        self.dim=int(self.qudit_arr.prod())
        #
        ## defines the vector states, used to print the states
        self.states=np.ones(self.nqudits,dtype=int);
        n=self.nqudits
        for i in range(0,n-1):
            self.states[n-2-i] = self.states[n-1-i]*self.qudit_arr[n-1-i]
        #
        ## the initial state of the system
        ## defaults to a zero vector
        ##
        self.rho   = np.zeros([self.dim, self.dim], dtype=np.complex128);


    def set_seed(self, seed):
        """
        Sets the random seed.
        """
        np.random.seed(seed)
        
    def state_from_number(self,n):
        """
        Starting from the index of a state in the product Hilbert space,
        returns a list of the states in the subsystems
        or an empty list if an error occurs
        """
        s=[]
        if(n < self.dim):
            N=n;
            #s=s+"|"
            x = N // self.states[0]
            s=[x]
            N = N - x * self.states[0]
            for i in range(1,self.nqudits):
                x = N // self.states[i]
                s.append(x) # s+" "+str(x)
                N = N - x * self.states[i]
            #s=s+">"
        return(s)

    def set_pure_state(self, v):
        """
        set a pure system
        """
        if v.size != self.dim:
            print("set_pure_state: dimension mismatch")
            print("                got",v.size,"while expecting",self.dim)
        else:
            self.rho = np.outer(v,np.conj(v))
            
    def random_pure_state(self, ndiag=None, complex_amplitude=True):
        self.rho = dmt.random_pure_state(self.dim,ndiag,complex_amplitude)
        
    def random_pure_state_real(self):
        """
        Creates a random state with real amplitudes
        """
        self.random_pure_state(complex_amplitude=False)

    def zero_extra_coefficients(self, M):
        """
        This routine zeroes out the coefficient in M that are not needed.
        M is of dimensions [nqudits, max_space]
        We create a mask that has zero in each row for the columns larger
        than the dimensionality of the qudit correponding to this row.
        """
        line=np.arange(0,max(self.qudit_arr))
        mask=np.tile(line,(self.nqudits,1))
        for i in range(0,self.nqudits):
            mask[i,:] = mask[i,:] < self.qudit_arr[i]
        return(M * mask)

    def normalize_coefficients(self, M):
        """
        Normalize the coefficients of the matrix M so that each qudit has a
        normalized wavefunction
        """
        norm = (M * M.conj()).sum(axis=1)
        norm = np.sqrt(norm.real)
        for i in range(0,self.nqudits):
            M[i,:] = M[i,:] / norm[i]
        return(M)                    

    def separable(self,M):
        """
        Creates a separable state, that is        
        a state of the form |psi1>|psi2> ... |psiN>
        where |psik> = sum M(k,i) |i>

        M must be a matrix of dimensions self.system and max(self.qudit_arr)
        whose extra entries are not considered.
        These are in fact set to zero in the routine zero_extra_coefficients()
        """
        # zero out the state        
        state = np.zeros(self.dim,dtype=np.complex128)
        # check the dimensions
        s=M.shape
        if s[0] != self.nqudits or s[1] != max(self.qudit_arr):
            print ("separable: wrong shape of input")
            print ("separable: setting state to zero")
            return
        #
        # calculate the coefficient of the given state
        k = range(self.nqudits)
        for n in range(0,self.dim):
            nk=self.state_from_number(n)
            state[n] = np.prod(M[k, nk])
            
        self.set_pure_state(state)
        
    def random_separable(self):
        """Build a separable state from random single states"""
        r=self.nqudits
        c=int(np.max(self.qudit_arr))
        M = np.random.uniform(-1.0, 1.0, size=c*r).reshape(r,c)
        M = M + 1j * np.random.uniform(-1.0, 1.0, size=c*r).reshape(r,c)
        M=self.zero_extra_coefficients(M)
        M=self.normalize_coefficients(M)
        self.separable(M)
        return(M)
    
    def density_matrix(self):
        return(self.rho)

##
## DENSITY MATRICES
##

    def set_density_matrix(self,rho):
        """
        Sets a gives state (which is a tuple) to the given value
        """
        self.is_pure = False
        self.rho = np.copy(rho)
            
    def random_density_matrix(self,n=0):
        self.rho = dmt.random_density_matrix(self.dim,n)

    def Frobenius_norm(self):
        """
            Frobenius norm of a matrix, that is the sum of the squares of the
            absolute values of A.
            If A is None then we return the norm of the density matrix of
            this system
        """
        A = self.rho
        return(dmt.Frobenius_norm(A))

    def purity(self):
        """
        Purity of a matrix A, that is tr(A^2).
        This is the same as the norm of the matrix.
        """
        A= self.rho
        return(dmt.purity(A))
    
    def fidelity(self,A):
        """
        Returns (tr( sqrt( sqrt(A) B sqrt(A) ))^2
        If B is none, then we take the density matrix of this system
        """
        B = self.rho
        return(dmt.fidelity(A,B))

    def concurrence(self):
        """
            Calculates the concurrence of the density matrix A
            At the moment implemented only for a two qubit system
            If A is None then we return the concurrence of the density matrix of
            this system
        """
        return(dmt.concurrence(self.rho))

    def entanglement_of_formation(self):
        """
            Calculates the entanglement of formation
            for a two qubit system
            If A is None then we return the concurrence of the density matrix of
            this system
        """
        return(dmt.entanglement_of_formation(self.rho))

##
## VISUALIZATION OF THE DENSITY MATRICES
##

    def plot_density_matrix_3D(self, **kwargs):
        # YlOrBr is a nice colormap for progressive data
        dmt.plot_density_matrix_3D(self.rho, **kwargs)
        
    def plot_density_matrix_2D(self, **kwargs):
        dmt.plot_density_matrix_2D(self.rho, **kwargs)

    ######################################################################        
    ### NOT TESTED BELOW HERE
    ######################################################################
    ##
    ## PARTIAL TRACING AND TRANSPOSING
    ##
    def number_from_state(self,s):
        """
        Returns the index of the state in the product space from the
        indices in the subsystems provided by the list s
        -1 in case of an error
        """
        #n=np.fromstring(s,sep=' ',dtype=int);
        n=np.array(s,dtype=int)
        if(n.size != self.nqudits): return(-1)
        isok = n < self.qudit_arr
        ## returns the scalar product
        if(isok.prod()):
            return( sum(n * self.states) )
        else:
            return(-1)
            
    def partial_trace(self,sub_list):
        """
        Performs a partial trace of the statistical operator.
        On input, sub_list is a tuple with the indices of the subsystems
        that are to be traced out.
        sub_list is at most of len(self.qudit_arr) and its elements are at
        most len(self.qudit_arr)
        
        Example
        -------
        suppose we have a [2,3,4] system (one qubit, one qutrit and one
        quadrit)
        - trace over the qutrit: sub_list = [1]
        - trace over the qubit and quatrit: sub_list = [0,2]        
        """
        qudits = self.qudit_arr.tolist()       # list of qudits
        rhop = self.rho.reshape(qudits+qudits) # reshape the density operator

        # now that we have reshaped rho[ qudits ; qudits ] we have to
        # perform the partial traces over the axis in sub_list
        #
        # to do so, we first reverse sort the list of indices
        sub_list.sort()
        sub_list.reverse()
        sub_tuple = tuple(sub_list)
        
        # and we loop over this list performing one partial trace in turn
        # over each qudit in sub_list
        #
        # the first axis to trace out is the index
        # the second is the index + half the length of the shape of the
        # partial trace so far
        
        for i in sub_tuple:
            j = i + int(len(rhop.shape)/2)
            rhop = np.trace(rhop, axis1=i, axis2=j)

        # at the end, we resize it as a square matrix
        dim = int(np.sqrt(rhop.size))

        return(rhop.reshape(dim,dim))
    
    def reduced_density_matrix(self,n):
        rho = self.density_matrix()
        return self.partial_trace(rho,n)
    
    def partial_transpose(self,rho,n):
        """Perform partial transposition on subsistem n"""
        rhopt = 0 * rho;
        for i in range(0,self.dim):
            li=self.state_from_number(i)
            for j in range(0,self.dim):
                lj=self.state_from_number(j)
                lii = list(li) # perform an actual COPY of the lists
                ljj = list(lj)
                tmp = lii[n]
                lii[n] = ljj[n]
                ljj[n] = tmp
                ii = self.number_from_state(lii)
                jj = self.number_from_state(ljj)
                rhopt[i,j] = rho[ii,jj]
        return(rhopt)

    def reduce_operator(self,rho,n):
        """Trace out everything except the nth system"""
        red=np.zeros( (self.qudit_arr[n],self.qudit_arr[n]) , dtype=np.complex128 )
        
        # create a list of the states of the from
        # | i1 i2 i3 ... 0 ... iN>
        #                ^-in
        L=[]
        for i in range(0, self.dim):
            l=self.state_from_number(i)
            if(l[n] == 0): L.append(l)

        for i in range(0, self.qudit_arr[n]):
            for j in range(0, self.qudit_arr[n]):
                # perform the sum on the states
                # < i1 i2 ... i ... iN | rho | j1 j2 ... j ... iN>
                # and put it in red(i,j)
                for k in range(0, len(L)):
                    l1=list(L[k])
                    l2=list(L[k])
                    l1[n] = i
                    l2[n] = j
                    ii = self.number_from_state(l1)
                    jj = self.number_from_state(l2)
                    red[i,j] += rho[ii,jj]
        return(red)


if __name__ == "__main__":
    q1 = Qudit([2,3,4])
    q1.random_density_matrix()
    #q1.rho = np.arange(q1.dim*q1.dim).reshape(q1.dim,q1.dim)
    
    for i in range(q1.dim):
        x = q1.state_from_number(i)
        j = q1.number_from_state(x)
        print(i,"->",x,"->",j)
    
    print("rho shape",q1.rho.shape)
    
    rhop_1 = q1.partial_trace_old(2)
    rhop_2 = q1.partial_trace([2])

    print("rhop_1 shape",rhop_1.shape)
    print("rhop_2 shape",rhop_2.shape)

    print("rhop_1\n",rhop_1)
    print("rhop_2\n",rhop_2)    

    print("trace rho",np.trace(q1.rho))    
    print("trace rhop_1",np.trace(rhop_1))
    print("trace rhop_2",np.trace(rhop_2))    

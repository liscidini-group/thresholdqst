#!/usr/bin/env python3

#
# Classes to generate LOCAL projectors for the threshold quantum state tomography
#
# *** All classes MUST have the methods ***
#
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

import os

import random
import pickle
import functools as ft
import itertools as it

import numpy as np
import scipy as sp
import scipy.sparse

from tqdm import tqdm

import psutil

def extended_projector_from_string(S):
        """
        Creates a projector from a string in the bn array
        """
        D = {}
        D['H'] = np.array([1, 0],  dtype=np.complex128)
        D['V'] = np.array([0, 1],  dtype=np.complex128)
        D['D'] = np.array([1, 1],  dtype=np.complex128) / np.sqrt(2)
        D['A'] = np.array([1, -1], dtype = np.complex128) / np.sqrt(2)
        D['R'] = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
        D['L'] = np.array([1, -1j], dtype=np.complex128) / np.sqrt(2)
        
        # creates a list of vectors
        vec = [D[x] for x in S]        
        proj = ft.reduce(np.kron,vec)        
        return(proj)


class Projectors_tQST_qubit_local:
    dictionary_real = {(False,False,None): 'H',
                       (True, True, None): 'V',
                       (False,True, True): 'D',
                       (False,True, False):'R',
                       (True,False, True): 'D',
                       (True,False, False):'R'}

    dictionary_real_inverted = {(False,False,None): 'H',
                                (True, True, None): 'V',
                                (False,True, True): 'R',
                                (False,True, False):'D',
                                (True,False, True): 'R',
                                (True,False, False):'D'}

    dictionary_imag = dictionary_real_inverted
    dictionary_imag_inverted = dictionary_real

    def __init__(self, n_qubit):
        self.n_qubit = n_qubit 

    def projector_real(self, i,j,verbose=False):
        #N = self.dim
        N = 2**(self.n_qubit)
        assert i<N
        assert j<N
        assert j >= i
        
        ik, jk = i,j
        inversion = False
        vector = ''
        dict_to_use = self.dictionary_real

        for i in range(self.n_qubit):
            
            quadrant = (ik>=N/2, jk>=N/2)
            
            # the matrix is divided into four quadrants
            # named with the value of the pair `quadrant`        
            # according to:
            #
            # (False,False) | (False, True)
            # -----------------------------
            # (True,False)  | (True,True)
            #
            
            if verbose == True:
                print((ik,jk),'of',N,'-> quadrant',quadrant)
        
            upper = None
        
            # need to know whether my index falls in 
            # the upper or lower part of the quadrant 
            # (upper includes the diagonal for the real part)
            if quadrant == (False,True):
                jp = int(jk-N/2)
                upper = jp >= ik # include the diagonal
                inversion = not upper # next step uses the inverted dictionary
            
            if quadrant == (True,False):
                ip = int(ik-N/2)
                upper = jk >= ip
                inversion = not upper
            
            pos = (*quadrant,upper)

            if verbose == True:
                print('pos',pos,'inversion',inversion)
        
            # add the next letter to the vector name
            vector = vector + dict_to_use[pos]
        
            if inversion == True:
                dict_to_use = self.dictionary_real_inverted
            else:
                dict_to_use = self.dictionary_real
        
            # indices of the element in the quadrant for the next step
            if ik >= N/2:
                ik = int(ik - N/2)
            if jk >= N/2:
                jk = int(jk - N/2)
        
            # recude the dimension by half and proceed
            N = int(N/2)
    
        return(vector)

    def projector_imag(self,i,j,verbose=False):
        #N = self.dim
        N = 2**(self.n_qubit)
        assert i<N
        assert j<N
        assert j>i
            
        ik, jk = i,j
        inversion = False
        vector = ''
        dict_to_use = self.dictionary_imag

        for i in range(self.n_qubit):

            quadrant = (ik>=N/2, jk>=N/2)
        
            if verbose == True:
                print((ik,jk),'of',N,'-> quadrant',quadrant)

            lower = None
            
            # need to know whether my index falls in 
            # the upper or lower part of the quadrant 
            # (lower includes the diagonal for the imaginary part)
            if quadrant == (False,True):
                jp = int(jk-N/2)
                lower = jp <= ik
                inversion = lower # next step uses the inverted dictionary
            
            if quadrant == (True,False):
                ip = int(ik-N/2)
                lower = jk <= ip
                inversion = lower
            
            pos = (*quadrant,lower)

            vector = vector + dict_to_use[pos]

            if verbose == True:
                print('pos',pos,'inversion',inversion)
                print('vector so far',vector)
            
            if inversion == True:
                dict_to_use = self.dictionary_imag_inverted
            else:
                dict_to_use = self.dictionary_imag
            
            if ik >= N/2:
                ik = int(ik - N/2)
            if jk >= N/2:
                jk = int(jk - N/2)
            N = int(N/2)
    
        return(vector)

    def projector_from_matrix_element(self,i,j,comp='r'):
        proj_name = self.projector_name_from_matrix_element(i,j,comp)
        return extended_projector_from_string(proj_name)
        
    def projector_name_from_matrix_element(self,i,j,comp='r'):
        if comp == 'r':
            return( self.projector_real(i,j) )
        else:
            return( self.projector_imag(i,j) )


####################################################################################################
if __name__ == '__main__':
    nq = 4
    
    P1 = Projectors_tQST_qubit_local(nq)

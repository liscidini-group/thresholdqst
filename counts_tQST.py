#!/usr/bin/env python3

#
# A general class to provide counts
# extends projectors
#
# This is to be used with maximum-likelihood estimation
#

import itertools as it

import numpy as np
import density_matrix_tool as dmt
import projectors_tQST_local as PtQST

class Counts_tQST():
    P        = None # Projector class
    projs  = []
    counts = []
    rho      = None # density matrix used for synthetic data

    
    def __init__(self, P):
        self.P = P

            
    def reset_counts(self):
        self.counts   = -np.ones(len(self.proj_names))
        self.proj_idx = []


    def get_counts(self):
        """
        Returns the set of projectors / measurements 
        """
        return self.projs, self.counts

                   
    def set_density_matrix(self, rho):
        self.rho = np.copy(rho)

        
    ######################################################################
    ### Threshold ML tomography
    ######################################################################    


    def get_matrix_elements_tQST(self, threshold, diagonal=None):
        """
        returns a list of tuple of matrix elements for threshold Quantum
        State Tomography
        """
        if diagonal is None:
            diagonal = np.diag(self.rho)
            # normalize the trace to 1
            diagonal = diagonal / np.sum(diagonal)
            
        ret = []

        # the diagonal elements
        for i in range(diagonal.size):
            ret.append( (i,i,'r') )

        # the expected density matrix
        exp_rho = np.sqrt(np.outer(diagonal, diagonal))
        M = exp_rho >= threshold        
        # remove the lower triangle (and the diagonal) from the game
        lower_idx = np.tril_indices(len(diagonal))
        M[lower_idx] = False

        # the upper-triangle indices that exceed the threshold
        I,J = np.where( M == True)
        for i,j in zip(I,J):
            ret.append( (i,j,'r') )
            ret.append( (i,j,'i') )

        return(ret)


    def get_counts_from_mat_el(self, matrix_element_list):
        """
        Returns a set of projectors / measurements 
        for the given matrix_element_list
        """        
        self.projs = []
        for x in matrix_element_list:
            p = self.P.projector_from_matrix_element(*x)
            self.projs.append( p )

        self.projs = np.array(self.projs)
        p = self.projs.T
        self.counts = np.sum(np.real(np.conj(p) * ( self.rho @ p)), axis=0).flatten()

        return self.projs, self.counts


    def get_counts_tQST(self, threshold):
        mel = self.get_matrix_elements_tQST(threshold)
        return self.get_counts_from_mat_el_tQST( mel )


################################################################################

    
#if __name__ == '__main__':
    #P = PtQST.Projectors_tQST_qubit_local(2)
    #rho = dmt.density_matrix_GHZ(2)

    #C = Counts_tQST(P)
    #C.set_density_matrix(rho)
    
    #mel = C.get_matrix_elements_tQST(0.1)
    #print(mel)
    #for x in mel:
     #print(x, "-> |",P.projector_name_from_matrix_element(*x),">")

    #projs, counts = C.get_counts_from_mat_el(mel)
    #print(np.shape(projs))
    #print(projs)
    #print(np.shape(counts))
    #print(counts)
    

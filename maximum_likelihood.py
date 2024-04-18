#!/usr/bin/env python3

import sys
#sys.path.insert(1,"../")
#sys.path.insert(1,"../qudits")

from tqdm import *

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#import qudits_class as qd
from ml_models import *
import density_matrix_tool as dmt

def random_index(vals):
    idx = np.random.randint(vals.size)
    return idx

def index_of_min(vals):
    idx = np.argmin(vals)
    return idx

def index_of_max(vals):
    idx = np.argmax(vals)
    return idx

# dictionary of suggested next measurements
meas_dict = {'min': index_of_min,
             'max': index_of_max,
             'random': random_index}

class Maximum_likelihood_tomography:
    model = None  # model for the density matrix
    x0 = None     # last x0 used in the minimization

    # this can be obtained from counts
    projectors = None # Projectors used
    counts     = None # Counts measured

    # default options for the minimization
    # (this is actually a tolerance on the gradient)
    default_minimization_options = {
        'gtol': 1e-4,
        'maxiter': 1000
        }
    
    def __init__(self, n_qubit, model=model_triangular, model_params=None):
        #qd.Qudit.__init__(self,qudit_list)
        self.n_qubit = n_qubit
        self.dim = 2**(self.n_qubit)
        if model_params is None:
            self.model = model(self.dim)
        else:
            self.model = model(self.dim, model_params)            
        self.x0 = np.zeros(model.nvars)
        
        print("Tomography of", [2] * self.n_qubit, "using", self.model.description)
        
        if self.model.has_minimization == False:
            print("Defaulting to L-BFGS-B method with options",self.default_minimization_options)
        
    # for random number generation
    def set_seed(self, seed):
        np.random.seed(seed)

    
    def model_density_matrix(self):
        rho = self.model.x_to_density_matrix(self.x0, normalize=True)
        return(rho)


    def set_counts(self, projs, counts):
        self.projs = np.copy(projs)
        self.counts = np.copy(counts)

        
    def likelihood(self, x):        
        """
        the likelihood function.
        x is a vector of length self.model.nvars
        We get the measurement vector from the observations and produce the
        vector of the measurements and the expected rho from the model
        """        
        N    = self.counts
        vecs = self.projs

        n = self.model.counts(x,vecs)

        DN = (n-N)/(2.0*np.sqrt(n))

        idx = ~np.isfinite(DN)
        DN[idx] = 0.0
        
        L = np.sum(DN*DN)
        
        return(L)

    
    def likelihood_jac(self, x):
        # N , vecs = self.get_measurements()
        N    = self.counts  # measured values
        vecs = self.projs   # projectors
        
        n , jac  = self.model.counts_jacobian(x,vecs)
        
        n2 = n*n
        N2 = N*N        
        DN2 = (n2-N2)/(4*n2)
        
        idx = ~np.isfinite(DN2)
        DN2[idx] = 0.0
        
        # sum over the observations
        dL = np.sum(jac * DN2, axis=1)
        
        return(dL.flatten())

  
    def minimize(self, x0=None, preserve=False, method='L-BFGS-B', options=None):
        """
        Local miminization.
        If x0 is not provided it is set to random.
        preserve = True means to use as starting point the result of the
        last minimization
        """
        opts = self.default_minimization_options
        if options is not None:
            for i in options.keys():
                opts[i] = options[i]
        
        if x0 is None:
            A = np.sqrt(self.counts.max())
            x0 = A * np.random.uniform(-1,1,size=self.model.nvars)
            #print("minimization starting from",x0)
        if preserve is True and np.sum(np.abs(self.x0))> 1e-10:
            x0 = np.copy(self.x0)

        #print("minimization from ",x0)
        #res = sp.optimize.minimize(self.likelihood, x0, method=method)
        if self.model.has_minimization == False:
            res = sp.optimize.minimize(self.likelihood, x0, jac=self.likelihood_jac, method=method, options=opts)
        else:
            res = self.model.minimize(x0, self.projs, self.counts)
        
        self.x0 = np.copy(res.x)
        self.rho = self.model.x_to_density_matrix(self.x0, normalize=True)
        return(res)


    def minimize_pool(self, pool=8, preserve=False, method='L-BFGS-B', opts=None):
        """
        Minimizes starting from a random pool and takes the value with the
        minimum function
        """
        res = []
        fmin = []
        if preserve == True:
            res.append( self.minimize(method=method, preserve=True, options=opts))
            fmin.append( res[0].fun )
            
        for i in range(pool):
            r = self.minimize(method=method, preserve=False,options=opts)
            fmin.append( r.fun )
            res.append( r )
        idx = np.argmin( np.array(fmin) )

        self.x0 = np.copy( res[idx].x )
        self.rho = self.model.x_to_density_matrix(self.x0, normalize=True)            
        return res[idx]


####################################################################################


if __name__ == '__main__':
    
    import projectors_tQST_local as prj
    import counts_tQST as cnt

    nq  = 4
    ml = Maximum_likelihood_tomography(nq, model=model_triangular)
    #ml = Maximum_likelihood_tomography(nq, model = model_g5)
    P = prj.Projectors_tQST_qubit_local(nq)
    C = cnt.Counts_tQST(P)
    rho = dmt.density_matrix_W(nq)
    #rho = dmt.density_matrix_random(dimension = 2**nq, rank = 2**nq, number_of_zeros = 8)

    C.set_density_matrix( rho  )
    mel = C.get_matrix_elements_tQST(0.06)

    projs, counts = C.get_counts_from_mat_el(mel)
    print('Number of measurements:', len(counts))
    ml.set_counts( projs, counts )
    print("Minimizing")
    np.random.seed(0)
    res = ml.minimize()
    print(res.message)
    rho_target = ml.model_density_matrix()
    print("Fidelity: ",dmt.fidelity(rho, rho_target))

    print("")

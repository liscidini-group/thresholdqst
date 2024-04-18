#!/usr/bin/env python3

#
# MODELS FOR MAXIMUM LIKELIHOOD TOMOGRAPHY
#
# 1. the original maximum likelihood with triangular matrix
# 2. the Garbe5 model, approximating rho with n eigenvectors
#
# Each model is initialized with the dimension of the density matrix
# and provides the following variables | functions
#
# 1. nvars: the number of _real_ variables of the models
# 2. x_to_density_matrix(x): input: the real variables | output: the density matrix
# 3. counts(x, vec): inputs: the variables and an observation vector
#                    output: N(x) = <vec | rho(x) | vec>
# 4. counts_jac(x, vec): inputs: the real variables and an observation vector
#                        output: dN/dx (that is, the gradient)
#

import numpy as np

class model_triangular():
    name = "Triangular model"
    # rho = T^\dagger T - with T triangular
    dim = 0 # dimension of the density matrix
    nvars = 0
    T = None # rho = T^+ T | T[dim,dim]
    idx_re = None
    idx_in = None
    has_minimization = False

    
    def __init__(self, dim):
        self.dim = dim
        self.T = np.zeros([dim,dim], dtype=np.complex128)
        self.idx_re = np.tril_indices(dim,0)
        self.idx_im = np.tril_indices(dim,-1)
        self.nvars = self.idx_re[0].size + self.idx_im[0].size
        self.description = self.name+" with "+str(self.nvars)+" variables"


    def x_to_T(self, x):
        Nre = int(self.dim*(self.dim+1)/2)
        Nim = int(self.dim*(self.dim-1)/2)

        self.T[self.idx_re] = x[0:Nre]
        self.T[self.idx_im] = self.T[self.idx_im] + 1j * x[-Nim:]

        
    def x_to_density_matrix(self, x, normalize=False):
        self.x_to_T(x)
        rho = np.conj(self.T.T) @ self.T
        if normalize is True:
            rho = rho / np.trace(rho)
        return(rho)

    
    def counts(self, x, projector_vectors):
        # returns the counts using vecs
        # if vecs is a matrix vecs[nmeas, dim]
        # n is a vector of dimension nmeas
        V = projector_vectors.T # vectors as columns
        self.x_to_T(x)
        M = self.T @ V
        n = np.real(np.sum( np.conj(M) * M, axis=0))
        return(n.flatten())


    def counts_jacobian(self, x, projector_vectors):
        # returns the gradient (jacobian) of counts with respect to x
        # jac is [dim,dim,nmeas]
        # jac[i,j,k] = dnk / dT*_{ij}        
        V = projector_vectors.T      # measurement vectors as columns
        self.x_to_T(x)
        M = self.T @ V  # M is [dim,nmeas]
        n = np.real(np.sum( M * np.conj(M), axis=0))
        
        jac = M[:,np.newaxis,:] * np.conj(V) # jac[dim,dim,nmeas]
        jac_re = 2*np.real(jac[self.idx_re])
        jac_im = 2*np.imag(jac[self.idx_im])
        # ret is a vector [nvars, nmeas]
        ret = np.vstack([jac_re, jac_im])
        return(n.flatten(),ret)


class model_g5:
    name = "Garbe5 model"
    dim = 0   # dimension of the density matrix
    nvec = 0  # number of vectors used to approximate rho
    nvars = 0 # number of variables = 2*nvec*dim
    M = None  # rho = M^+ M, M[nvec,dim] | in M eigenvectors are rows
    idx_re = None
    idx_in = None
    has_minimization = False

    
    def __init__(self, dim,nvec=2):
        self.dim   = dim
        self.nvec  = nvec
        self.nvars = 2*dim*nvec
        self.M = np.zeros([nvec,dim], dtype=np.complex128)
        self.description = self.name+" with "+str(self.nvars)+" variables"


    def x_to_M(self, x):        
        N = int(self.nvars/2)
        self.M = x[0:N] + 1j * x[-N:]
        self.M = self.M.reshape(self.nvec,self.dim)


    def x_to_density_matrix(self, x, normalize=False):
        self.x_to_M(x)
        rho = np.conj(self.M.T) @ self.M
        if normalize is True:
            rho = rho / np.trace(rho)
        return(rho)
    
    def counts(self, x, projector_vectors):
        # returns the counts using vecs
        # if vecs is a matrix vecs[nmeas, dim]
        # n is a vector of dimension nmeas
        V = projector_vectors.T # vectors as columns
        self.x_to_M(x)
        A = self.M @ V
        n = np.real(np.sum( A * np.conj(A), axis=0))
        return(n)


    def reset_nvec(self, new_nvec):
        self.nvec = new_nvec
        self.nvars = 2*self.dim*new_nvec
        self.M = np.zeros([new_nvec,self.dim], dtype=np.complex128)


    def counts_jacobian(self, x, projector_vectors):
        # returns the gradient (jacobian) of counts with respect to x
        # jac is [dim,dim,nmeas]
        # jac[i,j,k] = dnk / dT*_{ij}        
        self.x_to_M(x)
        V = projector_vectors.T      # measurement vectors as columns
        A = self.M @ V  # A is [nvec,nmeas]
        n = np.real(np.sum( A * np.conj(A), axis=0))        
        jac = A[:,np.newaxis,:] * np.conj(V) # jac[nvec,dim,nmeas]
        jac_re = 2*np.real(jac)
        jac_im = 2*np.imag(jac)
        ret = np.vstack([jac_re, jac_im])

        # returns jac[nvars,nmeas]
        return( n, ret.reshape(self.nvars,-1) )


#from ctypes import *
#from scipy.optimize import OptimizeResult

##########################################################################################

if __name__ == "__main__":
    dim=4
    m = model_triangular(dim)
    #m = model_g5(dim)
    print("variables",m.nvars)
    x = np.random.randn(m.nvars) + 1j * np.random.randn(m.nvars)
    nvecs = 2
    vecs = np.random.randn(nvecs*dim).reshape(nvecs,dim)
    print(vecs.shape)
    rho = m.x_to_density_matrix(x)
    print("shape of rho",rho.shape)
    print(rho)
    print(m.counts(x,vecs))
    n, jac = m.counts_jacobian(x,vecs)

    h = 1e-4
    for i in range(m.nvars):
        dx = np.zeros(m.nvars)
        dx[i] = h
        xp = x + dx
        xm = x - dx
        cp = m.counts(xp,vecs)
        cm = m.counts(xm,vecs)
        der = (cp-cm)/(2*h)

        print(der, jac[i])

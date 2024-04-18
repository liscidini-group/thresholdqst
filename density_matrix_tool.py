#!/usr/bin/env python3

#
# routines for general density matrices
# 1. generation
# 2. distances
# 3. characteristics (purity, entropy, etc.)
# 4. 2D and 3D plotting

import numpy as np

import scipy as sp
from scipy.stats import unitary_group
from scipy.stats import dirichlet

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def density_matrix_from_state_vector(state_vector):
    rho = np.outer(state_vector, state_vector.conj())
    return(rho)


def density_matrix_GHZ(n_qubit):
    """
    returns the density matrix of the |GHZ> state of n qubits
    |GHZ> = (|0...0> + |1...1>) / sqrt(2)
    https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state
    """
    N = 2**n_qubit
    v = np.zeros(N, dtype=np.complex128)
    v[0]  = 1/np.sqrt(2) # |0...0>
    v[-1] = 1/np.sqrt(2) # |1...1>
    
    return np.outer(v, np.conj(v))

def density_matrix_W(n_qubit):
    """
    returns the density matrix of the |W> state of n qubits
    |W> = (|0...1> + |0...10> + |0...100> + |10...>) / sqrt(n)
    https://en.wikipedia.org/wiki/W_state
    """
    N = 2**n_qubit
    v = np.zeros(N, dtype=np.complex128)
    idx = np.power(2,range(n_qubit))
    v[idx] = 1.0/np.sqrt(n_qubit)
    
    return np.outer(v, np.conj(v))


def density_matrix_random(dimension, rank, number_of_zeros):
    
    U = unitary_group.rvs(dimension)
    # create random probabilities and normalize them
    D = np.zeros(dimension)
    
    rank = dimension if rank==0 else rank # the number of nonzero probabilities
    D[:rank] = np.random.rand(rank)
    D = D/np.sum(D)
    #D[:n] = dirichlet.rvs(alpha=[1]*n,size=1) # better distribution of probabilities

    # here it is, a random matrix in the Hilbert space
    rho = U @ (D.reshape(-1,1) * U.conj().T)

    idx = np.random.choice(dimension, size=number_of_zeros, replace=False) # elements to zero out
    rho[idx,:] = 0 # zero out rows
    rho[:,idx] = 0 # zero out cols

    return rho / np.trace(rho)


def frobenius_norm(matrix_A):
    """
    Frobenius norm of a matrix, that is the sum of the squares of the
    absolute values of A.
    """
    ret = np.real(np.sum(matrix_A*np.conj(matrix_A)))
    return(ret)

def frobenius_distance(matrix_A, matrix_B):
    return( np.sqrt(frobenius_norm(matrix_A - matrix_B)) )

def p_norm(matrix_A, p=2):
    """
    p norm of a matrix A.
    """
    val, vec = np.linalg.eigh(matrix_A)
    ret = np.sum(np.abs(val)**p)**(1/p)

    return(ret)


def purity(matrix_A):
    """
    Purity of a matrix A, that is tr(A^2).
    This is the same as the norm of the matrix.
    """
    return(frobenius_norm(matrix_A))

def matrix_sqrt(matrix_A):
    """
    Try to use scipy, and revert to hand-made program if it fails.
    Because it fails with some matrices on "old" x86-64 systems.
    """
    ret = sp.linalg.sqrtm(matrix_A).astype(np.complex128)

    if np.all(np.isfinite(ret)) == False:
        val, vec = np.linalg.eigh(matrix_A)
        ALMOST_ZERO = 1e-6
        val = np.abs(val)
        val[val < ALMOST_ZERO] = 0.0
        val = np.sqrt(val)
        ret = vec @ (val.reshape(-1,1) *  np.conj(vec.T))

    assert np.all(np.isfinite(ret))
    return(ret)
        
def fidelity(matrix_A, matrix_B):
    """
    Returns tr( sqrt( sqrt(A) B sqrt(A) )
    If B is none, then we take the density matrix of this system
    """
    # sqrt_A = sp.linalg.sqrtm(A) 
    # sqrt_M = sp.linalg.sqrtm(sqrt_A @ (B @ sqrt_A))
    sqrt_A = matrix_sqrt(matrix_A) 
    sqrt_M = matrix_sqrt(sqrt_A @ (matrix_B @ sqrt_A))
    # fidelity should be real
    fi = np.real(np.trace(sqrt_M))
    fi = fi if 0<= fi <= 1 else 1
    
    return( fi )


def von_neumann_entropy(matrix_A):
    """
    Calculates the Von Neumann entropy of a matrix, 
    defined as the sum of the entropies opf the eigenvalues
    """

    val, vec = np.linalg.eigh(matrix_A)
    val = np.ma.masked_inside(val,-10**-10,10**-10).filled(0)

    with np.errstate(divide='ignore'):
        lg = np.log2(val)
    
    lg[np.isneginf(lg)]=0
    vne = np.ma.masked_inside(-val*lg,-10**-10,10**-10).filled(0)

    return ( np.sum(vne) )

def kullback_leibler_divergence(P,Q):
    """
    The Kullback-Leibler divergence between P and Q
    KL(A,B) = tr(P log(P) - log(Q))
    
    If P and Q are statistical operators
    This measures the "distance" between the distribution Q and P.
    """
    valP, vecP = np.linalg.eigh(P)
    idx = (valP <= 0.0)
    log_valP = np.log2(np.abs(valP))
    log_valP[idx] = 0.0
    
    valQ, vecQ = np.linalg.eigh(Q)
    idx = (valQ <= 0.0)
    log_valQ = np.log2(np.abs(valQ))
    log_valQ[idx] = 0.0

    lP = vecP @ (log_valP.reshape(-1,1) * np.conj(vecP.T))
    lQ = vecQ @ (log_valQ.reshape(-1,1) * np.conj(vecQ.T))    

    return np.real(np.trace(P @ (lP - lQ)))

def kullback_leibler_symmetric_divergence(P, Q, weight=0.5):
    """
    The symmetric Kullback-Leibler divergence between P and Q
    KL(A,B) = weight * (KL(P,Q) + KL(Q,P))
    defaults to weight = 0.5
    """
    KLPQ = kullback_leibler_divergence(P, Q)
    KLQP = kullback_leibler_divergence(Q, P)
    ret = weight * (KLPQ + KLQP)
    return(ret)

def get_rank(matrix_V):
    """
    Returns the rank of the matrix that checks if a set of projectors are independent
    On input, V is a matrix that has the projector |p> as rows
    """
    M = matrix_V @ np.conj(matrix_V.T)
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


from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d


def plot_density_matrix_3D(density_matrix):
    
    nofqubits = int(np.log2(density_matrix.shape[0]))
    
    _x = np.arange(density_matrix.shape[0])
    _y = np.arange(density_matrix.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    y, x = _xx.ravel(), _yy.ravel()
    
    # we plot the absolute value
    top = np.abs(density_matrix).flatten()
    bottom = 0.0 #np.zeros_like(top).flatten()
    w=0.8
    d=0.8
    
    mpl_cyclic_names = ['twilight','twilight_shifted','hsv']
    oth_cyclic_names = ['cmocean_phase','hue_L60','erdc_iceFire','nic_Edge','colorwheel','cyclic_mrybm','cyclic_mygbm']

    cmap_name = 'twilight_shifted'
    trans = None
    z_axis_res = 3
    colBar = True
    filename = None

    if nofqubits == 1 or nofqubits == 2 or nofqubits == 3:
        ticks_reduction_factor = 1
    else:
        ticks_reduction_factor = 2**(nofqubits - 3)

    if cmap_name in mpl_cyclic_names:
       cm_map = plt.colormaps[cmap_name]
    elif cmap_name in oth_cyclic_names:
       # load the requested color map; we need to change this to avoid hard coded paths
       script_dir = Path(__file__).resolve().parent
       path = script_dir.parent / 'colormaps' / 'cyclic'
       cmap_name = cmap_name + ".txt"
       cm_data = np.loadtxt(path / cmap_name)
       cm_map = LinearSegmentedColormap.from_list(cmap_name, cm_data)
    else:
       cm_map = plt.colormaps['twilight_shifted']

    # the quantity used for the colormap
    phase = np.angle(density_matrix).flatten() * 180.0/np.pi
    norm = plt.Normalize(-180, 180)              
    cols = cm_map(norm(phase))     
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #ax.xaxis.set_major_formatter(StrMethodFormatter(bin_strf))
    ax.xaxis.set_ticks(np.arange(0, 2**nofqubits, ticks_reduction_factor))
    x_lbls = ax.get_xticks().tolist()
    #x_lbls[-1]=x_lbls[-1]-1
    for i in range(len(x_lbls)):
       x_lbls[i] = np.binary_repr(x_lbls[i], width = nofqubits)
    ax.set_xticklabels(x_lbls)
    ax.tick_params(axis='x', labelrotation = -25)

    #ax.yaxis.set_major_formatter(StrMethodFormatter(bin_strf))
    ax.yaxis.set_ticks(np.arange(0, 2**nofqubits, ticks_reduction_factor))
    y_lbls = ax.get_yticks().tolist()
    #y_lbls[-1]=y_lbls[-1]-1
    for i in range(len(y_lbls)):
       y_lbls[i] = np.binary_repr(y_lbls[i], width = nofqubits)
    ax.set_yticklabels(y_lbls)
    ax.tick_params(axis='y', labelrotation = 47.5)

    #mask = top.nonzero()
    mask = np.ones_like(x,dtype=bool) # plot everything
    p = ax.bar3d(x[mask],y[mask],bottom,w,d,top[mask], color=cols[mask], alpha=trans, cmap=cm_map)
    ax.view_init(elev=40, azim=20)
    ax.set_title('Modulus of density matrix, colored by phase')
    #print("x:",x)
    # plot circles where measures have to be done
    #if circles is not None:
        #for c in circles:
            #addCircle(ax,c[0],c[1])
    
    # plot the colorbar
    if colBar == True: 
        #colorMap = plt.cm.ScalarMappable(cmap=cmap_name)    
        #colorMap.set_array([-1,0,1]) # phase)
        cbar = fig.colorbar(p, cmap=cm_map, pad=0.2, ticks=[0,1/8,1/4,3/8,1/2,5/8,3/4,7/8,1])
        cbar.ax.set_yticklabels(['-$\pi$', '-3$\pi$/4', '-$\pi$/2 ', '-$\pi$/4', '0', '$\pi$/4', '$\pi$/2', '3$\pi$/4', '$\pi$',])

    z_lbls = ax.get_zticks().tolist()
    for i in range(len(z_lbls)):
       z_lbls[i] = format(z_lbls[i],'.'+str(z_axis_res)+'f')
    z_lbls[0] = ''
    ax.set_zticklabels(z_lbls)
    ax.tick_params(axis='z', labelrotation = 45)
    
    plt.xticks(va='center', ha='left')
    plt.yticks(va='center', ha='right')

    if filename:
        plt.savefig(filename)
    
    plt.show()


def plot_density_matrix_2D(density_matrix):

    import matplotlib.ticker as mticker

    nofqubits = int(np.log2(density_matrix.shape[0]))

    if nofqubits == 1 or nofqubits == 2 or nofqubits == 3:
        ticks_reduction_factor = 1
    else:
        ticks_reduction_factor = 2**(nofqubits - 3)
    
    fig, ax = plt.subplots(1,2)
    mod   = np.abs(density_matrix)
    #phase = np.angle(rho) * 180.0/np.pi
    phase = np.angle(density_matrix) / np.pi
    
    im1 = ax[0].imshow(mod)
    ax[0].set_title("Modulus")
    xticks = np.arange(0, 2**nofqubits, ticks_reduction_factor)
    xlabels = xticks.tolist()
    for i in range(len(xlabels)):
       xlabels[i] = np.binary_repr(xlabels[i], width = nofqubits)
    ax[0].set_xticks(xticks, xlabels, rotation=270)
    fig.colorbar(im1,ax=ax[0],shrink=0.7)

    yticks = np.arange(0, 2**nofqubits, ticks_reduction_factor)
    ylabels = yticks.tolist()
    for i in range(len(ylabels)):
       ylabels[i] = np.binary_repr(ylabels[i], width = nofqubits)
    ax[0].set_yticks(yticks, ylabels)
    
    cmap_name='twilight_shifted'
    #norm = plt.Normalize(-180, 180)
    #print("min/max",np.min(phase),np.max(phase))
    #cmap = plt.colormaps[cmap_name]
    #cols = cmap(norm(phase))
    #cols = cmap(phase)              
    im2 = ax[1].imshow(phase,cmap=cmap_name, vmin=-1, vmax=1)    
    ax[1].set_title("Phase (rad)")
    xticks = np.arange(0, 2**nofqubits, ticks_reduction_factor)
    xlabels = xticks.tolist()
    for i in range(len(xlabels)):
       xlabels[i] = np.binary_repr(xlabels[i], width = nofqubits)
    ax[1].set_xticks(xticks, xlabels, rotation=270)

    yticks = np.arange(0, 2**nofqubits, ticks_reduction_factor)
    ylabels = yticks.tolist()
    for i in range(len(ylabels)):
       ylabels[i] = np.binary_repr(ylabels[i], width = nofqubits)
    ax[1].set_yticks(yticks, ylabels)
    
    fig.colorbar(im2,ax=ax[1],shrink=0.7, ticks=[-1, -3/4, -1/2, -1/4, 0, 1/4, 1/2, 3/4, 1], format=mticker.FixedFormatter(['-$\pi$', '-3$\pi$/4', '-$\pi$/2 ', '-$\pi$/4', '0', '$\pi$/4', '$\pi$/2', '3$\pi$/4', '$\pi$']))
    #cbar = fig.colorbar(im2, cmap=cmap_name, pad=0.2, ticks=[0,1/8,1/4,3/8,1/2,5/8,3/4,7/8,1])
    #cbar.set_yticklabels(['-$\pi$', '-3$\pi$/4', '-$\pi$/2 ', '-$\pi$/4', '0', '$\pi$/4', '$\pi$/2', '3$\pi$/4', '$\pi$',])
    
    plt.tight_layout()
    plt.show()

################################################################################################

#if __name__ == "__main__":

    #v = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    #rho = density_matrix_from_state_vector(v)
    #print(rho)


    #rhoGHZ = density_matrix_GHZ(n_qubit = 3)
    #print(rhoGHZ)


    #rhoW = density_matrix_W(n_qubit = 4)
    #print(np.shape(rhoW))


    #rho = density_matrix_random(dimension = 4, rank = 4, number_of_zeros = 0)
    #plt.imshow(np.abs(rho))
    #plt.colorbar()
    #plt.show()
    #for i in range(len(rho)):
    #    for j in range(len(rho)):
    #        if j < i:
    #            print(np.square(np.abs(rho[i,j])) - np.real(rho[i,i]) * np.real(rho[j,j]) )

    
    #rho1 = density_matrix_random(dimension = 4, rank = 4, number_of_zeros = 0)
    #rho2 = density_matrix_random(dimension = 4, rank = 4, number_of_zeros = 0)
    #print(frobenius_norm(rho1))
    #print(frobenius_distance(rho1, rho2))
    #print(frobenius_distance(rho2, rho1))
    #print(p_norm(rho1, p=3))

    #print(von_neumann_entropy(rho))

    #print(get_rank(rho))

    #print(kullback_leibler_divergence(rho1, rho2))
    #print(kullback_leibler_divergence(rho2, rho1))
    #print(kullback_leibler_symmetric_divergence(rho1, rho2))

    #plot_density_matrix_3D(rho)
    #plot_density_matrix_2D(rho)

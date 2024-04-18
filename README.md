# Threshold Quantum State Tomography (tQST)
### Daniele Binosi, Giovanni Garberoglio, Diego Maragnano, Maurizio Dapor, and Marco Liscidini

This Python package implements Threshold Quantum State Tomography (tQST), a novel approach for quantum state tomography that reduces the number of measurements required to reconstruct the density matrix of a single- or multi-qubit system.  
Details about the protocol can be found at [Binosi, D., Garberoglio, G., Maragnano, D., Dapor, M., & Liscidini, M. (2024). *Threshold Quantum State Tomography.* arXiv preprint arXiv:2401.12864.](https://arxiv.org/abs/2401.12864).  

- [Threshold Quantum State Tomography (tQST)](#threshold-quantum-state-tomography-tqst)
    - [Daniele Binosi, Giovanni Garberoglio, Diego Maragnano, Maurizio Dapor, and Marco Liscidini](#daniele-binosi-giovanni-garberoglio-diego-maragnano-maurizio-dapor-and-marco-liscidini)
- [Notation](#notation)
- [How to install](#how-to-install)
- [Set up the experiment](#set-up-the-experiment)
- [Find the projective measurements](#find-the-projective-measurements)
- [Perform tomography and evaluate the reconstruction](#perform-tomography-and-evaluate-the-reconstruction)
- [Final remarks](#final-remarks)

---

<a id="notation"></a>
# Notation
Before proceeding, we provide the relation between the polarization notation, used in the protocol to identify the projective measurements, and the notation based on the eigenvectors of the Pauli matrices. The vectors are represented in the computational basis $\lvert 0 \rangle, \lvert 1 \rangle$.  
$\lvert H \rangle = \lvert 0 \rangle = \binom{1}{0} = \lvert z_+ \rangle$  
$\lvert V \rangle = \lvert 1 \rangle = \binom{0}{1} = \lvert z_- \rangle$  
$\lvert D \rangle = \frac{1}{\sqrt{2}} \binom{1}{1} = \lvert x_+ \rangle$  
$\lvert A \rangle = \frac{1}{\sqrt{2}} \binom{1}{-1} = \lvert x_- \rangle$  
$\lvert R \rangle = \frac{1}{\sqrt{2}} \binom{1}{i} = \lvert y_+ \rangle$  
$\lvert L \rangle = \frac{1}{\sqrt{2}} \binom{1}{-i} = \lvert y_- \rangle$  

<a id="how-to-install"></a>
 # How to install

 We provide a Python package to generate and manipulate the density matrix of single- and multi-qubit systems, and to perform tomography experiments according to tQST protocol.  
 1. Download all the files in the Github repository.
 2. If not already installed, install the `prettytable` package following the instructions at [this link](https://pypi.org/project/prettytable/).
 3. Open a new Python script and import the required dependencies with the following command: `from tqst_class import *`  

Now we have all the necessary tools to deal with tQST.  

<a id="setup-the-experiment"></a>
# Set up the experiment

In this section we learn how to prepare the stage for a tQST experiment.  
The notebook `example.ipynb` reproduces the code presented here, together with a 2-qubit example, and can be used as a template for other tQST experiments. Click [here](https://github.com/liscidini-group/thresholdqst/blob/main/example.ipynb) to access the notebook.
After importing the necessary dependencies, we start by creating an instance of the tQST class, providing the number of qubits (in this case 3) as argument:
```{python}
tomo = tQST(3)
```
Then we can retrieve the number of qubits by calling the `get_num_of_qubits()` method:
```{python}
nq = tomo.get_num_of_qubits()
```
The first step of the tQST protocol is to measure of the diagonal elements of the density matrix.  
The diagonal counts are ordered in the same way as the computational basis. In the case of 3 qubits, according to the definition of $\lvert 0 \rangle$ and $\lvert 1 \rangle$, the ordering is:
- $\lvert HHH \rangle \rightarrow 0$
- $\lvert HHV \rangle \rightarrow 1/3$
- $\lvert HVH \rangle \rightarrow 1/3$
- $\lvert HVV \rangle \rightarrow 0$
- $\lvert VHH \rangle \rightarrow 1/3$
- $\lvert VHV \rangle \rightarrow 0$
- $\lvert VVH \rangle \rightarrow 0$
- $\lvert VVV \rangle \rightarrow 0$

 We set the diagonal counts by properly ordering and storing them in a list and calling the `set_diagonal_counts()` method, with the list just created as argument.  
```{python}
diagonal = [0., 333, 333, 0., 333, 0., 0., 0.]
tomo.set_diagonal_counts(diagonal)
```
Then we call the method `get_diagonal_counts()` and store the projectors and the measured diagonal elements in a dictionary as key-value pairs.   
We now set the threshold for the experiment.  This can be done via the `set_threshold()` method, with argument the chosen value of the threshold:
```{python}
threshold = 0.01
tomo.set_threshold(threshold)
```

<a id="find-projective-measurements"></a>
# Find the projective measurements

After providing and setting the diagonal counts and the threshold, we need to know which measurements to perform. This is done with the following line of code:
 ```{python}
proj_to_meas = tomo.get_projectors_to_measure()
```
When called, this method prints out a table with two columns, showing the correspondence between density matrix elements and projective measurements.  
The method returns also a list called `proj_to_meas`, containing the name of the projectors to be measured.

<a id="perform-tomography-and-evaluate-the-reconstruction" ></a>
# Perform tomography and evaluate the reconstruction

Once the measurements are known, we collect the outcomes and put them into the dictionary `tdict`. We can do this with the Python dictionary method `setdefault`, with two arguments: a string with the name of the projector, and the corresponding measurement outcomes.

```{python}
tdict.setdefault('HRR', 333)
tdict.setdefault('HRD', 167)
tdict.setdefault('RHR', 333)
tdict.setdefault('RHD', 167)
tdict.setdefault('RRH', 333)
tdict.setdefault('RDH', 167)
```

Now we can reconstruct the density matrix starting from the measurement outcomes.  
We first call the method `read_tomo_dictionary()`, with argument the dictionary `tdict`. This returns two arrays, one with the representation of the measured projectors in the computational basis, and the other one with the measurement outcomes.  
Then, we set the model of the density matrix that will be used in the reconstruction process. In this case we use a triangular model, that ensures the reconstructed density matrix to be positive, Hermitian, and with trace one:  
$\rho = \frac{T^{\dagger} T}{\text{Tr} \left(T T^{\dagger} \right)}$,  
with $T$ a lower (or upper) triangluar matrix.  
Finally, we call the method `get_density_matrix`, which returns the reconstructed density matrix.

```{python}
projs, counts = tomo.read_tomo_dictionary(tdict)

tomo.set_density_matrix_model(model_triangular)

rho_rec = tomo.get_density_matrix(projs, counts)
```

We can visualize a 3D plot of the reconstructed density matrix with the following command:

```{python}
dmt.plot_density_matrix_3D(rho_rec)
```

Finally, if we know the state that the device is supposed to generate, we can evaluate the quality of the reconstruction process by computing the fidelity between the two density matrices:
$F \left( \rho_t, \rho_r\right) = \text{Tr} \left[ \sqrt{\sqrt{\rho_t} \rho_r \sqrt{\rho_t}} \right]$, with $\rho_t$ the target density matrix generated by the device, and $\rho_r$ the reconstructed density matrix.  
Suppose the target state is a 3-qubit W state. Then we can generate the corresponding density matrix and then compute the fidelity with the reconstructed state.  
The following lines of code do the job:
```{python}
rho_target = dmt.density_matrix_W(3)

F = dmt.fidelity(rho_rec, rho_target)

print(F)
```

<a id="final-remarks" ></a>
# Final remarks

We have completed our first tQST experiment for a 3-qubit system. There are still a few things to mention
The user can find some notebooks in the Github repository.    
The three tutorial notebooks explore in more detail all the topics covered in this document.  
In particular, `tutorial1.ipynb` (click [here](https://github.com/liscidini-group/thresholdqst/blob/main/tutorial1.ipynb) to access the notebook) explains how to generate and visualize other kinds of density matrices.   
`tutorial2.ipynb`, which you can find at this [link](https://github.com/liscidini-group/thresholdqst/blob/main/tutorial2.ipynb), focuses on how to find the projective measurements, given the threshold and the diagonal counts.  
`tutorial3.ipynb` provides several examples of tQST experiments with simulated data, and an alternative way to provide the experimental measurement outcomes based on the use of Numpy array instead of Python dictionary. At [this link](https://github.com/liscidini-group/thresholdqst/blob/main/tutorial3.ipynb) you can find the corresponding notebook.  
Finally, the notebook `bell_state_test.ipynb` implements tQST using experimental data provided in this [paper](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.64.052312). This is the [link to the notebook](https://github.com/liscidini-group/thresholdqst/blob/main/bell_state_test.ipynb).

from tqst_class import *

#print('Import ok')

tomo = tQST([2,2])
#tomo = tQST([])
nq = tomo.get_num_of_qubits()


diagonal = np.array([0.25]*4)
#diagonal = np.array([0.25]*5)
tomo.set_diagonal_counts(diagonal)

threshold = 0.5
#threshold = -0.1
tomo.set_threshold(threshold)

###FIRST WAY TEST ###

#P, C, projs = tomo.get_projectors_to_measure()
#print(np.shape(projs))

#counts = [0.25]*4
#counts = [0.25]*5

#tomo.set_counts(projs, counts)

#projs, counts = tomo.get_projs_and_counts()

###   ###


### SECOND WAY TEST ###

tdict = tomo.get_projectors_and_tomo_dictionary()
tdict.setdefault('HH', 0.5) # setdefualt returns the value if the key is already in the dict, otherwise it creates the key-value pair. No need to check later.
tdict.setdefault('HV', 0.5)
tdict.setdefault('VH', 0.5)
tdict.setdefault('VV', 0.5)

#print(tdict)

projs, counts = tomo.read_tomo_dictionary(tdict)
#print(np.shape(projs))
#print(counts)

###   ###

tomo.set_density_matrix_model(model_triangular)

rho_rec = tomo.get_density_matrix(projs, counts)
print(rho_rec)

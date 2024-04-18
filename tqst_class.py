import density_matrix_tool as dmt
import projectors_tQST_local as prj
import counts_tQST as cnt
from maximum_likelihood import *
import prettytable as pt

class tQST():
  #print('Class created')

  def __init__(self, n_qubit):
    self.nq = n_qubit
    assert n_qubit > 0, 'Number of qubits must be at least 1, instead got {}'.format(self.nq)
    print('Number of qubits set to {}.'.format(self.nq))

  def get_num_of_qubits(self): 
    return self.nq
    
  def set_diagonal_counts(self, diag):
    assert len(diag) == 2**self.nq, 'Length of diagonal must be equal to {}, instead was {}'.format(2**self.nq, len(diag))
    self.diag = np.real(diag) / np.sum(np.real(diag))
    
    self.diag_dict = {}
    for i in range(len(self.diag)):
       self.diag_dict[str(np.binary_repr(i, width=self.nq)).replace('0', 'H').replace('1', 'V')] = self.diag[i]
    
    print('Diagonal counts are now set.')

  def get_diagonal_counts(self):
    return self.diag_dict

  def set_threshold(self, threshold):
    assert threshold >= 0, 'Threshold must be greater than, or equal to, 0, got {}'.format(threshold)
    self.threshold = threshold
    print('The threshold is now set to {}.'.format(self.threshold))

##### FIRST WAY TO PROJECTORS AND COUNTS #####
    
  def get_projs_to_measure(self):
    self.P = prj.Projectors_tQST_qubit_local(self.nq)
    self.C = cnt.Counts_tQST(self.P)

    mel = self.C.get_matrix_elements_tQST(self.threshold, self.diag)

    print('These are the projectors you have to measure, given the provided diagonal and threshold.')
    table = pt.PrettyTable(["Matrix element", "Projector"])
    for x in mel:
        if self.P.projector_name_from_matrix_element(*x) in self.diag_dict.keys():
            pass
        else:
            table.add_row([x, '|'+self.P.projector_name_from_matrix_element(*x)+'>'])
    print(table)

    offd_projs = []
    for x in mel:
        if self.P.projector_name_from_matrix_element(*x) in self.diag_dict.keys():
            pass
        else:
            offd_projs.append(self.P.projector_name_from_matrix_element(*x))
    
    return offd_projs

  def set_projs_and_counts(self, offd_projs, offd_counts):
    assert len(offd_projs) == len(offd_counts), 'Off-diagonal projectors has length {}, off-diagonal counts has length {}, but they must have the same length.'.format(len(offd_projs), len(offd_counts))
    
    self.projs = []
    for key in self.diag_dict.keys():
        self.projs.append(prj.extended_projector_from_string(key))
    for i in offd_projs:
        self.projs.append(prj.extended_projector_from_string(i))
    self.projs = np.asarray(self.projs)
    
    self.counts = []
    for value in self.diag_dict.values():
        self.counts.append(value)
    for i in offd_counts:
        self.counts.append(i)

    self.counts = np.asarray(self.counts)

  def get_projs_and_counts(self):
    return self.projs, self.counts

#####

##### SECOND WAY TO PROJECTORS AND COUNTS #####

  def get_projectors_to_measure(self):
    self.P = prj.Projectors_tQST_qubit_local(self.nq)
    self.C = cnt.Counts_tQST(self.P)

    mel = self.C.get_matrix_elements_tQST(self.threshold, self.diag)

    print('These are the projectors you have to measure, given the provided diagonal and threshold.')
    table = pt.PrettyTable(["Matrix element", "Projector"])
    for x in mel:
        if self.P.projector_name_from_matrix_element(*x) in self.diag_dict.keys():
            pass
        else:
            table.add_row([x, '|'+self.P.projector_name_from_matrix_element(*x)+'>'])
    print(table)

    projectors_to_measure = []
    for x in mel:
        if self.P.projector_name_from_matrix_element(*x) in self.diag_dict.keys():
            pass
        else:
            projectors_to_measure.append(self.P.projector_name_from_matrix_element(*x))
    
    
    return projectors_to_measure

  def read_tomo_dictionary(self, tomo_dict):

    assert len(tomo_dict) >= 2**self.nq, 'Missing diagonal measurements.'
          
    projs = []
    for key in tomo_dict.keys():
            projs.append(prj.extended_projector_from_string(key))
            #projs.append(prj.Projectors_tQST_qubit(self.nq).projector_from_string(key))
    projs = np.vstack(projs)

    counts = []
    for value in tomo_dict.values():
            counts.append(value)
    counts = np.asarray(counts)

    return projs, counts

#####

  def set_density_matrix_model(self, model):
    assert model == model_triangular or model == model_g5 or model == model_gsl, 'Choose a model between model_triangular, model_g5, model_gsl.'
    self.model = model
    print('The model for density matrix reconstruction is now set.')

  def get_density_matrix(self, projs, counts):
    ml = Maximum_likelihood_tomography(self.nq, model=self.model)
    ml.set_counts(projs, counts)
    print('The projectors and the corresponding counts are set. Ready to perform QST.')
    np.random.seed(0)
    res = ml.minimize()
    rho_reconstructed = ml.model_density_matrix()

    return rho_reconstructed

######################################################################################################

if __name__ == "__main__":

  tomo = tQST(2)
  nq = tomo.get_num_of_qubits()
  #print(nq)

  diagonal = np.array([0.5, 0, 0, 0.5])
  tomo.set_diagonal_counts(diagonal)
  print(tomo.get_diagonal_counts())

  threshold = tomo.set_threshold(0.2)
  
#####
  #P, C, projs = tomo.get_projectors_to_measure()
  #print(np.shape(projs))

  #counts = np.array([0.5, 0, 0, 0.5, 0.5, 0.25])
  #counts = counts * 1e3
  #tomo.set_counts(projs, counts)
#####


#####

  tdict = tomo.get_projectors_and_tomo_dictionary()
  tdict.setdefault('HH', 0.5)
  tdict.setdefault('HV', 0)
  tdict.setdefault('VH', 0)
  tdict.setdefault('VV', 0.5)
  tdict.setdefault('DD', 0.5)
  tdict.setdefault('DR', 0.25)

  projs, counts = tomo.read_tomo_dictionary(tdict)

#####

  tomo.set_density_matrix_model(model_triangular)

  rho_rec = tomo.get_density_matrix(projs, counts)

  dmt.plot_density_matrix_3D(rho_rec)

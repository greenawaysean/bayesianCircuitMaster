import numpy as np
from qutip import (sigmax, sigmay, sigmaz, qeye, basis, gate_expand_1toN, Qobj, tensor,
                   snot)
from scipy.sparse import csr_matrix, csc_matrix
import copy
import itertools


class ProbDist:
    """Base class for generating probability distributions for fidelity estimation
    """

    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.pauli_strings = self.pauli_permutations()

    def get_probabilities(self):
        raise NotImplementedError

    def pauli_permutations(self):
        return [''.join(i) for i in itertools.product('0123', repeat=self.nqubits)]


class ChiProbDist(ProbDist):
    """Probability distribution based off the work in 10.1103/PhysRevLett.106.230501
    """

    def __init__(self, nqubits: int, U: Qobj):
        super().__init__(nqubits)
        self.tens_ops = self.get_tensored_ops()
        self.tens_states = self.get_tensored_states()
        self.U = csc_matrix(U.full())
        self.probabilities, self.chi_dict = self.get_probs_and_chis()

    def get_probs_and_chis(self):
        d = 2**self.nqubits
        input_states, observables = self.generate_states_observables()
        probabilities = {}
        chi_dict = {}
        for _state_idx in self.pauli_strings:
            for _obs_idx in self.pauli_strings:
                _state = input_states[_state_idx]
                _obs = observables[_obs_idx]
                # chi defined as (U*W_k*U^\dagger*W_k') for Pauli ops W_k, W_k'
                _trace = np.dot(self.U, _state)
                _trace = np.dot(_trace, np.conj(np.transpose(self.U)))
                _trace = np.dot(_trace, _obs)
                _trace = _trace.diagonal()
                chi = _trace.sum()/np.sqrt(d)
                chi_dict[_state_idx, _obs_idx] = chi
                probabilities[(_state_idx, _obs_idx)] = (1/d**2)*np.abs(chi**2)
        return probabilities, chi_dict

    def generate_states_observables(self):
        input_states = {}
        observables = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            observables[_state] = copy.deepcopy(_op)
            input_states[_state] = copy.deepcopy(_op)

        return input_states, observables

    def get_tensored_ops(self):
        tens_ops = []
        for _state in self.pauli_strings:
            _ops = []
            for i in _state:
                if i == '0':
                    _ops.append(qeye(2))
                if i == '1':
                    _ops.append(sigmax())
                if i == '2':
                    _ops.append(sigmay())
                if i == '3':
                    _ops.append(sigmaz())
            _op = tensor(_ops)
            tens_ops.append(csc_matrix(_op.full()))

        return tens_ops

    def get_tensored_states(self):
        tens_ops = []
        for _state in self.pauli_strings:
            _ops = []
            for i in _state:
                if i == '0':
                    _ops.append(qeye(2))
                if i == '1':
                    _ops.append(sigmax())
                if i == '2':
                    _ops.append(sigmay())
                if i == '3':
                    _ops.append(sigmaz()
            _op=tensor(_ops)
            tens_ops.append(csc_matrix(_op.full()))

        return tens_ops


class AmpProbDist(ProbDist):
    """Probability distribution based on the amplitudes of the observables.
    TODO: implement this
    """

    def __init__(self):
        pass

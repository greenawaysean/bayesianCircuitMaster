import numpy as np
from qutip import sigmax, sigmay, sigmaz, qeye, basis, gate_expand_1toN, Qobj, tensor
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
        self.U = U

    def get_probabilities(self):
        d = 2**self.nqubits
        input_states = self.generate_input_states()
        observables = self.generate_observables()
        probabilities = {}
        for _state_idx in self.pauli_strings:
            for _obs_idx in self.pauli_strings:
                _state = input_states[_state_idx]
                _obs = observables[_obs_idx]
                chi = (self.U * _state * self.U.dag() * _obs).tr()
                probabilities[(_state_idx, _obs_idx)] = np.abs((1/d**3)*chi**2)
        print('probs:', np.sum([probabilities[k] for k in probabilities]))
        return probabilities

    def get_chi_dict(self):
        d = 2**self.nqubits
        input_states = self.generate_input_states()
        observables = self.generate_observables()
        chi_dict = {}
        for _state_idx in self.pauli_strings:
            for _obs_idx in self.pauli_strings:
                _state = input_states[_state_idx]
                _obs = observables[_obs_idx]
                chi = np.abs((self.U * _state * self.U.dag() * _obs).tr())
                chi_dict[_state_idx, _obs_idx] = chi

        return chi_dict

    def generate_input_states(self):
        init_state = tensor([basis(2, 0)] * self.nqubits)
        input_states = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            _copy = copy.deepcopy(init_state)
            state = _op * _copy
            input_states[_state] = state * state.dag()

        return input_states

    def generate_observables(self):
        observables = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            observables[_state] = _op

        return observables

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
            tens_ops.append(_op)

        return tens_ops


class AmpProbDist(ProbDist):
    """Probability distribution based on the amplitudes of the observables.
    TODO: implement this
    """

    def __init__(self):
        pass

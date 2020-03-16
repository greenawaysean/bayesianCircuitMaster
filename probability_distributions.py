import numpy as np
from qutip import sigmax, sigmay, sigmaz, qeye, basis, gate_expand_1toN, Qobj, tensor
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
                chi = _trace.sum()
                chi_dict[_state_idx, _obs_idx] = chi
                probabilities[(_state_idx, _obs_idx)] = np.abs((1/d**3)*chi**2)
        return probabilities, chi_dict

    def generate_states_observables(self):
        init_state = csc_matrix(tensor([basis(2, 0)] * self.nqubits).full())
        input_states = {}
        observables = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            observables[_state] = copy.deepcopy(_op)
            _copy = copy.deepcopy(init_state)
            state = np.dot(_op, _copy)
            input_states[_state] = np.dot(state, np.conj(np.transpose(state)))

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


class ZetaProbDist(ProbDist):
    """Probability distribution based off the work in 10.1103/PhysRevLett.106.230501 but built using a spanning set of trace-1 hermitian matrices rather than pauli ops
    """

    def __init__(self, nqubits: int, U: Qobj):
        super().__init__(nqubits)
        self.tens_ops = self.get_tensored_ops()
        self.tens_states = self.get_tensored_states()
        self.U = csc_matrix(U.full())
        self.w_rho_coeffs = [
            (1, 1, 0, 0),
            (-1, -1, 0.5, 0),
            (-1, -1, 0, 0.5),
            (1, -1, 0, 0)
        ]
        self.alphas = self.get_alphas()
        self.u_trace = self.generate_u_trace()
        self.probabilities, self.chi_dict = self.get_probs_and_chis()

    def get_probs_and_chis(self):
        d = 2**self.nqubits
        observables = self.generate_observables()
        input_states = self.generate_observables()
        probabilities = {}
        chi_dict = {}
        gam = {}
        for n in self.pauli_strings:
            for j in self.pauli_strings:
                _summand = []
                for k in self.pauli_strings:
                    _summand.append(self.alphas[(k, n)]*self.u_trace[(k, j)])
                gam[(n, j)] = np.sum(_summand)
        ########################
        gam_sum = np.sum([i**2 for i in copy.deepcopy(gam).values()])
        print(np.abs(gam_sum))
        ########################
        for n in self.pauli_strings:
            for j in self.pauli_strings:
                probabilities[(n, j)] = gam[(n, j)]**2/gam_sum
                chi_dict[(n, j)] = gam_sum/(d**3*gam[(n, j)])

        return probabilities, chi_dict

    def generate_u_trace(self):
        observables = self.generate_observables()
        input_states = self.generate_observables()
        u_trace = {}
        for _state_idx in self.pauli_strings:
            for _obs_idx in self.pauli_strings:
                _state = input_states[_state_idx]
                _obs = observables[_obs_idx]
                _trace = np.dot(self.U, _state)
                _trace = np.dot(_trace, np.conj(np.transpose(self.U)))
                _trace = np.dot(_trace, _obs)
                _trace = _trace.diagonal()
                chi = _trace.sum()
                u_trace[_state_idx, _obs_idx] = chi
        return u_trace

    def get_alphas(self):
        alphas = {}
        for w_idx in self.pauli_strings:
            for rho_idx in self.pauli_strings:
                prod = 1
                for i in w_idx:
                    for j in rho_idx:
                        prod *= self.w_rho_coeffs[np.int(i)][np.int(j)]
                alphas[(w_idx, rho_idx)] = prod
        return alphas

    def generate_observables(self):
        observables = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            observables[_state] = copy.deepcopy(_op)
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
                    _ops.append(sigmaz())
            _op = tensor(_ops)
            tens_ops.append(csc_matrix(_op.full()))

        return tens_ops


class AmpProbDist(ProbDist):
    """Probability distribution based on the amplitudes of the observables.
    TODO: implement this
    """

    def __init__(self):
        pass

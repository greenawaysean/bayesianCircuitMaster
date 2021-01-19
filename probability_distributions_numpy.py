import numpy as np
from qutip import (sigmax, sigmay, sigmaz, qeye, basis, gate_expand_1toN,
                   Qobj, tensor, snot)
from scipy import linalg
from scipy.sparse import csc_matrix
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
        return [''.join(i) for i in
                itertools.product('0123', repeat=self.nqubits)]


class ChiProbDist(ProbDist):
    """Probability distribution for estimating the 0-fidelity based on an 
    adaptation of 10.1103/PhysRevLett.106.230501
    """

    def __init__(self, nqubits: int, U: Qobj):
        super().__init__(nqubits)
        self.tens_ops = self.get_tensored_ops()
        self.tens_states = self.get_tensored_states()
        self.U = U.full()
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
                _trace = np.dot(self.U, _state)
                _trace = np.dot(_trace, np.conj(np.transpose(self.U)))
                _trace = np.dot(_trace, _obs)
                _trace = _trace.diagonal()
                chi = _trace.sum(axis=0)
                chi_dict[_state_idx, _obs_idx] = chi  # np.real(chi)
                probabilities[(_state_idx, _obs_idx)] = (
                    1/d**3)*np.real(chi)**2
        return probabilities, chi_dict

    def generate_states_observables(self):
        init_state = tensor([basis(2, 0)] * self.nqubits).full()
        input_states = {}
        observables = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            _input_state = copy.deepcopy(self.tens_states[i])
            observables[_state] = copy.deepcopy(_op)
            _init_copy = copy.deepcopy(init_state)
            state = np.dot(_input_state, _init_copy)
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
            tens_ops.append(_op.full())

        return tens_ops

    def get_tensored_states(self):
        tens_ops = []
        for _state in self.pauli_strings:
            _ops = []
            for i in _state:
                if i == '0':
                    _ops.append(qeye(2))
                if i == '1':
                    _ops.append(generate_u3(np.arccos(-1/3), 0, 0))
                if i == '2':
                    _ops.append(generate_u3(np.arccos(-1/3), 2*np.pi/3, 0))
                if i == '3':
                    _ops.append(generate_u3(np.arccos(-1/3), 4*np.pi/3, 0))
            _op = tensor(_ops)
            tens_ops.append(_op.full())

        return tens_ops


class FlammiaProbDist(ProbDist):
    """Probability distribution for estimating the process fidelity as in
       10.1103/PhysRevLett.106.230501
    """

    def __init__(self, nqubits: int, U: Qobj):
        super().__init__(nqubits)
        self.tens_ops = self.get_tensored_ops()
        self.tens_states = self.get_tensored_states()
        self.U = U.full()
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
                _trace = np.dot(self.U, _state)
                _trace = np.dot(_trace, np.conj(np.transpose(self.U)))
                _trace = np.dot(_trace, _obs)
                _trace = _trace.diagonal()
                chi = _trace.sum(axis=0)
                chi_dict[_state_idx, _obs_idx] = chi  # np.real(chi)
                probabilities[(_state_idx, _obs_idx)] = np.abs((1/d**4)*chi**2)
        return probabilities, chi_dict

    def generate_states_observables(self):
        input_states = {}
        observables = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            _input_state = copy.deepcopy(_op)
            observables[_state] = copy.deepcopy(_op)
            # _init_copy = copy.deepcopy(init_state)
            # state = np.dot(_input_state, _init_copy)
            # input_states[_state] = np.dot(state, np.conj(np.transpose(state)))
            input_states[_state] = _input_state

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
            tens_ops.append(_op.full())

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
            tens_ops.append(_op.full())

        return tens_ops


class FullProbDist(ProbDist):
    """Alternative probability distribution (not currently working)
    """

    def __init__(self, nqubits: int, U: Qobj):
        super().__init__(nqubits)
        self.tens_ops = self.get_tensored_ops()
        self.tens_states = self.get_tensored_states()
        self.U = csc_matrix(U.full())
        self.Bmat = self.generate_Bmat(nqubits)
        self.probabilities, self.chi_dict = self.get_probs_and_chis()

    # def generate_Bmat(self, nqubits):
    #     Bmat = np.ndarray([4**nqubits, 4**nqubits])
    #     for i, op1 in enumerate(self.tens_states):
    #         for j, op2 in enumerate(self.tens_states):
    #             _trace = np.dot(op1, op2)
    #             _trace = _trace.diagonal()
    #             Bmat[i][j] = _trace.sum()
    #     return np.linalg.inv(Bmat)

    def generate_Bmat(self, nqubits):
        order = nqubits
        _alpha = 1
        P = Qobj([[0.25]*4]*4) - 0.5*qeye(4)
        B_inv = None
        for k in range(order+1):
            _beta = (-1)**k
            s = [qeye(4)]*(nqubits - k) + [P]*k
            X_k = None
            track_ops = []
            for i in itertools.permutations(s):
                if i in track_ops:
                    continue
                track_ops.append(i)
                if X_k is None:
                    X_k = tensor(list(i))
                else:
                    X_k += tensor(list(i))
            if B_inv is None:
                B_inv = _alpha*_beta*X_k
            else:
                B_inv += _alpha*_beta*X_k
        return B_inv.full()

    # def get_probs_and_chis(self):
    #     d = 2**self.nqubits
    #     probabilities = {}
    #     chi_dict = {}
    #     for i, rho_i in enumerate(self.tens_states):
    #         for k, W_k in enumerate(self.tens_ops):
    #             # generate C_ik
    #             for j, rho_j in enumerate(self.tens_states):

    def get_probs_and_chis(self):
        d = 2**self.nqubits
        # input_states, observables = self.generate_states_observables()
        probabilities = {}
        chi_dict = {}
        for k, _statek in enumerate(self.tens_states):
            for kp, _obs in enumerate(self.tens_ops):
                gam = 0
                for j, _statej in enumerate(self.tens_states):
                    _trace = np.dot(self.U, _statej)
                    _trace = np.dot(_trace, np.conj(np.transpose(self.U)))
                    _trace = np.dot(_trace, _obs)
                    _trace = _trace.diagonal()
                    zeta = _trace.sum()
                    gam += self.Bmat[k][j]*zeta
                # _trace = np.dot(self.U, _statek)
                # _trace = np.dot(_trace, np.conj(np.transpose(self.U)))
                # _trace = np.dot(_trace, _obs)
                # _trace = _trace.diagonal()
                # chi = _trace.sum()
                chi_dict[self.pauli_strings[k],
                         self.pauli_strings[kp]] = np.sign(gam)
                p = np.abs(gam)
                probabilities[self.pauli_strings[k],
                              self.pauli_strings[kp]] = p
        _keys = probabilities.keys()
        _vals = [probabilities[k] for k in _keys]
        norm = np.sum(_vals)
        chi_dict = {key: chi_dict[key]*norm for i, key in enumerate(_keys)}
        probabilities = {key: probabilities[key]/norm for key in _keys}

        return probabilities, chi_dict

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
        init_state = csc_matrix(tensor([basis(2, 0)] * self.nqubits).full())
        for _state in self.pauli_strings:
            _ops = []
            for i in _state:
                if i == '0':
                    _ops.append(qeye(2))
                if i == '1':
                    _ops.append(generate_u3(np.arccos(-1/3), 0, 0))
                if i == '2':
                    _ops.append(generate_u3(np.arccos(-1/3), 2*np.pi/3, 0))
                if i == '3':
                    _ops.append(generate_u3(np.arccos(-1/3), 4*np.pi/3, 0))
            _op = tensor(_ops)
            # tens_ops.append(csc_matrix(_op.full()))
            _op2 = csc_matrix(_op.full())
            _state_op = np.dot(_op2, init_state)
            _state_op = np.dot(_state_op, np.conj(np.transpose(_state_op)))
            tens_ops.append(_state_op)
        return tens_ops


def generate_u3(theta, phi, lam):
    u_00 = np.cos(theta/2)
    u_01 = -np.exp(1j*lam)*np.sin(theta/2)
    u_10 = np.exp(1j*phi)*np.sin(theta/2)
    u_11 = np.exp(1j*(lam + phi))*np.cos(theta/2)

    return Qobj([[u_00, u_01], [u_10, u_11]])

import copy
import itertools
from typing import List, Union

import numpy as np
from qutip import (Qobj, basis, gate_expand_1toN, qeye,
                   sigmax, sigmay, sigmaz, snot, tensor, rx, ry, rz, cnot)
from general_utils import GateObj, U_from_hamiltonian
from probability_distributions_numpy import ProbDist, ChiProbDist, FlammiaProbDist, FullProbDist


class BaseAnsatz:
    def __init__(self, gate_list: List[GateObj], nqubits: int):
        self.gate_list = gate_list
        self.nqubits = nqubits

    def populate_ansatz(self, params):
        raise NotImplementedError


class QutipAnsatz(BaseAnsatz):
    def __init__(self, gate_list: List[GateObj], nqubits: int):
        super().__init__(gate_list, nqubits)

    def populate_ansatz(self, params):
        idx = 0
        ansatz = tensor([qeye(2)]*self.nqubits)
        for gate in self.gate_list:
            if gate.parameterise:
                if gate.name == 'U3':
                    u3 = generate_u3(gate.params[idx], gate.params[idx+1],
                                     gate.params[idx+2])
                    gate = gate_expand_1toN(u3, self.nqubits, gate.qubits)
                    ansatz = gate*ansatz
                    idx += 3
                elif gate.name == 'RX':
                    gate = rx(params[idx], self.nqubits, gate.qubits)
                    ansatz = gate*ansatz
                    idx += 1
                elif gate.name == 'RY':
                    gate = ry(params[idx], self.nqubits, gate.qubits)
                    ansatz = gate*ansatz
                    idx += 1
                elif gate.name == 'RZ':
                    gate = rz(params[idx], self.nqubits, gate.qubits)
                    ansatz = gate*ansatz
                    idx += 1
            else:
                if gate.name == 'X':
                    ansatz = gate_expand_1toN(sigmax(),
                                              self.nqubits, gate.qubits)*ansatz
                elif gate.name == 'Y':
                    ansatz = gate_expand_1toN(sigmay(),
                                              self.nqubits, gate.qubits)*ansatz
                elif gate.name == 'Z':
                    ansatz = gate_expand_1toN(sigmaz(),
                                              self.nqubits, gate.qubits)*ansatz
                elif gate.name == 'CX' or gate.name == 'CNOT':
                    ansatz = cnot(self.nqubits,
                                  gate.qubits[0], gate.qubits[1])*ansatz

        return ansatz


class Estimator:
    def __init__(self, prob_dist: ProbDist, nqubits: int, ansatz: BaseAnsatz):
        self.prob_dist = prob_dist
        self.prob_dict = self.prob_dist.probabilities
        self.chi_dict = self.prob_dist.chi_dict
        self.probs = [self.prob_dict[key] for key in self.prob_dict]
        self.keys = [key for key in self.prob_dict]
        self.nqubits = nqubits
        self.ansatz = ansatz

    def calculate_fom(self, params: List[float], length: int):
        self.length = length
        settings = self.select_settings()
        ideal_chi = [self.chi_dict[i] for i in settings]
        expects = self.evaluate_expectations(settings, params)

        fom = 0
        for i, _chi in enumerate(ideal_chi):
            fom += expects[i] / _chi
        fom += self.length - len(settings)
        fom /= self.length

        return np.real(fom)

    def select_settings(self):
        choices = np.random.choice(
            [i for i in range(len(self.keys))], self.length, p=self.probs, replace=True)
        settings = [self.keys[i] for i in choices]
        settings = [item for item in settings if item[1] != '0' * self.nqubits]

        return settings

    def evaluate_expectations(self, settings: List[str], params: List[float]):
        raise NotImplementedError


class QutipEstimator(Estimator):
    def __init__(self, prob_dist: ProbDist, nqubits: int, ansatz: QutipAnsatz):
        super().__init__(prob_dist, nqubits, ansatz)
        self.input_states, self.meas_bases = self.generate_states_bases()

    def evaluate_expectations(self, settings: List[str], params: List[float]):
        expectations = []
        for sett in settings:
            init = tensor([Qobj([[1, 0], [0, 0]])]*self.nqubits)
            state = self.input_states[sett[0]] * \
                init*self.input_states[sett[0]].dag()
            basis = self.meas_bases[sett[1]]
            circ = self.ansatz.populate_ansatz(params)

            exp = (circ*state*circ.dag()*basis).tr()
            expectations.append(exp)

        return expectations

    def generate_states_bases(self):
        states = {}
        bases = {}
        for sett in self.keys:
            state = sett[0]
            basis = sett[1]
            states[state] = self.get_input_state(state)
            bases[basis] = self.get_meas_basis(basis)

        return states, bases

    def get_input_state(self, op: str):
        operator = []
        for i in op:
            if i == '0':
                operator.append(qeye(2))
            elif i == '1':
                operator.append(generate_u3(np.arccos(-1/3), 0, 0))
            elif i == '2':
                operator.append(generate_u3(np.arccos(-1/3), 2*np.pi/3, 0))
            elif i == '3':
                operator.append(generate_u3(np.arccos(-1/3), 4*np.pi/3, 0))

        return tensor(operator)

    def get_meas_basis(self, op: str):
        operator = []
        for i in op:
            if i == '0':
                operator.append(qeye(2))
            elif i == '1':
                operator.append(sigmax())
            elif i == '2':
                operator.append(sigmay())
            elif i == '3':
                operator.append(sigmaz())
        return tensor(operator)


def generate_u3(theta, phi, lam):
    a = np.cos(theta/2)
    b = -np.exp(1j*lam)*np.sin(theta/2)
    c = np.exp(1j*phi)*np.sin(theta/2)
    d = np.exp(1j*(phi+lam))*np.cos(theta/2)

    return Qobj([[a, b], [c, d]])


class QutipFlammiaEstimator(Estimator):
    def __init__(self, prob_dist: ProbDist, nqubits: int, ansatz: QutipAnsatz):
        super().__init__(prob_dist, nqubits, ansatz)
        self.input_states, self.meas_bases = self.generate_states_bases()

    def evaluate_expectations(self, settings, params):
        expectations = []
        for sett in settings:
            state = self.input_states[sett[0]]
            basis = self.meas_bases[sett[1]]
            circ = self.ansatz.populate_ansatz(params)

            exp = (circ*state*circ.dag()*basis).tr()
            expectations.append(exp)

        return expectations

    def generate_states_bases(self):
        states = {}
        bases = {}
        for sett in self.keys:
            state = sett[0]
            states[state] = self.get_operator(state)
            bases[state] = self.get_operator(state)

        return states, bases

    def get_operator(self, op: List[str]):
        operator = []
        for i in op:
            if i == '0':
                operator.append(qeye(2))
            elif i == '1':
                operator.append(sigmax())
            elif i == '2':
                operator.append(sigmay())
            elif i == '3':
                operator.append(sigmaz())

        return tensor(operator)


class FullQutipEstimator(QutipEstimator):
    def __init__(self, prob_dist: ProbDist, nqubits: int, ansatz: QutipAnsatz):
        super().__init__(prob_dist, nqubits, ansatz)

    def calculate_fom(self, params: List[float], length: int):
        self.length = length
        settings = self.select_settings()
        ideal_chi = [self.chi_dict[i] for i in settings]
        expects = self.evaluate_expectations(settings, params)
        norm = np.sum(self.probs)
        fom = 0
        for i, _chi in enumerate(ideal_chi):
            fom += (1/2**(3*self.nqubits))*expects[i]*_chi
        fom += self.length - len(settings)
        fom /= self.length

        return np.real(fom)

    def generate_states_bases(self):
        states = {}
        bases = {}
        for sett in self.keys:
            state = sett[0]
            states[state] = self.get_state(state)
            bases[state] = self.get_basis(state)

        return states, bases

    def get_state(self, state):
        _ops = []
        for i in state:
            if i == '0':
                _ops.append(qeye(2))
            if i == '1':
                _ops.append(generate_u3(np.arccos(-1/3), 0, 0))
            if i == '2':
                _ops.append(generate_u3(np.arccos(-1/3), 2*np.pi/3, 0))
            if i == '3':
                _ops.append(generate_u3(np.arccos(-1/3), 4*np.pi/3, 0))
        _op = tensor(_ops)

        return _op

    def get_basis(self, op):
        operator = []
        for i in op:
            if i == '0':
                operator.append(qeye(2))
            elif i == '1':
                operator.append(sigmax())
            elif i == '2':
                operator.append(sigmay())
            elif i == '3':
                operator.append(sigmaz())

        return tensor(operator)


if __name__ == "__main__":
    # define hamiltonian
    nqubits = 3
    t = 0.84
    hamiltonian = []
    for i in range(nqubits):
        hamiltonian.append(GateObj('X', i, True, 0.55))
        hamiltonian.append(GateObj('Z', i, True, 0.35))
    for i in range(nqubits):
        if i < nqubits - 1:
            hamiltonian.append(GateObj(['Z', 'Z'], [i, i+1], True, 1.))

    ideal_U = U_from_hamiltonian(hamiltonian, nqubits, t)

    # define list of GateObjs which will be our estimate ansatz
    tsteps = 1
    ansatz = []
    params = []
    for k in range(tsteps):
        for i in range(nqubits):
            ansatz.append(GateObj('RX', i, True, 2*0.55*t/(tsteps)))
            params.append(2*0.55*t/(tsteps))
            ansatz.append(GateObj('RZ', i, True, 2*0.35*t/(tsteps)))
            params.append(2*0.35*t/(tsteps))
        for i in range(nqubits):
            if i < nqubits - 1:
                ansatz.append(GateObj('CNOT', [i, i+1], False))
                ansatz.append(GateObj('RZ', i+1, True, 2*t/(tsteps)))
                params.append(2*t/tsteps)
                ansatz.append(GateObj('CNOT', [i, i+1], False))

    prob_dist = FullProbDist(nqubits=nqubits, U=ideal_U)
    q_ansatz = QutipAnsatz(ansatz, nqubits)

    q_est = FullQutipEstimator(prob_dist, nqubits, q_ansatz)

    for i in range(100):
        print(q_est.calculate_fom(params, 100))

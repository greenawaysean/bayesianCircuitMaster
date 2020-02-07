from typing import List
import numpy as np
import itertools
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from probability_distributions import ProbDist
from general_utils import GateObj


class qCirc:
    """Separates state preparation, unitary evolution and measurement into separate
    sections and applies them onto a qiskit circuit
    """

    def __init__(self, nqubits: int, input_state: List[GateObj], U: List[GateObj],
                 meas_basis: List[GateObj]):
        """
        inputs: nqubits: number of qubits
                input_state: initial state settings for the simulation
                U: list of GateObj's defining the unitary evolution
                meas_basis: basis to measure in
        """
        self.nqubits = nqubits
        self.input_state = input_state
        self.U = U
        self.meas_basis = meas_basis

        self.qreg = QuantumRegister(self.nqubits)
        self.creg = ClassicalRegister(self.nqubits)
        self.qc = QuantumCircuit(self.qreg, self.creg)

    def apply_init_state(self):
        for _gate in self.input_state:
            apply_gate(self.qc, self.qreg, _gate)

    def apply_U(self):
        for _gate in self.U:
            apply_gate(self.qc, self.qreg, _gate)

    def rotate_meas_basis(self):
        for _gate in self.meas_basis:
            apply_gate(self.qc, self.qreg, _gate)

    def build_circuit(self):
        self.apply_init_state()
        self.apply_U()
        self.rotate_meas_basis()

        for i in range(self.nqubits):
            self.qc.measure(self.qreg[i], self.creg[i])

        return self.qc


class EstimateCircuits:
    """Generate a list of circuits which when run estimate the fidelity for a given
    unitary operator, given a precalculated probability distribution
    """

    def __init__(self, prob_dist: ProbDist, U: List[GateObj], nqubits: int,
                 length: int):
        self.prob_dist = prob_dist
        self.prob_dict = self.prob_dist.get_probabilities()
        self.U = U
        self.nqubits = nqubits
        self.length = length
        self.settings = None
        self.qutip_settings = None

    def calculate_fidelity(self, num_shots: int, backend):
        expects = self.run_circuits(num_shots, backend)
        chi_dict = self.prob_dist.get_chi_dict()
        ideal_chi = [chi_dict[i] for i in self.qutip_settings]
        fidelity = 0
        for i, _chi in enumerate(ideal_chi):
            fidelity += np.abs(expects[i]/_chi)
        fidelity /= self.length

        return fidelity

    def generate_circuits(self):
        probs = [np.abs(self.prob_dict[key]) for key in self.prob_dict]
        keys = [key for key in self.prob_dict]
        self.settings, self.qutip_settings = self.select_settings(probs, keys)
        circs = []
        for _setting in self.settings:
            init_state, observ = self.parse_setting(_setting)
            _circ = qCirc(self.nqubits, init_state, self.U, observ).build_circuit()
            circs.append(_circ)

        return circs

    def select_settings(self, probs, keys):
        choices = np.random.choice([i for i in range(len(keys))], self.length, p=probs)
        qutip_settings = [keys[i] for i in choices]
        # qutip and qiskit use mirrored qubit naming schemes
        settings = []
        for _set in qutip_settings:
            setting0 = _set[0][::-1]
            setting1 = _set[1][::-1]
            settings.append((setting0, setting1))
        return settings, qutip_settings

    def parse_setting(self, setting):
        _state, _obs = setting
        init_state = []
        for i, _op in enumerate(_state):
            if _op == '0':
                continue
            elif _op == '1':
                _s = GateObj(name='X', qubits=i, parameterise=False, params=None)
                init_state.append(_s)
            elif _op == '2':
                _s = GateObj(name='Y', qubits=i, parameterise=False, params=None)
                init_state.append(_s)
            elif _op == '3':
                _s = GateObj(name='Z', qubits=i, parameterise=False, params=None)
                init_state.append(_s)
        observe = []
        for i, _op in enumerate(_obs):
            # apply the gates which will rotate the qubits to the req'd basis
            if _op == '0':
                _o = GateObj(name='Z', qubits=i, parameterise=False, params=None)
            elif _op == '1':
                _o = GateObj(name='H', qubits=i, parameterise=False, params=None)
                observe.append(_o)
            elif _op == '2':
                _o = GateObj(name='HSdag', qubits=i, parameterise=False, params=None)
                observe.append(_o)
            elif _op == '3':
                _o = GateObj(name='I', qubits=i, parameterise=False, params=None)
                observe.append(_o)

        return init_state, observe

    def run_circuits(self, num_shots: int, backend):
        circs = self.generate_circuits()
        # qjobs = [transpile(c, optimization_level=2) for c in circs]
        results = execute(circs, backend=backend, shots=num_shots).result()
        expects = [generate_expectation(results.get_counts(i), self.nqubits)
                   for i in range(len(circs))]

        return expects


def apply_gate(circ: QuantumCircuit, qreg: QuantumRegister, gate: GateObj):
    """Applies a gate to a quantum circuit. More complicated gates such as RXX gates
    should be decomposed into single qubit gates and CNOTs prior to calling this
    function.
    """
    if not isinstance(gate.qubits, list):
        q = gate.qubits
        params = gate.params
        if gate.name == 'I':
            pass
        elif gate.name == 'H':
            circ.h(qreg[q])
        elif gate.name == 'HSdag':
            circ.h(qreg[q])
            circ.sdg(qreg[q])
        elif gate.name == 'X':
            circ.x(qreg[q])
        elif gate.name == 'Y':
            circ.y(qreg[q])
        elif gate.name == 'Z':
            circ.z(qreg[q])
        elif gate.name == 'RX':
            circ.rx(params, qreg[q])
        elif gate.name == 'RY':
            circ.ry(params, qreg[q])
        elif gate.name == 'RZ':
            circ.rz(params, qreg[q])
        elif gate.name == 'U3':
            circ.u3(params[0], params[1], params[2], qreg[q])
    else:
        cntrl = gate.qubits[0]
        trgt = gate.qubits[1]
        circ.cx(qreg[cntrl], qreg[trgt])

    return circ


def generate_expectation(counts_dict, nqubits):
    """Generate the expectation value for a Pauli string operator given a dictionary of
    the counts from the machine
    """
    total_counts = 0

    bitstrings = [''.join(i) for i in itertools.product('01', repeat=nqubits)]

    expect = 0
    # add any missing counts to dictionary to avoid errors
    for string in bitstrings:
        if string not in counts_dict:
            counts_dict[string] = 0
        count = 0
        for i in string:
            if i == '1':
                count += 1
        if count % 2 == 0:  # subtract odd product of -ve evalues, add even products
            expect += counts_dict[string]
            total_counts += counts_dict[string]
        else:
            expect -= counts_dict[string]
            total_counts += counts_dict[string]

    return expect / total_counts

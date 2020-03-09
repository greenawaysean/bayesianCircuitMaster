from typing import List, Union
import numpy as np
import copy
import itertools
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.aqua import QuantumInstance
from qiskit.circuit import Parameter
from qiskit.compiler import transpile
from probability_distributions import ProbDist
from general_utils import GateObj


class qCirc:
    """Quantum circuit class

    Separates state preparation, unitary evolution and measurement into separate
    sections and applies them onto a qiskit circuit

    Attributes:
    -----------
    nqubits: number of qubits
    input_state: initial state settings for the simulation
    V: list of GateObj's defining the simulated unitary evolution
    params_list: list of parameters to populate the circuit with

    meas_basis: settings for the measurement basis

    Methods:
    --------
    apply_init_state: input state
    apply_V: (Trotterised) unitary evolution operator, parameterised for optimisation
    rotate_meas_basis: rotate into the eigenbasis of the desired observable
    build_circuit: combine the three independent circuits together
    """

    def __init__(self, nqubits: int, input_state: List[GateObj], V: List[GateObj],
                 meas_basis: List[GateObj], backend: str, init_layout: dict = None):
        self.nqubits = nqubits
        self.input_state = input_state
        self.V = V
        self.meas_basis = meas_basis
        self.backend = backend
        self.init_layout = init_layout
        self.qreg = QuantumRegister(self.nqubits, name='qreg')
        self.creg = ClassicalRegister(self.nqubits)
        self.qc = QuantumCircuit(self.qreg, self.creg)
        self.build_circuit()
        self.qc = transpile(self.qc, backend=self.backend, initial_layout=self.init_layout)

    def populate_circuits(self, params_list):
        circ = copy.deepcopy(self.qc)
        param_dict = self.generate_params_dict(params_list)
        circ = circ.bind_parameters(param_dict)

        return circ

    def generate_params_dict(self, params_list):
        params_dict = {}
        idx = 0
        for _gate in self.V:
            if _gate.parameterise:
                params_dict[self.params[idx]] = params_list[idx]
                idx += 1
        return params_dict

    def apply_init_state(self):
        for _gate in self.input_state:
            apply_gate(self.qc, self.qreg, _gate)

    def apply_V(self):
        params = []
        idx = 0
        for _gate in self.V:
            if _gate.parameterise:
                if _gate.name == 'U3':
                    params.append((Parameter(f'{idx}'),
                                   Parameter(f'{idx + 1}'),
                                   Parameter(f'{idx + 2}')))
                    apply_gate(self.qc, self.qreg, _gate, parameterise=True,
                               param=params[-1])
                    idx += 3
                else:
                    params.append(Parameter(f'{idx}'))
                    apply_gate(self.qc, self.qreg, _gate, parameterise=True,
                               param=params[-1])
                    idx += 1
            else:
                apply_gate(self.qc, self.qreg, _gate)
        self.params = tuple(params)

    def rotate_meas_basis(self):
        meas_idx = []
        for _gate in self.meas_basis:
            apply_gate(self.qc, self.qreg, _gate)
            meas_idx.append(_gate.qubits)
            self.qc.measure(self.qreg[_gate.qubits], self.creg[_gate.qubits])

    def build_circuit(self):
        """Builds seperate circuits for input states, observables and unitary
        evolution, with the first two being static and the latter being parameterised.

        The full circuit is built by composing all three circuits together.
        """
        self.apply_init_state()
        self.apply_V()
        self.rotate_meas_basis()


class EstimateCircuits:
    """Class to handle generating circuits for fidelity estimation.

    Builds all possible circuits as parameterised qiskit circuits, selects subsets of
    them for fidelity estimation and runs them with a given set of parameters.
    """

    def __init__(self, prob_dist: ProbDist, V: List[GateObj], nqubits: int,
                 length: int, num_shots: int, backend: str, init_layout: dict,
                 noise_model=None):
        self.prob_dist = prob_dist
        self.prob_dict = self.prob_dist.probabilities
        self.chi_dict = self.prob_dist.chi_dict
        self.V = V
        self.nqubits = nqubits
        self.length = length
        self.num_shots = num_shots
        self.backend = backend
        self.init_layout = init_layout
        self.noise_model = noise_model
        self.circuits = self.generate_circuits()
        self.quant_inst = QuantumInstance(backend=self.backend, shots=self.num_shots,
                                          initial_layout=self.init_layout,
                                          skip_qobj_validation=False,
                                          noise_model=self.noise_model)

    def calculate_fidelity(self, params):
        probs = [self.prob_dict[key] for key in self.prob_dict]
        keys = [key for key in self.prob_dict]
        settings, qutip_settings = self.select_settings(probs, keys)
        ideal_chi = [self.chi_dict[i] for i in qutip_settings]
        expects = self.run_circuits(settings, params)
        fidelity = 0

        idx = 0
        for i in range(len(settings)):
            # if settings[1] != '0'*self.nqubits:
            fidelity += (1/np.sqrt(2**self.nqubits))*expects[idx] / ideal_chi[idx]
            idx += 1
            # else:
            #     fidelity += 1
        # for i, _chi in enumerate(ideal_chi):
        #     fidelity += expects[i] / _chi
        #
        # for i in range(int(self.length) - len(ideal_chi)):
        #     fidelity += 1/(2**(2*self.nqubits))

        fidelity += self.length - len(settings)

        fidelity /= self.length

        return np.abs(fidelity)  # np.abs(fidelity)

    def generate_circuits(self):
        """Builds circuits for all possible combinations of input states and
        observables.

        Returns:
        --------
        circs: dictionary indexing all possible circuits for fidelity estimation
        """
        settings = [key for key in self.prob_dict]
        circs = {}
        for _setting in settings:
            _init_states, observ = self.parse_setting(_setting)
            _circs = []
            for init_state in _init_states:
                _circ = qCirc(self.nqubits, init_state,
                              self.V, observ, self.backend, self.init_layout)
                _circs.append(_circ)
            circs[_setting] = _circs
        return circs

    def run_circuits(self, settings, params):
        """Choose a subset of <length> circuits for fidelity estimation and run them

        Parameters:
        -----------
        params: list of parameters to populate the circuits with (intended to be
        adapted through optimisation)

        Returns:
        --------
        expects: list of expectation values for each circuit in the list
        """
        chosen_circs = []
        for _setting in settings:
            _idx = np.random.choice([i for i in range(len(self.circuits[_setting]))])
            chosen_circs.append(self.circuits[_setting][_idx])
        # chosen_circs = [self.circuits[_setting] for _setting in settings]
        exec_circs = []
        exec_circs = [qc.populate_circuits(params) for qc in chosen_circs]
        results = self.quant_inst.execute(exec_circs, had_transpiled=True)
        expects = [
            generate_expectation(results.get_counts(i)) for i in range(len(exec_circs))
        ]

        return expects

    def select_settings(self, probs, keys):
        """Choose a set of settings given a probability distribution"""
        choices = []
        choices = np.random.choice(
            [i for i in range(len(keys))], self.length, p=probs, replace=True)
        qutip_settings = [keys[i] for i in choices]
        # qutip and qiskit use mirrored qubit naming schemes
        settings = []
        for _set in qutip_settings:
            setting0 = _set[0][::-1]
            setting1 = _set[1][::-1]
            settings.append((setting0, setting1))

        settings = [item for item in settings if item[1] != '0' * self.nqubits]
        qutip_settings = [
            item for item in qutip_settings if item[1] != '0' * self.nqubits]
        return settings, qutip_settings

    def parse_setting(self, setting):
        """Convert setting into a list of GateObj's for easier circuit conversion"""
        _state, _obs = setting
        iter_list = [''.join(i) for i in itertools.product('01', repeat=len(_state))]
        init_states = []
        for _comb in iter_list:
            init_state = []
            for i, _op in enumerate(_state):
                if _op == '0':
                    if _comb[i] == '0':
                        continue
                    else:
                        _s = GateObj(name='X', qubits=i,
                                     parameterise=False, params=None)
                elif _op == '1':
                    if _comb[i] == '0':
                        _s = GateObj(name='H', qubits=i,
                                     parameterise=False, params=None)
                    else:
                        _s = GateObj(name='U3', qubits=i,
                                     parameterise=True, params=[3*np.pi/2, 0, 0])
                elif _op == '2':
                    if _comb[i] == '0':
                        _s = GateObj(name='U3', qubits=i,
                                     parameterise=True, params=[np.pi/2, np.pi/2, 0])
                    else:
                        _s = GateObj(name='U3', qubits=i,
                                     parameterise=True, params=[3*np.pi/2, np.pi/2, 0])
                elif _op == '3':
                    if _comb[i] == '0':
                        continue
                    else:
                        _s = GateObj(name='X', qubits=i,
                                     parameterise=False, params=None)
                init_state.append(_s)
            init_states.append(init_state)
        observe = []
        for i, _op in enumerate(_obs):
            # apply the gates which will rotate the qubits to the req'd basis
            if _op == '0':
                continue
            elif _op == '1':
                _o = GateObj(name='H', qubits=i,
                             parameterise=False, params=None)
            elif _op == '2':
                _o = GateObj(name='HSdag', qubits=i,
                             parameterise=False, params=None)
            elif _op == '3':
                _o = GateObj(name='I', qubits=i,
                             parameterise=False, params=None)
            observe.append(_o)

        return init_states, observe


def apply_gate(circ: QuantumCircuit, qreg: QuantumRegister, gate: GateObj,
               parameterise: bool = False, param: Union[Parameter, tuple] = None):
    """Applies a gate to a quantum circuit.

    More complicated gates such as RXX gates should be decomposed into single qubit
    gates and CNOTs prior to calling this function. If parameterise is True, then
    qiskit's placeholder parameter theta will be used in place of any explicit
    parameters.
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
            circ.s(qreg[q])
            circ.h(qreg[q])
        elif gate.name == 'X':
            circ.x(qreg[q])
        elif gate.name == 'Y':
            circ.y(qreg[q])
        elif gate.name == 'Z':
            circ.z(qreg[q])
        elif gate.name == 'RX':
            if parameterise:
                circ.rx(param, qreg[q])
            else:
                circ.rx(params, qreg[q])
        elif gate.name == 'RY':
            if parameterise:
                circ.ry(param, qreg[q])
            else:
                circ.ry(params, qreg[q])
        elif gate.name == 'RZ':
            if parameterise:
                circ.rz(param, qreg[q])
            else:
                circ.rz(params, qreg[q])
        elif gate.name == 'U3':
            if parameterise:
                circ.u3([i for i in param], qreg[q])
            circ.u3(params[0], params[1], params[2], qreg[q])
    else:
        cntrl = gate.qubits[0]
        trgt = gate.qubits[1]
        circ.cx(qreg[cntrl], qreg[trgt])

    return circ


def generate_expectation(counts_dict):
    """Generate the expectation value for a Pauli string operator

    Parameters:
    -----------
    counts_dict: dictionary of counts generated from the machine (or qasm simulator)
    N: number of qubits being measured (note - NOT total number of qubits)

    Returns:
    --------
    expect: expectation value of the circuit in the measured basis
    """
    total_counts = 0
    key_len = [len(key) for key in counts_dict]
    N = key_len[0]
    bitstrings = [''.join(i) for i in itertools.product('01', repeat=N)]

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

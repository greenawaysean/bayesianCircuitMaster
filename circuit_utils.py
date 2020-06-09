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
        self.qc = transpile(self.qc, backend=self.backend,
                            initial_layout=self.init_layout)

    def populate_circuits(self, params_list):
        circ = copy.deepcopy(self.qc)
        param_dict = self.generate_params_dict(params_list)
        circ = circ.bind_parameters(param_dict)

        return circ

    def generate_params_dict(self, params_list):
        params_dict = {}
        idx = 0
        p_idx = 0
        for _gate in self.V:
            if _gate.parameterise:
                if isinstance(self.params[idx], tuple):
                    for i in range(3):
                        params_dict[self.params[idx][i]] = params_list[p_idx]
                        p_idx += 1
                else:
                    params_dict[self.params[idx]] = params_list[p_idx]
                    p_idx += 1
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

    Parameters
    ----------
    prob_dist : ProbDist
        Probability distribution over which measurement settings are sampled.
    V : List[GateObj]
        List of gate objects defining the ansatz.
    nqubits : int
        Number of qubits.
    length : int
        Number of measurement settings to sample from in the fidelity estimation.
    num_shots : int
        Number of shots to use in the IBMQ machines.
    backend : str
        Which IBMQ backend to use.
    init_layout : dict
        Qubit layout for the IBMQ machine.
    noise_model : type
        Only for use with simulators: noise model emulating one of the real devices.

    Attributes
    ----------
    prob_dict : dict
        Conversion of probability distribution to a callable form for convenience.
    chi_dict : dict
        Conversion of ideal expectation values to a callable form for convenience.
    circuits : list
        List of all possible circuits to sample from, prebuilt to lighten the computational cost during optimisation.
    """

    def __init__(self, prob_dist: ProbDist, V: List[GateObj], nqubits: int,
                 num_shots: int, backend: str, init_layout: dict, noise_model=None):
        self.prob_dist = prob_dist
        self.prob_dict = self.prob_dist.probabilities
        self.chi_dict = self.prob_dist.chi_dict
        self.V = V
        self.nqubits = nqubits
        self.num_shots = num_shots
        self.backend = backend
        self.init_layout = init_layout
        self.noise_model = noise_model
        self.circuits = self.generate_circuits()
        self.quant_inst = QuantumInstance(backend=self.backend, shots=self.num_shots,
                                          initial_layout=self.init_layout,
                                          skip_qobj_validation=False,
                                          noise_model=self.noise_model)

    def calculate_fidelity(self, params, length):
        self.length = length
        probs = [self.prob_dict[key] for key in self.prob_dict]
        keys = [key for key in self.prob_dict]
        settings, qutip_settings = self.select_settings(probs, keys)
        ideal_chi = [self.chi_dict[i] for i in qutip_settings]
        expects = self.run_circuits(settings, params)

        fidelity = 0
        for i, _chi in enumerate(ideal_chi):
            fidelity += expects[i] / _chi

        fidelity += self.length - len(settings)
        fidelity /= self.length

        return np.real(fidelity)

    def calculate_bulk_fidelity(self, params: List, temp_len: int):
        self.length = temp_len
        probs = [self.prob_dict[key] for key in self.prob_dict]
        keys = [key for key in self.prob_dict]
        settings = []
        qutip_settings = []
        ideal_chi = []
        for i, _p in enumerate(params):
            _sett, _q_sett = self.select_settings(probs, keys)
            _chi = [self.chi_dict[i] for i in _q_sett]
            for _l in [_sett, _q_sett]:
                if len(_l) <= temp_len:
                    _l += ['X']*np.int(temp_len - len(_l))
            if len(_chi) <= temp_len:
                _chi += [1000.0j]*np.int(temp_len - len(_chi))
            settings += _sett
            qutip_settings += _q_sett
            ideal_chi += _chi
        flat_params = [item for sublist in params for item in sublist]
        expects = self.run_circuits(settings, flat_params)
        idx = 0
        reshaped_exp = []
        for i, _s in enumerate(settings):
            if _s != 'X':
                reshaped_exp.append(expects[idx])
                idx += 1
            else:
                reshaped_exp.append(1.0)
        reshaped_exp = np.reshape(reshaped_exp, [-1, temp_len])
        ideal_chi = np.reshape(ideal_chi, [-1, temp_len])

        fidels = []
        for i, _exp in enumerate(reshaped_exp):
            _f = 0.0
            for j, _chi in enumerate(ideal_chi[i]):
                if _chi == 1000.0j:
                    _f += 1.0
                else:
                    _f += _exp[j] / _chi
            fidels.append(_f/temp_len)
        return fidels

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
            init_state, observ = self.parse_setting(_setting)
            _circ = qCirc(self.nqubits, init_state,
                          self.V, observ, self.backend, self.init_layout)
            circs[_setting] = _circ
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
        chosen_circs = [self.circuits[_s] for _s in settings if _s != 'X']
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
        init_state = []
        for i, _op in enumerate(_state):
            if _op == '0':
                continue
            elif _op == '1':
                _s = GateObj(name='U3', qubits=i, parameterise=True,
                             params=[np.arccos(-1/3), 0.0, 0.0])
            elif _op == '2':
                _s = GateObj(name='U3', qubits=i, parameterise=True,
                             params=[np.arccos(-1/3), 2*np.pi/3, 0.0])
            elif _op == '3':
                _s = GateObj(name='U3', qubits=i, parameterise=True,
                             params=[np.arccos(-1/3), 4*np.pi/3, 0.0])
            init_state.append(_s)
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

        return init_state, observe


class FlammiaEstimateCircuits(EstimateCircuits):
    """Subclass of EstimateCircuits for implementation of the true fidelity estimation
    given in Flammia et al.'s paper.

    """

    def __init__(self, prob_dist: ProbDist, V: List[GateObj], nqubits: int,
                 length: int, num_shots: int, backend: str, init_layout: dict,
                 p_length: int, noise_model=None):

        self.p_length = p_length
        super().__init__(prob_dist, V, nqubits, length, num_shots, backend,
                         init_layout, noise_model)

    def calculate_fidelity(self, params, eval_var=False):
        probs = [self.prob_dict[key] for key in self.prob_dict]
        keys = [key for key in self.prob_dict]
        settings, qutip_settings = self.select_settings(probs, keys)
        expects = self.run_circuits(settings, params)

        ideal_chi = [self.chi_dict[_q] for _q in qutip_settings]

        _expects = []
        idx = 0
        _e = 0
        count = 0
        for i in range(len(settings)):
            _eig = self.generate_eigenvalue(settings[i][0], settings[i][2])
            _e += _eig * expects[i]
            idx += 1
            if idx == self.p_length:
                # print(_e, ideal_chi[count])
                count += 1
                _expects.append((2**self.nqubits)*_e/self.p_length)
                _e = 0
                idx = 0

        fidelity = 0
        for i, _chi in enumerate(ideal_chi):
            fidelity += _expects[i] / _chi

        fidelity += np.int(self.length/self.p_length) - len(ideal_chi)
        fidelity /= np.int(self.length/self.p_length)
        # fidelity /= np.int(len(ideal_chi))

        print(fidelity)

        return np.real(fidelity)

    def generate_circuits(self):
        settings = [key for key in self.prob_dict]
        bases = [''.join(i) for i in itertools.product('01',
                                                       repeat=len(settings[0][0]))]

        circs = {}
        for _setting in settings:
            for _base in bases:
                _sett = (_setting[0], _setting[1], _base)
                init_state, observ = self.parse_setting(_sett)
                _circ = qCirc(self.nqubits, init_state,
                              self.V, observ, self.backend, self.init_layout)
                circs[_sett] = _circ
        return circs

    def parse_setting(self, setting):
        _state, _obs, _base = setting
        init_state = []
        for i, _op in enumerate(_state):
            _s = None
            if _op == '0':
                if _base[i] == '0':
                    continue
                elif _base[i] == '1':
                    _s = GateObj(name='X', qubits=i, parameterise=False, params=None)
            elif _op == '1':
                if _base[i] == '0':
                    GateObj(name='H', qubits=i, parameterise=False, params=None)
                elif _base[i] == '1':
                    _s = GateObj(name='U3', qubits=i, parameterise=True,
                                 params=[np.pi, np.pi, 0.0])
            elif _op == '2':
                if _base[i] == '0':
                    _s = GateObj(name='U3', qubits=i, parameterise=True,
                                 params=[np.pi, np.pi/2, 0.0])
                elif _base[i] == '1':
                    _s = GateObj(name='U3', qubits=i, parameterise=True,
                                 params=[np.pi, 3*np.pi/4, 0.0])
            elif _op == '3':
                if _base[i] == '0':
                    continue
                elif _base[i] == '1':
                    _s = GateObj(name='X', qubits=i, parameterise=False)
            if _s is not None:
                init_state.append(_s)

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

        return init_state, observe

    def select_settings(self, probs, keys):
        # first choose which input states/observables to use
        choices = np.random.choice(
            [i for i in range(len(keys))], np.int(self.length/self.p_length), p=probs, replace=True)
        qutip_settings = [keys[i] for i in choices]

        # next choose which pauli eigenstates to input
        p_choices = np.random.choice(
            [''.join(i) for i in itertools.product('01', repeat=len(keys[0][0]))], self.p_length, replace=False
        )

        # qutip and qiskit use mirrored qubit naming schemes
        settings = []
        for _set in qutip_settings:
            for _pset in p_choices:
                setting0 = _set[0][::-1]
                setting1 = _set[1][::-1]
                setting2 = _pset[::-1]
                settings.append((setting0, setting1, setting2))

        settings = [item for item in settings if item[1] != '0' * self.nqubits]
        qutip_settings = [
            item for item in qutip_settings if item[1] != '0' * self.nqubits]

        return settings, qutip_settings

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
        chosen_circs = [self.circuits[_setting] for _setting in settings]
        exec_circs = [qc.populate_circuits(params) for qc in chosen_circs]
        results = self.quant_inst.execute(exec_circs, had_transpiled=True)
        expects = [
            generate_expectation(results.get_counts(i)) for i in range(len(exec_circs))
        ]

        return expects

    def generate_eigenvalue(self, state_in: str, base: str):
        test = True
        for i, _b in enumerate(base):
            if state_in[i] == '0':
                continue
            else:
                if test:
                    if _b == '1':
                        test = False
                else:
                    if _b == '1':
                        test = True
        if test:
            return 1
        else:
            return -1


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
                _params = [i for i in param]
                circ.u3(_params[0], _params[1], _params[2], qreg[q])
            else:
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

from typing import List
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from probability_distributions import ProbDist


class GateObj:
    def __init__(self, name, qubits: List, parameterise: bool = False, params: List = None):
        self.name = name
        self.qubits = qubits
        self.parameterise = parameterise
        self.params = params


class qCirc:
    """Separates state preparation, unitary evolution and measurement into separate
    sections and applies them onto a qiskit circuit
    """

    def __init__(self, nqubits: int, input_state: List[GateObj], U: List[GateObj],
                 meas_basis: List[GateObj]):
        self.nqubits = nqubits
        self.input_state = input_state
        self.U = U
        self.meas_basis = meas_basis

        self.qreg = QuantumRegister(self.nqubits)
        self.creg = ClassicalRegiter(self.nqubits)
        self.qc = QuantumCircuit(qreg, creg)

    def apply_init_state(self):
        for _gate in self.input_state:
            apply_gate(self.qc, self.qreg)

    def apply_U(self):
        for _gate in self.U:
            apply_gate(self.qc, self.qreg, _gate)

    def rotate_meas_basis(self):
        for _gate in self.meas_basis:
            apply_gate(self.qc, self.qreg)

    def build_circuit(self):
        self.apply_init_state()
        self.apply_U()
        self.rotate_meas_basis()

        for i in range(self.nqubits):
            self.qc.measure(qreg[i], creg[i])

        return self.qc


class EstimateCircuits:
    """Generate a list of circuits which when run estimate the fidelity for a given
    unitary operator, given a precalculated probability distribution
    """

    def __init__(self, prob_dist: ProbDist, U: List[GateObj], nqubits: int,
                 length: int):
        self.prob_dict = prob_dist.get_probabilities()
        self.U = U
        self.nqubits = nqubits
        self.length = length

    def generate_circuits(self):
        length = get_length()
        probs = [np.abs(self.prob_dict[key]) for key in self.prob_dict]
        keys = [key for key in self.prob_dict]
        settings = select_settings(probs, keys)
        circs = []
        for _setting in settings:
            init_state, observ = parse_setting(_setting)
            _circ = qCirc(self.nqubits, init_state, self.U, observ).build_circuit()
            circs.append(_circ)

        return circs

    def select_settings(self, probs, keys):
        choices = np.random.choice([i for i in range(len(keys))], self.length, p=probs)
        settings = [keys[i] for i in choices]
        return settings

    def parse_setting(self, setting):
        _state, _obs = setting
        init_state = []
        for i, _op in enumerate(_state):
            if _op == '0':
                continue
            elif _op == '1':
                _s = GateObj(name='X', qubits=[i], parameterise=False, params=None)
                init_state.append(_s)
            elif _op == '2':
                _s = GateObj(name='Y', qubits=[i], parameterise=False, params=None)
                init_state.append(_s)
            elif _op == '3':
                _s = GateObj(name='Z', qubits=[i], parameterise=False, params=None)
                init_state.append(_s)
        observ = []
        for i, _op in enumerate(_obs):
            if _op == '0':
                continue
            elif _op == '1':
                _o = GateObj(name='X', qubits=[i], parameterise=False, params=None)
                observe.append(_o)
            elif _op == '2':
                _o = GateObj(name='Y', qubits=[i], parameterise=False, params=None)
                observe.append(_o)
            elif _op == '3':
                _o = GateObj(name='Z', qubits=[i], parameterise=False, params=None)
                observe.append(_o)

        return init_state, observ


def apply_gate(circ: QuantumCircuit, qreg: QuantumRegister, gate: GateObj):
    """Applies a gate to a quantum circuit. More complicated gates such as RXX gates
    should be decomposed into single qubit gates and CNOTs prior to calling this
    function.
    """
    if not isinstance(gate.qubits, list):
        q = gate.qubits
        params = gate.params
        if gate.name == 'X':
            circ.x(qreg[q])
        elif gate.name == 'Y':
            circ.y(qreg[q])
        elif gate.name == 'Z':
            circ.z(qreg[q])
        elif gate.name == 'RX':
            circ.rx(params[0], qreg[q])
        elif gate.name == 'RY':
            circ.rx(params[0], qreg[q])
        elif gate.name == 'RZ':
            circ.rx(params[0], qreg[q])
        elif gate.name == 'U3':
            circ.u3(params[0], params[1], params[2], qreg[q])
    else:
        cntrl = gate.qubits[0]
        trgt = gate.qubits[1]
        circ.cx(qreg[cntrl], qreg[trgt])

    return circ

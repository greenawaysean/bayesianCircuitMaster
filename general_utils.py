from typing import List, Union
from qutip import Qobj, sigmax, sigmay, sigmaz, qeye, tensor


class GateObj:
    """Class representing an abstraction of a quantum gate.

    Designed to be an intermediary between qutip, qiskit and the optimisation algorithm.
    """

    def __init__(self, name: Union[str, List], qubits: Union[int, List],
                 parameterise: bool = None, params: List = None):
        self.name = name
        self.qubits = qubits
        self.parameterise = parameterise
        self.params = params


def U_from_hamiltonian(hamiltonian: List[GateObj], nqubits: int, t: float):
    """Generates the unitary operator from a hamiltonian

    Parameters:
    -----------
    hamiltonian: list of GateObjs defining the Hamiltonian of the system - these should
    always be simple tensor products of Pauli operators with coefficients, CNOTs and
    rotation gates are not supported and should be decomposed before calling this
    nqubits: number of qubits
    t: time for the simulation

    Returns:
    --------
    U_ideal: qutip Qobj of the ideal unitary operator
    """
    exponent = None
    for gate in hamiltonian:
        assert gate.params is not None, "Hamiltonian terms must be supplied with scalar coefficients"
        _op = []
        if isinstance(gate.qubits, int):
            for k in range(nqubits):
                if k == gate.qubits:
                    _op.append(qutip_gate(gate.name))
                else:
                    _op.append(qeye(2))
        else:
            idx = 0
            for k in range(nqubits):
                if k in gate.qubits:
                    _op.append(qutip_gate(gate.name[idx]))
                    idx += 1
                else:
                    _op.append(qeye(2))
        if exponent is None:
            exponent = gate.params*tensor(_op)
        else:
            exponent += gate.params*tensor(_op)
    ideal_U = (-1j*t*exponent).expm()

    return ideal_U


def qutip_gate(gate_name: str):
    """Generates the Pauli gate from a name

    Parameters:
    -----------
    gate_name: string representing the gate, e.g. 'X' for sigmax() etc.
    """
    if gate_name == 'X':
        return sigmax()
    elif gate_name == 'Y':
        return sigmay()
    elif gate_name == 'Z':
        return sigmaz()

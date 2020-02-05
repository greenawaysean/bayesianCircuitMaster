from typing import List


class GateObj:
    """Class representing an abstraction of a quantum gate, designed to be an
    intermediary between qiskit and the optimisation algorithm
    """

    def __init__(self, name, qubits: List, parameterise: bool = None, params: List = None):
        self.name = name
        self.qubits = qubits
        self.parameterise = parameterise
        self.params = params

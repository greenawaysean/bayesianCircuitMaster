from typing import List, Union


class GateObj:
    """Class representing an abstraction of a quantum gate, designed to be an
    intermediary between qiskit and the optimisation algorithm
    """

    def __init__(self, name, qubits: Union[int, List], parameterise: bool = None, params: List = None):
        self.name = name
        self.qubits = qubits
        self.parameterise = parameterise
        self.params = params

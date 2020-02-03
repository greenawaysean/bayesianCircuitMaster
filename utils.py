import numpy as np
from typing import List


class gateObj:
    """Class representing an abstraction of a quantum gate, designed to be an
    intermediary between qiskit and the optimisation algorithm
    """

    def __init__(self, name, qubits: List, parameterise: bool = None, params: List = None):
        self.name = name
        self.qubits = qubits
        self.parameterise = parameterise
        self.params = params

    def update_params(self, new_params: List):
        """Update the parameters given a set of optimised values
        """
        self.params = new_params


class baseCircuit:
    """Class defining a list of gateObjs corresponding to a quantum circuit.
    """

    def __init__(self):
        pass

    def generate_qiskit_circ(self):
        pass


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

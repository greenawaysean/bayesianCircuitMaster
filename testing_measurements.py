import numpy as np
from qutip import basis, qeye, sigmax, sigmay, sigmaz, tensor, gate_expand_1toN, Qobj
import itertools

a = np.random.rand()
b = np.sqrt(1-a**2)
c = np.random.rand()
d = np.sqrt(1-c**2)

init_state = tensor((a*basis(2, 0) + b*basis(2, 1)), (c*basis(2, 0) + d*basis(2, 1)))

rho = init_state*init_state.dag()

_op1 = sigmaz()
_op2 = sigmax()

_op = tensor(_op1, _op2)

print((_op*rho).tr())

_exp1 = (tensor(_op1, qeye(2))*rho).tr()
_exp2 = (tensor(qeye(2), _op2)*rho).tr()


pos_evalue_1 = (_exp1 + 1)/2
pos_evalue_2 = (_exp2 + 1)/2


combined_expect = (pos_evalue_1*pos_evalue_2 + (1 - pos_evalue_1)*(1 - pos_evalue_2) -
                   (1 - pos_evalue_1)*pos_evalue_2 - (1 - pos_evalue_2)*pos_evalue_1)

print(combined_expect)


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

    return expect/total_counts


counts = {'00': 19991, '10': 20000, '11': 8}

print(generate_expectation(counts, 2))

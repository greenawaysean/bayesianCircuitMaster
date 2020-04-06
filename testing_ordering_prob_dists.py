import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
from os import path, getcwd, makedirs
from plotter_params import plot_setup
from qutip import sigmax, sigmay, sigmaz, qeye, basis, Qobj, tensor, cnot, gate_expand_1toN, gate_expand_2toN

""" Testing whether the figure of merit F ~ Tr[U(rho_k)W_k']Tr[E(rho_k)W_k'] preserves
the ordering of circuits (i.e. if one circuit implements a higher fidelity unitary than another is this reflected in the values for the figure of merit?).
"""


def get_pauli_basis(nqubits):
    iters = [''.join(i) for i in itertools.product('0123', repeat=nqubits)]
    p_ops = {'0': qeye(2), '1': sigmax(), '2': sigmay(), '3': sigmaz()}
    basis = []
    for item in iters:
        _ops = []
        for k in item:
            _ops.append(p_ops[k])
        basis.append(tensor(_ops))
    return basis


def get_state_basis(nqubits):
    iters = [''.join(i) for i in itertools.product('0123', repeat=nqubits)]
    s_ops = {'0': Qobj([[1, 0], [0, 0]]),
             '1': Qobj([[0, 0], [0, 1]]),
             '2': Qobj([[0.5, 0.5], [0.5, 0.5]]),
             '3': Qobj([[0.5, -1j*0.5], [1j*0.5, 0.5]])}
    basis = []
    for item in iters:
        _ops = []
        for k in item:
            _ops.append(s_ops[k])
        basis.append(tensor(_ops))
    return basis


def fidelity(U, V, nqubits):
    basis = get_pauli_basis(nqubits)
    d = 2**nqubits
    sum = 0
    for op in basis:
        sum += (U*op*U.dag()*V*op*V.dag()).tr()
    return np.abs(sum/d**3)


def figure_of_merit(U, V, nqubits):
    basis = get_state_basis(nqubits)
    d = 2**nqubits
    sum = 0
    for op in basis:
        sum += (U*op*U.dag()*V*op*V.dag()).tr()
    return np.abs(sum/d**2)


def generate_u3(theta, phi, lam):
    u_00 = np.cos(theta/2)
    u_01 = -np.exp(1j*lam)*np.sin(theta/2)
    u_10 = np.exp(1j*phi)*np.sin(theta/2)
    u_11 = np.exp(1j*(lam + phi))*np.cos(theta/2)

    return Qobj([[u_00, u_01], [u_10, u_11]])


def generate_rand_herm(nqubits):
    ops = get_pauli_basis(nqubits)
    coeffs = [np.random.rand() for i in range(len(ops))]
    herm_op = None
    for i, _op in enumerate(ops):
        if herm_op is None:
            herm_op = coeffs[i]*ops[i]
        else:
            herm_op += coeffs[i]*ops[i]
    return herm_op


if __name__ == "__main__":
    nqubits = 4
    # U_herm_op = generate_rand_herm(nqubits)
    # U = (-1j*U_herm_op).expm()
    # U = cnot()
    H = None
    for i in range(nqubits):
        if H is None:
            H = 0.5*gate_expand_1toN(sigmax(), nqubits, i)
        else:
            H += 0.5*gate_expand_1toN(sigmax(), nqubits, i)
        if i < nqubits-1:
            H += gate_expand_2toN(tensor(sigmaz(), sigmaz()), nqubits, i, i+1)
    U = (-1j*H).expm()

    F_list = []
    FOM_list = []

    for k in range(20000):
        eps = 0.00005*k
        herm_op = generate_rand_herm(nqubits)
        # V = (((-1j*eps)*herm_op).expm())*copy.deepcopy(U)*(((1j*eps)*herm_op).expm())
        V = copy.deepcopy(U)*(((-1j*eps)*herm_op).expm())

        F = fidelity(U, V, nqubits)
        if F >= 1.0:
            F = 1.0
        FOM = figure_of_merit(U, V, nqubits)
        if FOM >= 1.0:
            print(FOM)
            F = 1.0

        F_list.append(F)
        FOM_list.append(FOM)

    filename = path.join(getcwd(), 'data', 'FOM_evals')
    if not path.exists(filename):
        makedirs(filename)

    plot_setup()
    plt.scatter(F_list, FOM_list, marker='o', facecolors='none',
                edgecolors='blue', s=7.5, alpha=0.2)
    plt.plot(F_list, F_list, color='black')
    plt.xlabel('$F$')
    plt.ylabel('$FOM$')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.savefig(path.join(filename, 'four_qubit_ising.png'))
    plt.show()

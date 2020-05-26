import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
from os import path, getcwd, makedirs
from plotter_params import plot_setup
from qutip import sigmax, sigmay, sigmaz, qeye, basis, Qobj, tensor, cnot, gate_expand_1toN, gate_expand_2toN
from scipy import linalg

""" Testing whether the figure of merit F ~ Tr[U(rho_k)W_k']Tr[E(rho_k)W_k'] preserves
the ordering of circuits (i.e. if one circuit implements a higher fidelity unitary than another is this reflected in the values for the figure of merit?).
"""


def choose(n, k):
    return np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(np.int(n-k)))


def get_pauli_basis(nqubits):
    iters = [''.join(i) for i in itertools.product('0123', repeat=nqubits)]
    p_ops = {'0': qeye(2), '1': sigmax(), '2': sigmay(), '3': sigmaz()}
    basis = []
    for item in iters:
        _ops = []
        for k in item:
            _ops.append(p_ops[k])
        basis.append(tensor(_ops).full())
    return basis


def get_state_basis(nqubits):
    iters = [''.join(i) for i in itertools.product('0123', repeat=nqubits)]
    theta = 2*np.arctan(np.sqrt(2))

    A = Qobj([[1.0, 0.0]])
    B = Qobj([[np.cos(theta/2), np.sin(theta/2)]])
    C = Qobj([[np.cos(theta/2), np.exp(1j*2*np.pi/3)*np.sin(theta/2)]])
    D = Qobj([[np.cos(theta/2), np.exp(1j*4*np.pi/3)*np.sin(theta/2)]])

    s_ops = {'0': A.dag()*A,
             '1': B.dag()*B,
             '2': C.dag()*C,
             '3': D.dag()*D
             }

    basis = []
    for item in iters:
        _ops = []
        for k in item:
            _ops.append(s_ops[k])
        basis.append(tensor(_ops).full())
    return basis


def fidelity(U, V, nqubits, basis=None):
    if basis is None:
        basis = get_pauli_basis(nqubits)
    d = 2**nqubits
    sum = 0
    for op in basis:
        _rho = np.dot(np.dot(U, op), np.conj(np.transpose(U)))
        _sigma = np.dot(np.dot(V, op), np.conj(np.transpose(V)))
        _trace = np.dot(_rho, _sigma)
        _trace = _trace.diagonal()
        sum += _trace.sum()
    return sum/d**3


def figure_of_merit(U, V, nqubits, Bmat=None, basis=None, rot_angle=0):
    if basis is None:
        basis = get_state_basis(nqubits)
    rot_U = linalg.expm(-1j*rot_angle*generate_rand_herm(nqubits))
    d = 2**nqubits
    sum = 0
    if Bmat is not None:
        for i, op1 in enumerate(basis):
            for j, op2 in enumerate(basis):
                coeff = Bmat[i][j]
                if coeff == 0:
                    continue
                _op1 = np.dot(np.dot(rot_U, op1), np.conj(np.transpose(rot_U)))
                _op2 = np.dot(np.dot(rot_U, op2), np.conj(np.transpose(rot_U)))
                _rho = np.dot(np.dot(U, _op1), np.conj(np.transpose(U)))
                _sigma = np.dot(np.dot(V, _op2), np.conj(np.transpose(V)))
                _trace = np.dot(_rho, _sigma)
                _trace = _trace.diagonal().sum()
                sum += coeff*_trace
    else:
        for op in basis:
            op = np.dot(np.dot(rot_U, op), np.conj(np.transpose(rot_U)))
            _rho = np.dot(np.dot(U, op), np.conj(np.transpose(U)))
            _sigma = np.dot(np.dot(V, op), np.conj(np.transpose(V)))
            _trace = np.dot(_rho, _sigma)
            _trace = _trace.diagonal()
            sum += _trace.sum()
    return sum/d**2


def generate_u3(theta, phi, lam):
    u_00 = np.cos(theta/2)
    u_01 = -np.exp(1j*lam)*np.sin(theta/2)
    u_10 = np.exp(1j*phi)*np.sin(theta/2)
    u_11 = np.exp(1j*(lam + phi))*np.cos(theta/2)

    return Qobj([[u_00, u_01], [u_10, u_11]])


def generate_rand_herm(nqubits, basis=None):
    if basis is None:
        basis = get_pauli_basis(nqubits)
    ops = np.array(basis)
    coeffs = np.array([np.random.rand() for i in range(len(basis))])
    herm_op = np.array([coeffs[i]*ops[i] for i in range(len(coeffs))]).sum(axis=0)

    return herm_op


def generate_Bmat(nqubits, order):
    _alpha = 1
    P = Qobj([[0.25]*4]*4) - 0.5*qeye(4)
    B_inv = None
    for k in range(order+1):
        _beta = (-1)**k
        s = [qeye(4)]*(nqubits - k) + [P]*k
        X_k = None
        track_ops = []
        for i in itertools.permutations(s):
            if i in track_ops:
                continue
            track_ops.append(i)
            if X_k is None:
                X_k = tensor(list(i))
            else:
                X_k += tensor(list(i))
        if B_inv is None:
            B_inv = _alpha*_beta*X_k
        else:
            B_inv += _alpha*_beta*X_k
    return B_inv.full()


if __name__ == "__main__":
    plot_setup()
    nqubits = 4
    W_basis = get_pauli_basis(nqubits)
    rho_basis = get_state_basis(nqubits)
    U_herm_op = generate_rand_herm(nqubits, W_basis)
    U = linalg.expm((-1j*U_herm_op))

    lim = 0.0

    for order in range(nqubits+1):
        Bmat = generate_Bmat(nqubits, order)
        F_list = []
        FOM_list = []
        num_steps = 500
        for k in range(num_steps):
            eps = (1/num_steps)*k
            herm_op = generate_rand_herm(nqubits, W_basis)
            V = np.dot(U, linalg.expm((-1j*eps)*herm_op))

            F = fidelity(U, V, nqubits, basis=W_basis)
            if F >= 1.0:
                F = 1.0
            FOM = figure_of_merit(U, V, nqubits, Bmat, basis=rho_basis)
            # if FOM >= 1.0:
            #     print(FOM)
            #     F = 1.0

            F_list.append(F)
            FOM_list.append(FOM)

        if min(FOM_list) < lim:
            lim = min(FOM_list)
        # plt.scatter(F_list, FOM_list, marker='o', facecolors='none',
        #             edgecolors='blue', s=7.5, alpha=0.2)
        plt.scatter(F_list, FOM_list, marker='o', s=12.5, alpha=0.4, label=f'k={order}')
        filename = path.join(getcwd(), 'data', 'alt_FOM_expansions')
        if not path.exists(filename):
            makedirs(filename)
        plt.plot(F_list, F_list, color='black')
        plt.xlabel('$F$')
        plt.ylabel('$FOM$')
        # plt.ylim(lim, 1.0)
        # plt.xlim(0.0, 1.0)
        plt.legend(markerscale=2.)
    # plt.savefig(path.join(filename, f'n{nqubits}_k1.png'))
    plt.savefig(path.join(filename, f'n{nqubits}_all_orders.png'))
    plt.show()

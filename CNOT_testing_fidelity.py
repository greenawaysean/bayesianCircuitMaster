import numpy as np
from qutip import (sigmax, sigmay, sigmaz, qeye, basis, gate_expand_1toN, Qobj, tensor,
                   snot, rz)
from scipy import linalg
import copy
import itertools
import matplotlib.pyplot as plt

from os import path, makedirs, getcwd
import pickle

from plotter_params import plot_setup

from typing import List, Union

from general_utils import GateObj

from circuit_utils import qCirc, EstimateCircuits, apply_gate, generate_expectation
from alt_fom_expansion_different_unitaries import generate_Bmat
from probability_distributions_numpy import ChiProbDist, ProbDist

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise.errors import ReadoutError


class TrueFidelityEst(EstimateCircuits):
    def __init__(self, prob_dist: ProbDist, V: List[GateObj], nqubits: int,
                 num_shots: int, backend: str, init_layout: dict, noise_model=None, params=None):

        super().__init__(prob_dist, V, nqubits, num_shots, backend, init_layout, noise_model)

        self.expects = self.run_circuits(params)

    def run_circuits(self, params=None):
        print(len(self.circuits))
        perms = [''.join(i) for i in itertools.product('0123', repeat=self.nqubits)]
        self.B_dict = {}
        for i, p in enumerate(perms):
            self.B_dict[p] = i
        if len(self.circuits) > 900:
            expects = []
            length = np.int((len(self.circuits) / 900) + 1)
            keys = [key for key in self.circuits]
            for i in range(length):
                _min = min(len(self.circuits)+1, 900 * i + 900)
                print(i, _min)
                _keys = keys[900 * i: _min]
                if params is not None:
                    exec_circs = [self.circuits[qc].populate_circuits(params) for qc in _keys]
                else:
                    exec_circs = [self.circuits[qc].qc for qc in _keys]
                results = self.quant_inst.execute(
                    exec_circs, had_transpiled=True)

                q_list = [i for i in range(self.nqubits)][::-1]
                _exp = []
                for i, _c in enumerate(_keys):
                    _ig = []
                    for j, _b in enumerate(_c[1]):
                        if _b == '0':
                            _ig.append(j)
                    _ignore = [q_list[i] for i in _ig]
                    # _ignore = []
                    _exp.append(generate_expectation(results.get_counts(i), _ignore))

                # _exp = [
                #     generate_expectation(results.get_counts(i)) for i in range(len(exec_circs))
                # ]

                expects += _exp
                print(len(expects))
        else:
            if params is not None:
                print('params is not none')
                exec_circs = [self.circuits[qc].populate_circuits(
                    params) for qc in self.circuits]
            else:
                exec_circs = [self.circuits[qc].qc for qc in self.circuits]
            # perms = [''.join(i) for i in itertools.product('0123', repeat=self.nqubits)]
            # self.B_dict = {}
            # for i, p in enumerate(perms):
            #     self.B_dict[p] = i
            results = self.quant_inst.execute(exec_circs, had_transpiled=True)
            keys = [key for key in self.circuits]
            q_list = [i for i in range(self.nqubits)][::-1]
            expects = []
            for i, _c in enumerate(keys):
                _ig = []
                for j, _b in enumerate(_c[1]):
                    if _b == '0':
                        _ig.append(j)
                _ignore = [q_list[i] for i in _ig]
                # _ignore = []
                expects.append(generate_expectation(results.get_counts(i), _ignore))

            # expects = [
            #     generate_expectation(results.get_counts(i)) for i in range(len(exec_circs))
            # ]


        # pm = 0.1
        # for exp in expects:
        #     if exp > 1-pm:
        #         exp -= pm*np.random.rand()*exp
        #     elif exp < -(1-pm):
        #         exp += pm*np.random.rand()*exp
        #     else:
        #         exp += (2*pm*np.random.rand() - pm)*exp
        return expects

    def calculate_FOM(self):
        d = 2**self.nqubits
        chis = [self.chi_dict[key] for key in self.chi_dict]
        FOM = 0
        for i, chi in enumerate(chis):
            FOM += chi*self.expects[i]
        return FOM/d**3

    def calculate_F(self):
        d = 2**self.nqubits
        Bmat = generate_Bmat(self.nqubits, self.nqubits)
        F = 0
        chis = [self.chi_dict[key] for key in self.chi_dict]
        chi_keys = [key for key in self.chi_dict]
        keys = [key[0] for key in self.chi_dict]
        for i, _key in enumerate(chi_keys):
            chi = self.chi_dict[_key]
            for j, exp in enumerate(self.expects):
                _set1 = self.B_dict[keys[i]]
                _set2 = self.B_dict[keys[j]]
                F += Bmat[_set1, _set2]*chi*exp
        # for i, chi in enumerate(chis):
        #     for j in range(len(chis)):
        #         _set1 = self.B_dict[keys[i]]
        #         _set2 = self.B_dict[keys[j]]
        #         F += Bmat[_set1, _set2]*chi*self.expects[j]
        return F/d**3

    def plot_expects_errors(self, it=0):
        plot_setup()
        chis = [self.chi_dict[key] for key in self.chi_dict]
        res = [self.expects[i] - chis[i] for i in range(len(chis))]
        if it == 0:
            plt.scatter(chis, res, marker='o', facecolors='none',
                        edgecolors='#1f77b4', s=15, alpha=0.7, label='Noisy emulator')
        elif it == 1:
            plt.scatter(chis, res, marker='o', facecolors='none',
                        edgecolors='#ff7f0e', s=15, alpha=0.7, label='IBMQ Paris')
        plt.plot([-2 + 0.01*i for i in range(1000)],
                 [0 for i in range(1000)], color='black', linestyle='--', linewidth=1)
        plt.xlabel('Ideal expectation value')
        plt.ylabel('Error in expectation value')
        plt.xlim(-1.02, 1.02)


if __name__ == "__main__":
    from qutip import cnot, toffoli
    from qiskit import QuantumRegister, IBMQ, Aer

    IBMQ.load_account()
    provider = IBMQ.get_provider(group='samsung', project='imperial')

    nqubits = 3
    theta = 0.0
    # U = toffoli()
    U = rz(theta, N=3, target=0)*rz(theta, N=3, target=1)*cnot(nqubits,
                                                               0, nqubits-1)*rz(theta, N=3, target=0)*rz(theta, N=3, target=1)

    device = provider.get_backend('ibmq_paris')
    properties = device.properties()
    noise_paris = noise.device.basic_device_noise_model(properties)

    backend = Aer.get_backend('qasm_simulator')
    # real_backend = provider.get_backend('ibmq_paris')
    real_backend = provider.get_backend('ibmq_qasm_simulator')

    prob_dist = ChiProbDist(nqubits=nqubits, U=U)

    qreg = QuantumRegister(nqubits, name='qreg')
    # init_layout = {12: qreg[0], 13: qreg[1]}
    init_layout = {0: qreg[0], 1: qreg[1], 2: qreg[2]}
    ansatz = [GateObj('CNOT', [0, 1], False),
              GateObj('CNOT', [1, 2], False),
              GateObj('CNOT', [0, 1], False),
              GateObj('CNOT', [1, 2], False)]

    # est_circ = TrueFidelityEst(prob_dist, ansatz, nqubits=nqubits, length=300,
    #                            num_shots=8192, backend=backend, init_layout=init_layout, noise_model=noise_paris)
    #
    # print(est_circ.calculate_FOM())
    # print(est_circ.calculate_F())
    #
    # est_circ.plot_expects_errors(it=0)

    est_circ = TrueFidelityEst(prob_dist, ansatz, nqubits=nqubits,
                               num_shots=8192, backend=real_backend, init_layout=init_layout, noise_model=None)

    est_circ.plot_expects_errors(it=1)

    plt.legend(markerscale=2.)

    # filename = path.join(getcwd(), 'data', 'noisy_expectations', 'CNOT')
    # if not path.exists(filename):
    #     makedirs(filename)
    # plt.savefig(path.join(filename, 'noisy_expect_Paris.png'))
    plt.show()
    print(est_circ.calculate_FOM())
    print(est_circ.calculate_F())

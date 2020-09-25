from typing import List
import numpy as np
from os import path, getcwd, makedirs
import sys
from general_utils import GateObj, U_from_hamiltonian, get_filename
from circuit_utils import qCirc, EstimateCircuits, ParallelEstimateCircuits
from probability_distributions_numpy import ChiProbDist
from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qutip import Qobj, cnot, toffoli
from GPyOpt_fork.GPyOpt import GPyOpt
import pickle


# sys.path.insert(0, path.join(sys.path[0], 'bayesianCircuitMaster'))
#
# savepath = path.join(getcwd(), 'data', 'BO', 'real_machine', 'batch_results',
#                      '3q_CNOT_batch')
# savepath = get_filename(savepath)

ideal_U = cnot(3, 2, 0)
prob_dist = ChiProbDist(nqubits=3, U=ideal_U)

# define list of GateObjs which will be our estimate ansatz
ansatz = []
ansatz.append(GateObj('U3', 0, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 1, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 2, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('U3', 0, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 1, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 2, True, (0.0, 0.0, 0.0)))

_vals = [0.0]*18

load = True
if load:
    IBMQ.load_account()
provider = IBMQ.get_provider(group='samsung', project='imperial')
# backend = provider.get_backend('ibmq_singapore')
backend = Aer.get_backend('qasm_simulator')

qreg = QuantumRegister(3, name='qreg')
init_layout1 = {0: qreg[0], 1: qreg[1], 2: qreg[2]}

est_circ_1 = EstimateCircuits(prob_dist, ansatz, nqubits=3, num_shots=512,
                            backend=backend, init_layout=init_layout1)

provider = IBMQ.get_provider(group='samsung', project='imperial')
# backend = provider.get_backend('ibmq_singapore')
backend = Aer.get_backend('qasm_simulator')

qreg = QuantumRegister(3, name='qreg')
init_layout2 = {3: qreg[0], 4: qreg[1], 5: qreg[2]}

est_circ_2 = EstimateCircuits(prob_dist, ansatz, nqubits=3, num_shots=512,
                            backend=backend, init_layout=init_layout2)

circ_list = [est_circ_1, est_circ_2]

qreg = QuantumRegister(6, name='qreg')
overall_layout = {i: qreg[i] for i in range(6)}

p_est_circs = ParallelEstimateCircuits(circ_list, 6, backend, 512, overall_layout)

fidels = p_est_circs.parallel_fidelities([_vals, _vals], 150)

print(fidels)

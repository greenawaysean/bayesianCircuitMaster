from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qutip import Qobj, cnot
from general_utils import GateObj, U_from_hamiltonian
from circuit_utils import qCirc, EstimateCircuits
from probability_distributions import ChiProbDist
import numpy as np
import matplotlib.pyplot as plt
from GPyOpt_fork.GPyOpt import GPyOpt
import sys
from os import path, getcwd, makedirs
import pickle

sys.path.insert(0, path.join(sys.path[0], 'bayesianCircuitMaster'))

savepath = path.join(getcwd(), 'data', 'BO', 'real_machine', 'CCNOT_opt_1')

if not path.exists(savepath):
    makedirs(savepath)

ideal_U = ideal_cnot = cnot(3, 2, 0)

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

prob_dist = ChiProbDist(nqubits=2, U=ideal_U)

load = True
if load:
    IBMQ.load_account()
provider = IBMQ.get_provider(group='samsung', project='imperial')
# backend = provider.get_backend('ibmq_singapore')
backend = provider.get_backend('ibmq_qasm_simulator')

qreg = QuantumRegister(3, name='qreg')
init_layout = {0: qreg[0], 1: qreg[1], 6: qreg[2]}

est_circ = EstimateCircuits(prob_dist, ansatz, nqubits=3, length=150,
                            num_shots=8192, backend=backend, init_layout=init_layout)


def F(params):
    return 1 - est_circ.calculate_fidelity(params[0])


NB_INIT = 50
NB_ITER = 50
DOMAIN_DEFAULT = [(val - np.pi/4, val + np.pi/4) for i, val in enumerate(_vals)]
# DOMAIN_DEFAULT = [(0, 2*np.pi) for i, val in enumerate(_vals)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d}
             for i, d in enumerate(DOMAIN_DEFAULT)]

BO_ARGS_DEFAULT = {'domain': DOMAIN_BO, 'initial_design_numdata': NB_INIT,
                   'model_update_interval': 1, 'hp_update_interval': 5,
                   'acquisition_type': 'LCB', 'acquisition_weight': 5,
                   'acquisition_weight_lindec': True, 'optim_num_anchor': 5,
                   'optimize_restarts': 1, 'optim_num_samples': 10000, 'ARD': False}

myBopt = GPyOpt.methods.BayesianOptimization(f=F, **BO_ARGS_DEFAULT)
myBopt.run_optimization(max_iter=NB_ITER)
myBopt.plot_acquisition(path.join(savepath, 'acquisition_plot.png'))
myBopt.plot_convergence(path.join(savepath, 'convergence_plot.png'))

with open(path.join(savepath, 'model_data.pickle'), 'wb') as f:
    pickle.dump(myBopt, f)

(x_seen, y_seen), (x_exp, y_exp) = myBopt.get_best()

with open(path.join(savepath, 'best_params.pickle'), 'wb') as f:
    pickle.dump(((x_seen, y_seen), (x_exp, y_exp)), f)

regular_cnot = est_circ.calculate_fidelity([0.0]*18)
seen_opt_cnot = est_circ.calculate_fidelity(x_seen)
exp_opt_cnot = est_circ.calculate_fidelity(x_exp)

with open(path.join(savepath, 'FOMs.pickle'), 'wb') as f:
    pickle.dump((regular_cnot, seen_opt_cnot, exp_opt_cnot), f)

# test the optimised cnot on a two-qubit Ising model


# def apply_cnot(qc, params, cntrl, trgt):
#     qc.u3(theta=params[0], phi=params[1], lam=params[2], qubit=cntrl)
#     qc.u3(params[3], params[4], params[5], trgt)
#     qc.cx(cntrl, trgt)
#     qc.u3(params[6], params[7], params[8], cntrl)
#     qc.u3(params[9], params[10], params[11], trgt)
#
#
# tsteps = 4
# tlist = [0.1*i for i in range(40)]
#
# normal_circs = []
# opt_circs = []
#
# for t in tlist:
#     qreg = QuantumRegister(2, name='qreg')
#     creg = ClassicalRegister(2)
#     qc = QuantumCircuit(qreg, creg)
#
#     init_layout = {0: qreg[0], 1: qreg[1]}
#     for i in range(tsteps):
#         for k in range(2):
#             qc.rx(2*0.5*t/tsteps, qreg[k])
#         _params = x_exp
#         apply_cnot(qc, _params, qreg[0], qreg[1])
#         qc.rz(2*t/tsteps, qreg[1])
#         apply_cnot(qc, _params, qreg[0], qreg[1])
#         for k in range(2):
#             qc.rx(2*0.5*t/tsteps, qreg[k])
#
#     opt_circs.append(qc)
#
#     qreg = QuantumRegister(2, name='qreg')
#     creg = ClassicalRegister(2)
#     qc = QuantumCircuit(qreg, creg)
#
#     init_layout = {0: qreg[0], 1: qreg[1]}
#     for i in range(tsteps):
#         for k in range(2):
#             qc.rx(2*0.5*t/tsteps, qreg[k])
#         qc.cx(qreg[0], qreg[1])
#         qc.rz(2*t/tsteps, qreg[1])
#         qc.cx(qreg[0], qreg[1])
#         for k in range(2):
#             qc.rx(2*0.5*t/tsteps, qreg[k])
#
#     normal_circs.append(qc)
#
# quant_inst = QuantumInstance(backend=backend, shots=8192)
# circs = opt_circs + normal_circs
# job = quant_inst.execute(circs)
#
# opt_counts = []
# for i in range(len(tlist)):
#     opt_counts.append(job.get_counts(i))
#
# normal_counts = []
# for i in range(len(tlist), np.int(2*len(tlist))):
#     normal_counts.append(job.get_counts(i))
#
# with open(path.join(savepath, 'ising_results.pickle'), 'wb') as f:
#     pickle.dump((opt_counts, normal_counts), f)

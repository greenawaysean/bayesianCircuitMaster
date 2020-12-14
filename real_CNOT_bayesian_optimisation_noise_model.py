from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qutip import Qobj, cnot
from general_utils import GateObj, U_from_hamiltonian
from circuit_utils import qCirc, EstimateCircuits
from probability_distributions_numpy import ChiProbDist
from CNOT_testing_fidelity import TrueFidelityEst
import numpy as np
import matplotlib.pyplot as plt
from GPyOpt_fork.GPyOpt import GPyOpt
import sys
from os import path, getcwd, makedirs
import pickle

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

sys.path.insert(0, path.join(sys.path[0], 'bayesianCircuitMaster'))

def get_filename(filename):
    """ Ensures that a unique filename is used with consequential numbering
    """
    if not path.exists(filename):
        makedirs(filename)
        filename = filename
    else:
        test = False
        idx = 2
        filename += f'_{idx}'
        while not test:
            if not path.exists(filename):
                makedirs(filename)
                filename = filename
                test = True
            else:
                idx += 1
                filename = filename[:-(len(str(idx-1))+1)] + f'_{idx}'
    return filename


savepath = path.join(getcwd(), 'data', 'BO', 'noisy_simulations', 'testing_rochester',
                     '3q_CNOT_batch')
savepath = get_filename(savepath)

if not path.exists(savepath):
    makedirs(savepath)

ideal_U = cnot(3, 2, 0)
prob_dist = ChiProbDist(nqubits=3, U=ideal_U)

reversed_U = cnot(3, 0, 2)
rev_prob_dist = ChiProbDist(nqubits=3, U=reversed_U)

load = True
if load:
    IBMQ.load_account()
load = False
provider = IBMQ.get_provider(group='samsung', project='imperial')
# backend = provider.get_backend('ibmq_rochester')
# noise_model = NoiseModel.from_backend(backend)
backend = Aer.get_backend('qasm_simulator')

# Error probabilities
prob_1 = 0.0003  # 1-qubit gate
prob_2 = 0.008   # 2-qubit gate
prob_3 = 0.03 # readout error

# Depolarizing quantum errors
error_1 = depolarizing_error(prob_1, 1)
error_2 = depolarizing_error(prob_2, 2)
error_3 = depolarizing_error(prob_3, 1)

# Add errors to noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
noise_model.add_all_qubit_quantum_error(error_3, 'measure')

qreg = QuantumRegister(3, name='qreg')
init_layout = {0: qreg[0], 1: qreg[1], 2: qreg[2]}


val_list = [0.15, 0.5, 0.75]

results = []
z0_results = []
# for err_x in val_list:
#     for err_y in val_list:
#         for err_z in val_list:
_r_res = 0
_e_res = 0
# define list of GateObjs which will be our estimate ansatz
ansatz = []

ansatz.append(GateObj('U3', 0, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 1, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 2, True, (0.0, 0.0, 0.0)))
# ansatz.append(GateObj('RZ', 0, True, 0.0))
# ansatz.append(GateObj('RZ', 1, True, 0.0))
# ansatz.append(GateObj('RZ', 2, True, 0.0))

# Add coherent gates for error
theta_errors = val_list
# for i in range(3):
#     for j, _op in enumerate(['RX', 'RY', 'RZ']):
#         ansatz.append(GateObj(_op, i, False, theta_errors[j]))

ansatz.append(GateObj('CNOT', [0, 1], False))
for i in [0,1]:
    for j, _op in enumerate(['RZ']):
        ansatz.append(GateObj(_op, i, False, theta_errors[j]))
ansatz.append(GateObj('CNOT', [1, 2], False))
for i in [1,2]:
    for j, _op in enumerate(['RZ']):
        ansatz.append(GateObj(_op, i, False, theta_errors[j]))
ansatz.append(GateObj('CNOT', [0, 1], False))
for i in [0,1]:
    for j, _op in enumerate(['RZ']):
        ansatz.append(GateObj(_op, i, False, theta_errors[j]))
ansatz.append(GateObj('CNOT', [1, 2], False))
for i in [1,2]:
    for j, _op in enumerate(['RZ']):
        ansatz.append(GateObj(_op, i, False, theta_errors[j]))
# ansatz.append(GateObj('RZ', 0, True, 0.0))
# ansatz.append(GateObj('RZ', 1, True, 0.0))
# ansatz.append(GateObj('RZ', 2, True, 0.0))
ansatz.append(GateObj('U3', 0, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 1, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 2, True, (0.0, 0.0, 0.0)))

_vals = [0.0]*18

est_circ = EstimateCircuits(prob_dist, ansatz, nqubits=3,
                            num_shots=2048, backend=backend, init_layout=init_layout,
                            noise_model=noise_model)

# _vals = [0.0]*6

# load = True
# if load:
#     IBMQ.load_account()
# load = False
# provider = IBMQ.get_provider(group='samsung', project='imperial')
# # backend = provider.get_backend('ibmq_rochester')
# # noise_model = NoiseModel.from_backend(backend)
# backend = Aer.get_backend('qasm_simulator')
#
# # Error probabilities
# prob_1 = 0.0003  # 1-qubit gate
# prob_2 = 0.008   # 2-qubit gate
# prob_3 = 0.05 # readout error
#
# # Depolarizing quantum errors
# error_1 = depolarizing_error(prob_1, 1)
# error_2 = depolarizing_error(prob_2, 2)
# error_3 = depolarizing_error(prob_3, 1)
#
# # Add errors to noise model
# noise_model = NoiseModel()
# noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
# noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
# noise_model.add_all_qubit_quantum_error(error_3, 'measure')
#
# qreg = QuantumRegister(3, name='qreg')
# init_layout = {0: qreg[0], 1: qreg[1], 2: qreg[2]}
#
# est_circ = EstimateCircuits(prob_dist, ansatz, nqubits=3,
#                             num_shots=1024, backend=backend, init_layout=init_layout,
#                             noise_model=noise_model)


def F(params):
    return 1 - est_circ.calculate_fidelity(params[0], length=150)

NB_INIT = 70
NB_ITER = 70
DOMAIN_DEFAULT = [(val - np.pi/8, val + np.pi/8) for i, val in enumerate(_vals)]
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
# myBopt.plot_acquisition(path.join(savepath, 'acquisition_plot.png'))
# myBopt.plot_convergence(path.join(savepath, 'convergence_plot.png'))

# with open(path.join(savepath, 'model_data.pickle'), 'wb') as f:
#     pickle.dump(myBopt, f)

(x_seen, y_seen), (x_exp, y_exp) = myBopt.get_best()

# with open(path.join(savepath, 'best_params.pickle'), 'wb') as f:
#     pickle.dump(((x_seen, y_seen), (x_exp, y_exp)), f)

regular_cnot = est_circ.calculate_fidelity([0.0]*18, length=500)
seen_opt_cnot = est_circ.calculate_fidelity(x_seen, length=500)
exp_opt_cnot = est_circ.calculate_fidelity(x_exp, length=500)

print(regular_cnot, seen_opt_cnot, exp_opt_cnot)

_r_res += regular_cnot
_e_res += exp_opt_cnot

with open(path.join(savepath, 'FOMs.pickle'), 'wb') as f:
    pickle.dump((regular_cnot, seen_opt_cnot, exp_opt_cnot), f)

reg_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                    num_shots=2048, backend=backend, init_layout=init_layout,
                                    params=[0.0]*18, noise_model=noise_model)

regular_true_F = reg_true_est_circ.calculate_F()
regular_true_FOM = reg_true_est_circ.calculate_FOM()

print(regular_true_F)
print(regular_true_FOM)

seen_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                     num_shots=2048, backend=backend, init_layout=init_layout,
                                     params=x_seen, noise_model=noise_model)

seen_true_F = seen_true_est_circ.calculate_F()
seen_true_FOM = seen_true_est_circ.calculate_FOM()

print(seen_true_F)
print(seen_true_FOM)

exp_opt_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                        num_shots=2048, backend=backend, init_layout=init_layout,
                                        params=x_exp, noise_model=noise_model)

exp_opt_true_F = exp_opt_true_est_circ.calculate_F()
exp_opt_true_FOM = exp_opt_true_est_circ.calculate_FOM()

print(exp_opt_true_F)
print(exp_opt_true_FOM)

with open(path.join(savepath, 'fidels.pickle'), 'wb') as f:
    pickle.dump((regular_true_F, seen_true_F, exp_opt_true_F), f)


with open(path.join(savepath, 'true_FOMs.pickle'), 'wb') as f:
    pickle.dump((regular_true_FOM, seen_true_FOM, exp_opt_true_FOM), f)
results.append(_e_res - _r_res)
if err_z == 0.0:
    z0_results.append(_e_res - _r_res)
else:
    z0_results.append(0.0)


# plt.plot([0.25*i for i in range(3)], _r_res, label='regular')
# plt.plot([0.25*i for i in range(3)], _s_res, label='seen')
plt.scatter([i for i in range(len(results))], results)
plt.scatter([i for i in range(len(z0_results))], z0_results)
plt.show()

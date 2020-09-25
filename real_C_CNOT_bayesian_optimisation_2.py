from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qutip import Qobj, cnot, toffoli
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
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise.errors import ReadoutError

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


savepath = path.join(getcwd(), 'data', 'BO', 'real_machine',
                     'toffoli_optimisation', 'TEST')
savepath = get_filename(savepath)

ideal_U = toffoli(N=3, controls=[2, 1], target=0)
prob_dist = ChiProbDist(nqubits=3, U=ideal_U)

reversed_U = toffoli(N=3, controls=[0, 1], target=2)
rev_prob_dist = ChiProbDist(nqubits=3, U=reversed_U)

# define list of GateObjs which will be our estimate ansatz
vals = []
ansatz = []
ansatz.append(GateObj('U3', 0, True, (0.0, 0.0, 0.0)))
vals += [0.0]*3
ansatz.append(GateObj('U3', 1, True, (0.0, 0.0, 0.0)))
vals += [0.0]*3
ansatz.append(GateObj('U3', 2, True, (np.pi/2, 0.0, np.pi)))
vals.append(np.pi/2)
vals.append(0.0)
vals.append(np.pi)
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('RZ', 2, True, -np.pi/4))
vals.append(-np.pi/4)
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('RZ', 2, True, np.pi/4))
vals.append(np.pi/4)
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('RZ', 2, True, -np.pi/4))
vals.append(-np.pi/4)
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('RZ', 1, True, np.pi/4))
vals.append(np.pi/4)
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('RZ', 0, True, np.pi/4))
vals.append(np.pi/4)
ansatz.append(GateObj('RZ', 1, True, -np.pi/4))
vals.append(-np.pi/4)
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('U3', 0, True, (0.0, 0.0, 0.0)))
vals += [0.0]*3
ansatz.append(GateObj('U3', 1, True, (0.0, 0.0, 0.0)))
vals += [0.0]*3
ansatz.append(GateObj('U3', 2, True, (np.pi/2, np.pi/4, np.pi)))
vals.append(np.pi/2)
vals.append(0.)
vals.append(5*np.pi/4)

load = True
if load:
    IBMQ.load_account()
provider = IBMQ.get_provider(group='samsung', project='imperial')
# backend = provider.get_backend('ibmq_manhattan')
backend = Aer.get_backend('qasm_simulator')

qreg = QuantumRegister(3, name='qreg')
init_layout = {0: qreg[0], 1: qreg[1], 2: qreg[2]}

#######################################################################################
# device = provider.get_backend('ibmq_singapore')
# properties = device.properties()
# noise_paris = noise.device.basic_device_noise_model(properties)
#######################################################################################

est_circ = EstimateCircuits(prob_dist, ansatz, nqubits=3,
                            num_shots=2048, backend=backend, init_layout=init_layout)

print('done est circs')


def F(params):
    return 1 - est_circ.calculate_fidelity(params[0], length=200)


NB_INIT = 50
NB_ITER = 120
DOMAIN_DEFAULT = [(val - np.pi/4, val + np.pi/4) for i, val in enumerate(vals)]
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

est_circ.num_shots = 8192

regular_cnot = est_circ.calculate_fidelity(vals, length=500)
seen_opt_cnot = est_circ.calculate_fidelity(x_seen, length=500)
exp_opt_cnot = est_circ.calculate_fidelity(x_exp, length=500)

print(regular_cnot, seen_opt_cnot, exp_opt_cnot)

with open(path.join(savepath, 'FOMs.pickle'), 'wb') as f:
    pickle.dump((regular_cnot, seen_opt_cnot, exp_opt_cnot), f)

reg_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                    num_shots=8192, backend=backend, init_layout=init_layout,
                                    params=vals)

regular_true_F = reg_true_est_circ.calculate_F()
regular_true_FOM = reg_true_est_circ.calculate_FOM()

print(regular_true_F)
print(regular_true_FOM)

seen_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                     num_shots=8192, backend=backend, init_layout=init_layout,
                                     params=x_seen)

seen_true_F = seen_true_est_circ.calculate_F()
seen_true_FOM = seen_true_est_circ.calculate_FOM()

print(seen_true_F)
print(seen_true_FOM)

exp_opt_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                        num_shots=8192, backend=backend, init_layout=init_layout,
                                        params=x_exp)

exp_opt_true_F = exp_opt_true_est_circ.calculate_F()
exp_opt_true_FOM = exp_opt_true_est_circ.calculate_FOM()

print(exp_opt_true_F)
print(exp_opt_true_FOM)

with open(path.join(savepath, 'fidels.pickle'), 'wb') as f:
    pickle.dump((regular_true_F, seen_true_F, exp_opt_true_F), f)


with open(path.join(savepath, 'true_FOMs.pickle'), 'wb') as f:
    pickle.dump((regular_true_FOM, seen_true_FOM, exp_opt_true_FOM), f)

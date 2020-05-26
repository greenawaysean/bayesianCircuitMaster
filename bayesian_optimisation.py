from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qutip import Qobj
from general_utils import GateObj, U_from_hamiltonian
from circuit_utils import qCirc, EstimateCircuits
from probability_distributions import ChiProbDist
import numpy as np
import matplotlib.pyplot as plt
from GPyOpt_fork.GPyOpt import GPyOpt
import sys
import sys
from os import path, getcwd, makedirs
import pickle

sys.path.insert(0, path.join(sys.path[0], 'bayesianCircuitMaster'))

savepath = path.join(getcwd(), 'data', 'BO', 'initial_runs', 'test')

if not path.exists(savepath):
    makedirs(savepath)

# define hamiltonian
nqubits = 3
########################################################################################
t = 0.75
########################################################################################
hamiltonian = []
for i in range(nqubits):
    hamiltonian.append(GateObj('X', i, True, 0.75))
    hamiltonian.append(GateObj('Z', i, True, 0.25))
    if i < nqubits - 1:
        hamiltonian.append(GateObj(['Z', 'Z'], [i, i+1], True, 1.0))

ideal_U = U_from_hamiltonian(hamiltonian, nqubits, t)

# define list of GateObjs which will be our estimate ansatz
tsteps = 1
ansatz = []
_vals = []
for k in range(tsteps):
    for i in range(nqubits):
        ansatz.append(GateObj('U3', i, True, (0.0, 0.0, 0.0)))
    for i in range(nqubits-1):
        ansatz.append(GateObj('CNOT', [i, i+1], False))
    for i in range(nqubits):
        ansatz.append(GateObj('U3', i, True, (0.0, 0.0, 0.0)))

_vals = [0.0]*(2*3*nqubits)

prob_dist = ChiProbDist(nqubits=nqubits, U=ideal_U)

load = True
if load:
    IBMQ.load_account()
provider = IBMQ.get_provider(group='samsung', project='imperial')
backend = provider.get_backend('ibmq_qasm_simulator')

qreg = QuantumRegister(3, name='qreg')
init_layout = {0: qreg[0], 1: qreg[1], 2: qreg[2]}

est_circ = EstimateCircuits(prob_dist, ansatz, nqubits=nqubits, length=150,
                            num_shots=8192, backend=backend, init_layout=init_layout)


def F(params):
    return 1 - est_circ.calculate_fidelity(params[0])


NB_INIT = 30
NB_ITER = 30
# DOMAIN_DEFAULT = [(val - np.pi/4, val + np.pi/4) for i, val in enumerate(_vals)]
DOMAIN_DEFAULT = [(0, 2*np.pi) for i, val in enumerate(_vals)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d}
             for i, d in enumerate(DOMAIN_DEFAULT)]

BO_ARGS_DEFAULT = {'domain': DOMAIN_BO, 'initial_design_numdata': NB_INIT,
                   'model_update_interval': 1, 'hp_update_interval': 5,
                   'acquisition_type': 'LCB', 'acquisition_weight': 5,
                   'acquisition_weight_lindec': True, 'optim_num_anchor': 5,
                   'optimize_restarts': 1, 'optim_num_samples': 10000, 'ARD': False}

myBopt = GPyOpt.methods.BayesianOptimization(f=F, **BO_ARGS_DEFAULT)
myBopt.run_optimization(max_iter=NB_ITER)
# myBopt.plot_acquisition()
# myBopt.plot_convergence()

with open(path.join(savepath, 'model_data.pickle'), 'wb') as f:
    pickle.dump(myBopt, f)

(x_seen, y_seen), (x_exp, y_exp) = myBopt.get_best()

with open(path.join(savepath, 'best_params.pickle'), 'wb') as f:
    pickle.dump(((x_seen, y_seen), (x_exp, y_exp)), f)

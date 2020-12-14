from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qutip import Qobj, cnot, toffoli
from general_utils import GateObj, U_from_hamiltonian, get_filename
from circuit_utils import qCirc, EstimateCircuits, FlammiaEstimateCircuits
from probability_distributions_numpy import ChiProbDist, FlammiaProbDist
from CNOT_testing_fidelity import TrueFidelityEst, TrueFOMEstSave
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import path, getcwd, makedirs
import pickle
import scipy.stats as stats

# Simulation params
NB_CIRCS_FOM = 300
NB_SHOTS_FOM = 2048
NB_CIRCS_F = 300
# NB_SHOTS_F = 512
NB_SHOTS_F = np.ceil((NB_CIRCS_FOM*NB_SHOTS_FOM)/NB_CIRCS_F)
qreg = QuantumRegister(3, name='qreg')
init_layout = {0: qreg[0], 1: qreg[1], 4: qreg[2]}

# savepath = path.join(getcwd(), 'data', 'BO', 'generating_all_FOM_dict', 'real_machine', 'toronto', f"{NB_CIRCS_FOM*NB_SHOTS_FOM}_experiments")
# savepath = get_filename(savepath)

savepath = path.join(getcwd(), 'data', 'BO', 'generating_all_FOM_dict', 'fake', 'manhattan', f"{300*2048}_experiments")
savepath = get_filename(savepath)


# ideal_U = cnot(3, 2, 0)
# chi_prob_dist = ChiProbDist(nqubits=3, U=ideal_U)
# flam_prob_dist = FlammiaProbDist(nqubits=3, U=ideal_U)

# reversed_U = cnot(3, 0, 2)
# rev_chi_prob_dist = ChiProbDist(nqubits=3, U=reversed_U)
# rev_flam_prob_dist = FlammiaProbDist(nqubits=3, U=reversed_U)

# print("True overlap", np.abs((1/(2**3))*(ideal_U*reversed_U.dag()).tr())**2)

load = True
if load:
    IBMQ.load_account()
load = False
provider = IBMQ.get_provider(group='samsung', project='imperial')
# provider = IBMQ.get_provider(group='open', project='main')
# backend = provider.get_backend('ibmq_toronto')
backend = Aer.get_backend('qasm_simulator')

# _vals = [0.0*np.pi*np.random.rand() for i in range(18)]

# define ansatz (necessary for the simulation, but won't be changed)
# ansatz = []
# # ansatz.append(GateObj('U3', 0, True, (_vals[0], _vals[1], _vals[2])))
# # ansatz.append(GateObj('U3', 1, True, (_vals[3], _vals[4], _vals[5])))
# # ansatz.append(GateObj('U3', 2, True, (_vals[6], _vals[7], _vals[8])))
# # ansatz.append(GateObj('RZ', 0, True, 0.05))
# # ansatz.append(GateObj('Z', 0, False))
# # ansatz.append(GateObj('Z', 1, False))
# # ansatz.append(GateObj('Z', 1, False))
# ansatz.append(GateObj('CNOT', [0, 1], False))
# ansatz.append(GateObj('CNOT', [1, 2], False))
# ansatz.append(GateObj('CNOT', [0, 1], False))
# ansatz.append(GateObj('CNOT', [1, 2], False))
# ansatz.append(GateObj('U3', 0, True, (_vals[9], _vals[10], _vals[11])))
# ansatz.append(GateObj('U3', 1, True, (_vals[12], _vals[13], _vals[14])))
# ansatz.append(GateObj('U3', 2, True, (_vals[15], _vals[16], _vals[17])))


# define hamiltonian
nqubits = 3
t = 0.84
hamiltonian = []
coeffs = []
for i in range(nqubits)[::-1]:
    hamiltonian.append(GateObj('X', i, True, 0.55))
    hamiltonian.append(GateObj('Z', i, True, 0.35))
for i in range(nqubits)[::-1]:
    if i < nqubits - 1:
        hamiltonian.append(GateObj(['Z', 'Z'], [i, i+1], True, 1.))

ideal_U = U_from_hamiltonian(hamiltonian, nqubits, t)

# define list of GateObjs which will be our estimate ansatz
tsteps = 1
ansatz = []
params = []
for k in range(tsteps):
    for i in range(nqubits):
        ansatz.append(GateObj('RX', i, True, 2*0.55*t/(tsteps)))
        params.append(2*0.55*t/(tsteps))
        ansatz.append(GateObj('RZ', i, True, 2*0.35*t/(tsteps)))
        params.append(2*0.35*t/(tsteps))
    for i in range(nqubits):
        if i < nqubits - 1:
            ansatz.append(GateObj('CNOT', [i, i+1], False))
            ansatz.append(GateObj('RZ', i+1, True, 2*t/(tsteps)))
            params.append(2*t/tsteps)
            ansatz.append(GateObj('CNOT', [i, i+1], False))

# flam_prob_dist = FlammiaProbDist(nqubits=nqubits, U=ideal_U)
rev_chi_prob_dist = ChiProbDist(nqubits=nqubits, U=ideal_U)

# _vals = [1.05]
_vals = []

done = False
while not done:
    try:
        exp_opt_true_est_circ = TrueFOMEstSave(rev_chi_prob_dist, ansatz, nqubits=3,
                                                num_shots=8192, backend=backend, init_layout=init_layout,
                                                params=params, noise_model=None)
        done = True
    except:
        print('not done')
        continue

all_FOM_dict = exp_opt_true_est_circ.counts_dicts

print(exp_opt_true_est_circ.calculate_FOM())

with open(path.join(savepath, "all_FOM_dict.pickle"), 'wb') as f:
    pickle.dump(all_FOM_dict, f)

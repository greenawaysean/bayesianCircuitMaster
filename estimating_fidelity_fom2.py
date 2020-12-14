from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qutip import Qobj, cnot, toffoli
from general_utils import GateObj, U_from_hamiltonian, get_filename
from circuit_utils import qCirc, EstimateCircuits, FlammiaEstimateCircuits
from probability_distributions_numpy import ChiProbDist, FlammiaProbDist
from CNOT_testing_fidelity import TrueFidelityEst
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import path, getcwd, makedirs
import pickle
import scipy.stats as stats

# Simulation params
NB_CIRCS_FOM = 160
NB_SHOTS_FOM = 1024
NB_CIRCS_F = 160
# NB_SHOTS_F = 512
NB_SHOTS_F = np.ceil((NB_CIRCS_FOM*NB_SHOTS_FOM)/NB_CIRCS_F)
qreg = QuantumRegister(3, name='qreg')
init_layout = {2: qreg[0], 3: qreg[1], 4: qreg[2]}

savepath = path.join(getcwd(), 'data', 'BO', 'comparing_FOM_F', 'real_machine', 'manhattan', f"{NB_CIRCS_FOM*NB_SHOTS_FOM}_experiments")
savepath = get_filename(savepath)

ideal_U = cnot(3, 0, 2)
chi_prob_dist = ChiProbDist(nqubits=3, U=ideal_U)
flam_prob_dist = FlammiaProbDist(nqubits=3, U=ideal_U)

reversed_U = cnot(3, 2, 0)
rev_chi_prob_dist = ChiProbDist(nqubits=3, U=reversed_U)
# rev_flam_prob_dist = FlammiaProbDist(nqubits=3, U=reversed_U)

# print("True overlap", np.abs((1/(2**3))*(ideal_U*reversed_U.dag()).tr())**2)

load = True
if load:
    IBMQ.load_account()
load = False
provider = IBMQ.get_provider(group='samsung', project='imperial')
# provider = IBMQ.get_provider(group='open', project='main')
backend = provider.get_backend('ibmq_santiago')
# backend = Aer.get_backend('qasm_simulator')

# _vals = [0.0*np.pi*np.random.rand() for i in range(18)]

# define ansatz (necessary for the simulation, but won't be changed)
ansatz = []
# ansatz.append(GateObj('U3', 0, True, (_vals[0], _vals[1], _vals[2])))
# ansatz.append(GateObj('U3', 1, True, (_vals[3], _vals[4], _vals[5])))
# ansatz.append(GateObj('U3', 2, True, (_vals[6], _vals[7], _vals[8])))
# ansatz.append(GateObj('RZ', 0, True, 1.05))
# ansatz.append(GateObj('Z', 0, False))
# ansatz.append(GateObj('Z', 1, False))
# ansatz.append(GateObj('Z', 1, False))
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
# ansatz.append(GateObj('U3', 0, True, (_vals[9], _vals[10], _vals[11])))
# ansatz.append(GateObj('U3', 1, True, (_vals[12], _vals[13], _vals[14])))
# ansatz.append(GateObj('U3', 2, True, (_vals[15], _vals[16], _vals[17])))

# _vals = [1.05]
_vals = []

fom_est_circ = EstimateCircuits(chi_prob_dist, ansatz, nqubits=3,
                            num_shots=NB_SHOTS_FOM, backend=backend, init_layout=init_layout,
                            noise_model=None)

f_est_circ = FlammiaEstimateCircuits(flam_prob_dist, ansatz, nqubits=3,
                            num_shots=NB_SHOTS_F, backend=backend, init_layout=init_layout, length=NB_CIRCS_F, p_length=8,
                            noise_model=None)

done = False
while not done:
    try:
        exp_opt_true_est_circ = TrueFidelityEst(chi_prob_dist, ansatz, nqubits=3,
                                                num_shots=8192, backend=backend, init_layout=init_layout,
                                                params=_vals, noise_model=None)
        done = True
    except:
        continue

true_F = exp_opt_true_est_circ.calculate_F()
print(true_F)
true_FOM = exp_opt_true_est_circ.calculate_FOM()
print(true_FOM)

with open(path.join(savepath, "F_value.pickle"), 'wb') as f:
    pickle.dump(true_F, f)

with open(path.join(savepath, "FOM_value.pickle"), 'wb') as f:
    pickle.dump(true_FOM, f)

F_diff = []
FOM_diff = []
for i in range(50):
    x = np.real(true_FOM - fom_est_circ.calculate_fidelity(_vals, length=NB_CIRCS_FOM))
    y = np.real(true_F - f_est_circ.calculate_fidelity(_vals))
    print(np.abs(x) < np.abs(y))
    F_diff.append(y)
    FOM_diff.append(x)
    done = False
    while not done:
        try:
            x = f_est_circ.calculate_fidelity(_vals)
            F_diff.append(x)
            done = True
        except:
            continue
    done = False
    while not done:
        try:
            y = fom_est_circ.calculate_fidelity(_vals, length=NB_CIRCS_FOM)
            FOM_diff.append(y)
            done = True
        except:
            continue

with open(path.join(savepath, "raw_Fs.pickle"), 'wb') as f:
    pickle.dump(F_diff, f)

with open(path.join(savepath, "raw_FOMs.pickle"), 'wb') as f:
    pickle.dump(FOM_diff, f)

F_true_diff = [i-true_F for i in F_diff]
FOM_true_diff = [i-true_FOM for i in FOM_diff]

with open(path.join(savepath, "sub_true_Fs.pickle"), 'wb') as f:
    pickle.dump(F_diff, f)

with open(path.join(savepath, "sub_true_FOMs.pickle"), 'wb') as f:
    pickle.dump(FOM_diff, f)

F_diff = [i-np.mean(F_diff) for i in F_diff]
FOM_diff = [i-np.mean(FOM_diff) for i in FOM_diff]

with open(path.join(savepath, "sub_mean_Fs.pickle"), 'wb') as f:
    pickle.dump(F_diff, f)

with open(path.join(savepath, "sub_mean_FOMs.pickle"), 'wb') as f:
    pickle.dump(FOM_diff, f)
#
# F_diff = sorted(F_diff)
# FOM_diff = sorted(FOM_diff)
#
# plt.figure(1)
# weights = np.ones_like(F_diff)/float(len(F_diff))
# a1,b1,_ = plt.hist(F_diff, weights=weights, bins=20, alpha=0.85)
# fit1 = stats.norm.pdf(F_diff, np.mean(F_diff), np.std(F_diff))
# fit1 = fit1*(np.max(a1)/np.max(fit1))
# plt.plot(F_diff, fit1, marker=None)
#
# weights = np.ones_like(FOM_diff)/float(len(FOM_diff))
# a2,b2,_ = plt.hist(FOM_diff, weights=weights, bins=20, alpha=0.85)
# fit2 = stats.norm.pdf(FOM_diff, np.mean(FOM_diff), np.std(FOM_diff))
# fit2 = fit2*(np.max(a2)/np.max(fit2))
# plt.plot(FOM_diff, fit2, marker=None)
#
# plt.xlabel("Error")
# plt.ylabel("Frequency")
#
# plt.show()

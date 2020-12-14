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

savepath = path.join(getcwd(), 'data', 'BO', 'comparing_FOM_F', 'real_machine', 'toronto', f"25600_experiments")


# savepath = path.join(getcwd(), 'data', 'BO', 'comparing_FOM_F', 'fake', 'manhattan', f"614400_experiments_39")

# 75 circs, 128 shots

with open(path.join(savepath, "F_value.pickle"), 'rb') as f:
    true_F = pickle.load(f)

with open(path.join(savepath, "FOM_value.pickle"), 'rb') as f:
    true_FOM = pickle.load(f)

with open(path.join(savepath, "raw_FOMs.pickle"), 'rb') as f:
    raw_FOMs = pickle.load(f)

with open(path.join(savepath, "raw_Fs.pickle"), 'rb') as f:
    raw_Fs = pickle.load(f)

with open(path.join(savepath, "sub_mean_FOMs.pickle"), 'rb') as f:
    sub_mean_FOMs = pickle.load(f)

with open(path.join(savepath, "sub_mean_Fs.pickle"), 'rb') as f:
    sub_mean_Fs = pickle.load(f)

with open(path.join(savepath, "sub_true_FOMs.pickle"), 'rb') as f:
    sub_true_FOMs = pickle.load(f)

with open(path.join(savepath, "sub_true_Fs.pickle"), 'rb') as f:
    sub_true_Fs = pickle.load(f)

print(true_F)
print(raw_Fs)
print('var Fs', np.var(raw_Fs))
print("F mean", true_F - np.mean(raw_Fs))
print(true_FOM)
print(raw_FOMs)
print('var FOMs', np.var(raw_FOMs))

F_true_diff = [i-true_F for i in raw_Fs]
print(np.mean(F_true_diff))
FOM_true_diff = [i-true_FOM for i in raw_FOMs]
print(np.mean(FOM_true_diff))

# plt.scatter([i for i in range(len(FOM_true_diff))], FOM_true_diff, label='FOM')
# plt.scatter([i for i in range(len(F_true_diff))], F_true_diff, label='F')
plt.scatter([i for i in range(len(FOM_true_diff))], raw_FOMs, label='FOM Estimates')
plt.scatter([i for i in range(len(F_true_diff))], raw_Fs, label='F Estimates')
plt.hlines(true_FOM, 0.0, len(FOM_true_diff), color='red', linestyle='--', label='True FOM')
plt.hlines(true_F, 0.0, len(F_true_diff), color='black', linestyle='--', label='True F')
plt.legend()
plt.show()

########################################################################################

#
# raw_Fs = sorted(raw_Fs)
# raw_FOMs = sorted(raw_FOMs)
#
# plt.figure(1)
# weights = np.ones_like(raw_Fs)/float(len(raw_Fs))
# a1,b1,_ = plt.hist(raw_Fs, weights=weights, bins=5, alpha=0.85)
# fit1 = stats.norm.pdf(raw_Fs, np.mean(raw_Fs), np.std(raw_Fs))
# fit1 = fit1*(np.max(a1)/np.max(fit1))
# plt.plot(raw_Fs, fit1, marker=None)
# plt.vlines(true_F, 0.0, 0.5, color='black', linestyle='--')
#
# weights = np.ones_like(raw_FOMs)/float(len(raw_FOMs))
# a2,b2,_ = plt.hist(raw_FOMs, weights=weights, bins=5, alpha=0.85)
# fit2 = stats.norm.pdf(raw_FOMs, np.mean(raw_FOMs), np.std(raw_FOMs))
# fit2 = fit2*(np.max(a2)/np.max(fit2))
# plt.plot(raw_FOMs, fit2, marker=None)
# # plt.vlines(true_FOM, 0.0, 0.5, color='red', linestyle='--')
#
#
# plt.xlabel("Error")
# plt.ylabel("Frequency")
#
# plt.show()

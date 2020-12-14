from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qutip import Qobj, cnot, toffoli
from general_utils import GateObj, U_from_hamiltonian, get_filename
from circuit_utils import qCirc, EstimateCircuits, FlammiaEstimateCircuits
from probability_distributions_numpy import ChiProbDist, FlammiaProbDist
from CNOT_testing_fidelity import TrueFidelityEst, TrueFOMEstSave
from circuit_utils import generate_expectation
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import path, getcwd, makedirs
import pickle
import scipy.stats as stats

def sample_keys(num_circs, prob_dict):
    probs = [prob_dict[key] for key in prob_dict]
    keys = [key for key in prob_dict]
    choices = np.random.choice([i for i in range(len(keys))], num_circs, p=probs, replace=True)
    qutip_settings = [keys[i] for i in choices]
    settings = []
    for sett in qutip_settings:
        setting0 = sett[0][::-1]
        setting1 = sett[1][::-1]
        settings.append((setting0, setting1))

    # settings = [sett for sett in settings if sett[1] != '000']
    # qutip_settings = [sett for sett in qutip_settings if sett[1] != '000']

    return qutip_settings, settings

def subsample_counts(counts_dict, num_shots):
    probs = [counts_dict[key]/8192 for key in counts_dict]
    keys = [key for key in counts_dict]
    choices = np.random.choice([i for i in range(len(keys))], num_shots, p=probs, replace=True)
    # print(choices)
    temp_counts_dict = {i: 0 for i in range(8)}
    for i in choices:
        temp_counts_dict[i] += 1
    # print(temp_counts_dict)
    new_counts_dict = {keys[i]: temp_counts_dict[i] for i in range(len(keys))}
    # print(new_counts_dict)
    return new_counts_dict

def get_expect(counts_dict, num_shots, key):
    _ignore = [2-i for i, k in enumerate(key[1]) if k == '0']
    # _ignore = None
    # counts_dict = subsample_counts(counts_dict, num_shots)
    exp = generate_expectation(counts_dict, _ignore)
    return exp

def generate_fidelity(sett, q_sett, chi_dict, FOM_dicts, num_circs, num_shots, probabilities):
    ideal_chi = [chi_dict[k] for k in q_sett]
    expects = [get_expect(FOM_dicts[k], num_shots, k) for k in sett]
    probs = [probabilities[k] for k in sett]
    add = 0
    # for k in sett:
    #     if k[1] == '000':
    #         # print('dfijn')
    #         add += 1
    fidelity = 0
    for i, chi in enumerate(ideal_chi):
        # if sett[i][1] == '000':
        #     print(chi, expects[i])
        # fidelity += probs[i]*expects[i]/chi
        fidelity += expects[i]/chi

    # print(fidelity/num_circs)
    # fidelity += add
    return np.real(fidelity/len(expects))
    # return np.real(fidelity/np.sum(probs))

########################################################################################

# num_circs = 75

true_F = 0.8326287726006335

# savepath = path.join(getcwd(), 'data', 'BO', 'generating_all_FOM_dict', 'fake', 'manhattan', f"{300*2048}_experiments")

# savepath = path.join(getcwd(), 'data', 'BO', 'generating_all_FOM_dict', 'real_machine', 'toronto', f"{300*2048}_experiments_3")

savepath = path.join(getcwd(), 'data', 'BO', 'generating_all_FOM_dict', 'fake', 'manhattan', "614400_experiments_4")
nqubits = 3
t = 0.84
hamiltonian = []
coeffs = []
for i in range(nqubits)[::]:
    hamiltonian.append(GateObj('X', i, True, 0.55))
    hamiltonian.append(GateObj('Z', i, True, 0.35))
for i in range(nqubits)[::]:
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
chi_prob_dist = ChiProbDist(nqubits=nqubits, U=ideal_U)

with open(path.join(savepath, "all_FOM_dict.pickle"), 'rb') as f:
    FOM_dicts = pickle.load(f)

# q_sett, sett = sample_keys(num_circs, chi_prob_dist.probabilities)
chis = chi_prob_dist.chi_dict

# num_iter = 1000
# res = []
# for i in range(num_iter):
#     q_sett, sett = sample_keys(num_circs, chi_prob_dist.probabilities)
#     res.append(generate_fidelity(sett,q_sett,chis,FOM_dicts,num_circs))
#
# plt.scatter([i for i in range(num_iter)], res)
# plt.hlines(true_F,0,num_iter,color='black',linestyle='--')
# plt.ylim([0,1])
# plt.show()

nb_iter = 10000
# probs = chi_prob_dist.probabilities
# keys = [i for i in chi_prob_dist.probabilities]
# non_zero = len([probs[k] for k in keys if probs[k] != 0])
# print(non_zero)
non_zero = 2187
# circs_range = [10*(i+1) for i in range(np.int(np.floor(non_zero/10)-1))]

circs_range = [16*(i+1) for i in range(56)]
# shots_range = [(i+1) for i in range(50)]

# num_circs = 112
num_shots = 1024

# q_sett, sett = sample_keys(num_circs, chi_prob_dist.probabilities)
# print(generate_fidelity(sett,q_sett,chis,FOM_dicts,num_circs, num_shots,chi_prob_dist.probabilities))

avs = []
stds_upp = []
stds_low = []
mins = []
maxs = []
all_res = {}
for num_circs in circs_range:
# for num_shots in shots_range:
    res = []
    for i in range(nb_iter):
        q_sett, sett = sample_keys(num_circs, chi_prob_dist.probabilities)
        res.append(generate_fidelity(sett,q_sett,chis,FOM_dicts,num_circs, num_shots,chi_prob_dist.probabilities))
    all_res[num_circs] = res
    mean = np.mean(res)
    avs.append(mean)
    std = np.std(res)
    stds_upp.append(mean+std)
    stds_low.append(mean-std)
    mins.append(np.min(res))
    maxs.append(np.max(res))

with open(path.join(savepath, "FOM_all_res_circs.pickle"), 'wb') as f:
    pickle.dump(all_res, f)

with open(path.join(savepath, "FOM_avs_circs.pickle"), 'wb') as f:
    pickle.dump(avs, f)

with open(path.join(savepath, "FOM_stds_upp_circs.pickle"), 'wb') as f:
    pickle.dump(stds_upp, f)

with open(path.join(savepath, "FOM_stds_low_circs.pickle"), 'wb') as f:
    pickle.dump(stds_low, f)

with open(path.join(savepath, "FOM_mins_circs.pickle"), 'wb') as f:
    pickle.dump(mins, f)

with open(path.join(savepath, "FOM_maxs_circs.pickle"), 'wb') as f:
    pickle.dump(maxs, f)

# # circs_range = shots_range
#
plt.plot(circs_range, avs)
# plt.hlines(true_F, np.min(circs_range), np.max(circs_range))
# plt.hlines(np.min(stds_upp), np.min(circs_range), np.max(circs_range))
# plt.hlines(np.max(stds_low), np.min(circs_range), np.max(circs_range))
# plt.vlines(112, 0, 1)
plt.fill_between(circs_range, avs, maxs, alpha=0.3, color='green')
plt.fill_between(circs_range, mins, avs, alpha=0.3, color='green')
plt.fill_between(circs_range, avs, stds_upp, alpha=0.5, color='red')
plt.fill_between(circs_range, stds_low, avs, alpha=0.5, color='red')
plt.ylim([0.,1])
plt.show()

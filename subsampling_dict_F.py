from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qutip import Qobj, cnot, toffoli
from general_utils import GateObj, U_from_hamiltonian, get_filename
from circuit_utils import qCirc, EstimateCircuits, FlammiaEstimateCircuits
from probability_distributions_numpy import ChiProbDist, FlammiaProbDist
from CNOT_testing_fidelity import TrueFidelityEst, TrueFOMEstSave, generate_Bmat
from circuit_utils import generate_expectation
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
from os import path, getcwd, makedirs
import pickle
import scipy.stats as stats
import time

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
    _ignore = [2-i for i, k in enumerate(key[1][::]) if k == '0']
    # _ignore = None
    start = time.time()
    # counts_dict = subsample_counts(counts_dict, num_shots)
    # print('sample time', time.time()-start)
    exp = generate_expectation(counts_dict, _ignore)
    return exp

def generate_coeff_mapping():
    # a = 1
    # _x = -3/(2*np.sqrt(2)+3)
    # # print('x', _x)
    # _y = 1/np.sqrt(3)
    # coeff_mapping = {
    # '0': {'0': a*1/2, '1': a*2*np.sqrt(2)/2*_x+(3/2), '2': -a*_x*(np.sqrt(2)/2), '3': -a*_x*(np.sqrt(2)/2)},
    # '1': {'0': 0, '1': -a*2*_x, '2': a*_x, '3': a*_x},
    # '2': {'0': 0, '1': 0, '2': a*_y, '3': -a*_y},
    # '3': {'0': a*3/2, '1': -a*2*np.sqrt(2)/2*_x-(3/2), '2': a*_x*(np.sqrt(2)/2), '3': a*_x*(np.sqrt(2)/2)}
    # }
    _x = -1/(np.sqrt(2))
    # print('x', _x)
    _y = -3/np.sqrt(6)
    coeff_mapping = {
    '0': {'0': 1/2, '1': 2*np.sqrt(2)/2*_x+(3/2), '2': -_x*(np.sqrt(2)/2), '3': -_x*(np.sqrt(2)/2)},
    '1': {'0': 0, '1': -2*_x, '2': _x, '3': _x},
    '2': {'0': 0, '1': 0, '2': -_y, '3': _y},
    '3': {'0': 3/2, '1': -2*np.sqrt(2)/2*_x-(3/2), '2': _x*(np.sqrt(2)/2), '3': _x*(np.sqrt(2)/2)}
    }
    return coeff_mapping

def generate_map_coeff(w_idx, state_in, coeff_mapping):
    coeff = 1
    for i, idx in enumerate(w_idx):
        coeff *= coeff_mapping[idx][state_in[i]]
        # coeff *= coeff_mapping[state_in[i]][idx]
    return coeff

def generate_input_state_mapping(coeff_mapping):
    all_perms = [''.join(i) for i in itertools.product('0123', repeat=3)]
    mapping = {}
    for w_idx in all_perms:
        for state_in in all_perms:
            coeff = generate_map_coeff(w_idx, state_in, coeff_mapping)
            mapping[(w_idx, state_in)] = coeff
    return mapping


def get_expects_from_w_input(sett, mapping, counts_dicts, num_shots):
    w_in = sett[0][::]
    meas_basis= sett[1][::]
    all_perms = [''.join(i) for i in itertools.product('0123', repeat=3)]
    coeffs_map = {_p: mapping[(w_in[::], _p)] for _p in all_perms if mapping[(w_in[::], _p)] != 0}
    # if w_in[::-1] == '212' or meas_basis == '122' or meas_basis == '221':
    #     print(w_in[::-1], coeffs_map.keys())
    expect = 0
    # print(len(coeffs_map))
    for _p in coeffs_map:
        key = (_p[::], meas_basis[::])
        expect += get_expect(counts_dicts[key],num_shots,key)*coeffs_map[_p[::]]
    return expect

def generate_fidelity(sett, q_sett, chi_dict, counts_dicts, num_circs, num_shots, probabilities, mapping):
    ideal_chi = [chi_dict[k] for k in q_sett]
    # expects = [get_expect(counts_dicts[k], num_shots, k) for k in sett]
    expects =[
    get_expects_from_w_input(k[::], mapping, counts_dicts, num_shots) for k in sett
    ]
    probs = [probabilities[k] for k in q_sett]

    fidelity = 0
    for i, chi in enumerate(ideal_chi):
        # fidelity += probs[i]*expects[i]/chi
        fidelity += expects[i]/(chi)

    return np.real(fidelity/len(expects))
    # print('##############################################')
    # print(fidelity)
    # print(probs)
    # print('##############################################')
    # return np.real(fidelity/np.sum(probs))

def calculate_F(chi_dict, expects):
    perms = [''.join(i) for i in itertools.product('0123', repeat=3)]
    B_dict = {}
    for i, p in enumerate(perms):
        B_dict[p] = i
    d = 2**3
    Bmat = generate_Bmat(3, 3)
    F = 0
    chis = [chi_dict[key] for key in chi_dict]
    chi_keys = [key for key in chi_dict]
    keys = [key[0] for key in chi_dict]
    for i, _key in enumerate(chi_keys):
        chi = chi_dict[_key]
        for j, exp in enumerate(expects):
            _set1 = B_dict[keys[i]]
            _set2 = B_dict[keys[j]]
            F += Bmat[_set1, _set2]*chi*exp

    return F/d**3

# ideal_U = cnot(3, 2, 0)
# prob_dist = FlammiaProbDist(nqubits=3, U=ideal_U)
# # prob_dist = ChiProbDist(nqubits=3, U=ideal_U)
#
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

prob_dist = FlammiaProbDist(nqubits=nqubits, U=ideal_U)
# chi_prob_dist = ChiProbDist(nqubits=nqubits, U=ideal_U)

with open(path.join(savepath, "all_FOM_dict.pickle"), 'rb') as f:
    counts_dicts = pickle.load(f)

coeff_mapping = generate_coeff_mapping()

mapping = generate_input_state_mapping(coeff_mapping)
#
# num_circs = 112
# num_shots = 256
# results = []
# for i in range(20):
#     q_sett, sett = sample_keys(num_circs, prob_dist.probabilities)
#
# # print(generate_fidelity(sett, q_sett, prob_dist.chi_dict, counts_dicts, num_circs, num_shots, prob_dist.probabilities, mapping))
#
#     results.append(generate_fidelity(sett, q_sett, prob_dist.chi_dict, counts_dicts, num_circs, num_shots, prob_dist.probabilities, mapping))
# plt.scatter([i for i in range(len(results))], results)
# plt.hlines(np.mean(results), 0, len(results))
# plt.ylim([min(min(results),0),max(max(results),1)])
# plt.show()
#######################################################################################

nb_iter = 10000
# probs = chi_prob_dist.probabilities
# keys = [i for i in chi_prob_dist.probabilities]
# non_zero = len([probs[k] for k in keys if probs[k] != 0])
# print(non_zero)
non_zero = 2187
# circs_range = [10*(i+1) for i in range(np.int(np.floor(non_zero/10)-1))]

circs_range = [2*(i+1) for i in range(56)]
# shots_range = [(i+1) for i in range(50)]

coeff_mapping = generate_coeff_mapping()

mapping = generate_input_state_mapping(coeff_mapping)

# num_circs = 112
num_shots = 1024

avs = []
stds_upp = []
stds_low = []
mins = []
maxs = []
all_res = {}
for num_circs in circs_range:
    # print(num_circs)
# for num_shots in shots_range:
    res = []
    for i in range(nb_iter):
        q_sett, sett = sample_keys(num_circs, prob_dist.probabilities)
        res.append(generate_fidelity(sett, q_sett, prob_dist.chi_dict, counts_dicts, num_circs, num_shots, prob_dist.probabilities, mapping))
    all_res[num_circs] = res
    mean = np.mean(res)
    avs.append(mean)
    std = np.std(res)
    stds_upp.append(mean+std)
    stds_low.append(mean-std)
    mins.append(np.min(res))
    maxs.append(np.max(res))

# # savepath = path.join(savepath, 'fidelity_data')
# # if not path.exists(savepath):
# #     makedirs(savepath)
# #
# # with open(path.join(savepath, "avs.pickle"), 'wb') as f:
# #     pickle.dump(avs, f)
# #
# # with open(path.join(savepath, "stds_upp.pickle"), 'wb') as f:
# #     pickle.dump(stds_upp, f)
# #
# # with open(path.join(savepath, "stds_low.pickle"), 'wb') as f:
# #     pickle.dump(stds_low, f)
# #
# # with open(path.join(savepath, "mins.pickle"), 'wb') as f:
# #     pickle.dump(mins, f)
# #
# # with open(path.join(savepath, "maxs.pickle"), 'wb') as f:
# #     pickle.dump(maxs, f)
#
# # circs_range = shots_range

with open(path.join(savepath, "F_all_res_circs.pickle"), 'wb') as f:
    pickle.dump(all_res, f)

with open(path.join(savepath, "F_avs_circs.pickle"), 'wb') as f:
    pickle.dump(avs, f)

with open(path.join(savepath, "F_stds_upp_circs.pickle"), 'wb') as f:
    pickle.dump(stds_upp, f)

with open(path.join(savepath, "F_stds_low_circs.pickle"), 'wb') as f:
    pickle.dump(stds_low, f)

with open(path.join(savepath, "F_mins_circs.pickle"), 'wb') as f:
    pickle.dump(mins, f)

with open(path.join(savepath, "F_maxs_circs.pickle"), 'wb') as f:
    pickle.dump(maxs, f)

plt.plot(circs_range, avs)
# plt.hlines(true_F, np.min(circs_range), np.max(circs_range))
plt.hlines(np.min(stds_upp), np.min(circs_range), np.max(circs_range))
plt.hlines(np.max(stds_low), np.min(circs_range), np.max(circs_range))
# plt.vlines(112, 0, 1)
plt.fill_between(circs_range, avs, maxs, alpha=0.3, color='green')
plt.fill_between(circs_range, mins, avs, alpha=0.3, color='green')
plt.fill_between(circs_range, avs, stds_upp, alpha=0.5, color='red')
plt.fill_between(circs_range, stds_low, avs, alpha=0.5, color='red')
plt.ylim([0.,1])
plt.show()


#######################################################################################

#
# coeff_mapping = generate_coeff_mapping()['2']
#
# # print(coeff_mapping['0'])
#
# rho0 = Qobj([[1,0],[0,0]])
# rho1 = Qobj([[1/3,np.sqrt(2)/3],[np.sqrt(2)/3,2/3]])
# # rho2 = Qobj([[1/3,-(1/2) - (1j*np.sqrt(3))/2],[-(1/2) + (1j*np.sqrt(3))/2,2/3]])
# # rho3 = Qobj([[1/3,-(1/2) + (1j*np.sqrt(3))/2],[-(1/2) - (1j*np.sqrt(3))/2,2/3]])
# rho2 = Qobj([[1/3,(-1j*np.sqrt(6)-np.sqrt(2))/6],[(1j*np.sqrt(6)-np.sqrt(2))/6,2/3]])
# rho3 = Qobj([[1/3,(+1j*np.sqrt(6)-np.sqrt(2))/6],[(-1j*np.sqrt(6)-np.sqrt(2))/6,2/3]])
#
# c0 = coeff_mapping['0']
# c1 = coeff_mapping['1']
# c2 = coeff_mapping['2']
# c3 = coeff_mapping['3']
#
# out = c0*rho0 + c1*rho1 + c2*rho2 + c3*rho3
#
# # oui = 3/np.sqrt(2)
#
# print(out)
#
# ######################################################################################
#
# ideal_U = cnot(3, 2, 0)
# # prob_dist = FlammiaProbDist(nqubits=3, U=ideal_U)
# prob_dist = ChiProbDist(nqubits=3, U=ideal_U)
#
# savepath = path.join(getcwd(), 'data', 'BO', 'generating_all_FOM_dict', 'real_machine', 'toronto', f"{300*2048}_experiments_3")
#
# with open(path.join(savepath, "all_FOM_dict.pickle"), 'rb') as f:
#     counts_dicts = pickle.load(f)
#
# expects = [get_expect(counts_dicts[key],8192,key) for key in counts_dicts]
#
# print(calculate_F(prob_dist.chi_dict, expects))

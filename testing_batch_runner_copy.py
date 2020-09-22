from typing import List
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from os import path, getcwd, makedirs
import sys
from qcoptim.optimisers import Method, MethodBO, ParallelRunner
from qcoptim.utilities import (gen_default_argsbo, get_best_from_bo, Batch,
                               gen_random_str, prefix_to_names)
from qcoptim.cost import CostInterface, Cost
from qcoptim.ansatz import AnsatzInterface
import GPyOpt
from general_utils import GateObj, U_from_hamiltonian
from circuit_utils import qCirc, EstimateCircuits
from probability_distributions_numpy import ChiProbDist
from CNOT_testing_fidelity import TrueFidelityEst
from BO_utils import ProcessFidelityAnsatz, ProcessFidelityCost, BayesianOptimiser
from qutip import cnot
from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit


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


sys.path.insert(0, path.join(sys.path[0], 'bayesianCircuitMaster'))

savepath = path.join(getcwd(), 'data', 'BO', 'real_machine', 'batch_results',
                     '3q_CNOT_batch')
savepath = get_filename(savepath)

ideal_U = cnot(3, 2, 0)
prob_dist = ChiProbDist(nqubits=3, U=ideal_U)

reversed_U = cnot(3, 0, 2)
rev_prob_dist = ChiProbDist(nqubits=3, U=reversed_U)

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
init_layout = {0: qreg[0], 1: qreg[1], 2: qreg[2]}

est_circ = EstimateCircuits(prob_dist, ansatz, nqubits=3, num_shots=512,
                            backend=backend, init_layout=init_layout)


N = 4
opt_kwargs = {
    'circs': est_circ,
    'init_params': _vals,
    'nb_init': 50,
    'nb_iter': 30,
    'filename': savepath,
    'init_len': 75,
    'length': 75,
    'incr': np.pi/4,
    'invert': True
}

_optimiser = [BayesianOptimiser(**opt_kwargs).method_bo,
              BayesianOptimiser(**opt_kwargs).method_bo,
              BayesianOptimiser(**opt_kwargs).method_bo,
              BayesianOptimiser(**opt_kwargs).method_bo]

_batch_handler = Batch(est_circ.quant_inst)
wrapped_ansatz = ProcessFidelityAnsatz(est_circ, len(_vals))
_cost_list = [ProcessFidelityCost(est_circ, wrapped_ansatz,
                                  length=opt_kwargs['length']),
              ProcessFidelityCost(est_circ, wrapped_ansatz,
                                  length=opt_kwargs['length']),
              ProcessFidelityCost(est_circ, wrapped_ansatz,
                                  length=opt_kwargs['length']),
              ProcessFidelityCost(est_circ, wrapped_ansatz,
                                  length=opt_kwargs['length'])]

p_runner = ParallelRunner(_cost_list, _optimiser)  # , optimizer_args=opt_kwargs)

# print(p_runner.optim_list[0].next_evaluation_params())
# print(p_runner.optim_list[1].next_evaluation_params())

qc = p_runner.next_evaluation_circuits()
# print(len(qc))
results = _batch_handler.submit_exec_res(p_runner)
# sett_keys = [key for key in p_runner.cost_objs[0].ran_settings.keys()]
# print(sett_keys)
# res_names = [results.to_dict()['results'][i]['header']['name']
#              for i in range(len(results.to_dict()['results']))]
# print(res_names)
# idx = 0
# for i in results.to_dict()['results']:
#     if i['header']['name'].split('circuit')[0][-4:] == sett_keys[0]:
#         print(i['header']['name'])
#         print(i['data']['counts'])
# elif idx <= 3:
#     print(i['header']['name'].split('circuit')[0])
#     print(i['header']['name'].split('circuit')[0][-4:])
#     idx += 1
res = results.get_counts()
p_runner.init_optimisers(res)
for i in range(opt_kwargs['nb_iter']):
    for cst in p_runner.cost_objs:
        cst.reset()
    p_runner.next_evaluation_circuits()
    results = _batch_handler.submit_exec_res(p_runner)
    # print(results)
    p_runner.update(results)

for i in range(N):
    p_runner.optimizer[i].optimiser._compute_results()
    p_runner.optimizer[i].optimiser.plot_acquisition(
        path.join(savepath, f'acquisition_plot_{i+1}.png'))
    p_runner.optimizer[i].optimiser.plot_convergence(
        path.join(savepath, f'convergence_plot_{i+1}.png'))

    plt.close()

    (x_seen, y_seen), (x_exp, y_exp) = get_best_from_bo(p_runner.optimizer[i].optimiser)

    with open(path.join(savepath, f'experiment_data_{i+1}.pickle'), 'wb') as f:
        pickle.dump(((x_seen, y_seen), (x_exp, y_exp)), f)

    reg_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                        num_shots=1024, backend=backend, init_layout=init_layout,
                                        params=_vals, noise_model=None)

    regular_true_F = reg_true_est_circ.calculate_F()
    regular_true_FOM = reg_true_est_circ.calculate_FOM()

    print(regular_true_F)
    print(regular_true_FOM)

    seen_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                         num_shots=1024, backend=backend, init_layout=init_layout,
                                         params=x_seen, noise_model=None)

    seen_true_F = seen_true_est_circ.calculate_F()
    seen_true_FOM = seen_true_est_circ.calculate_FOM()

    print(seen_true_F)
    print(seen_true_FOM)

    exp_opt_true_est_circ = TrueFidelityEst(rev_prob_dist, ansatz, nqubits=3,
                                            num_shots=1024, backend=backend, init_layout=init_layout,
                                            params=x_exp, noise_model=None)

    exp_opt_true_F = exp_opt_true_est_circ.calculate_F()
    exp_opt_true_FOM = exp_opt_true_est_circ.calculate_FOM()

    print(exp_opt_true_F)
    print(exp_opt_true_FOM)

    with open(path.join(savepath, f'true_fidels_{i+1}.pickle'), 'wb') as f:
        pickle.dump((regular_true_F, seen_true_F, exp_opt_true_F), f)

    with open(path.join(savepath, f'true_FOMs_{i+1}.pickle'), 'wb') as f:
        pickle.dump((regular_true_FOM, seen_true_FOM, exp_opt_true_FOM), f)

from typing import List
import numpy as np
import time
import pickle
from os import path, getcwd, makedirs
from qcoptim.optimisers import Method, MethodBO, ParallelRunner
from qcoptim.utilities import (gen_default_argsbo, get_best_from_bo, Batch,
                               gen_random_str, prefix_to_names)
from qcoptim.cost import CostInterface, Cost
from qcoptim.ansatz import AnsatzInterface
import GPyOpt
from circuit_utils import EstimateCircuits


class ProcessFidelityAnsatz(AnsatzInterface):
    """ Instance of AnsatzInterface to go between the qcoptim code and my code
    """

    def __init__(self, est_circs: EstimateCircuits, nparams: int):
        self.est_circs = est_circs
        self.nparams = nparams

    @property
    def depth(self):
        return 1

    @property
    def params(self):
        return [Parameter(str(i)) for i in range(self.nparams)]

    @property
    def circuit(self):
        return self.est_circs.circuits

    @property
    def nb_qubits(self):
        return self.est_circs.nqubits

    @property
    def nb_params(self):
        return self.nparams


class ProcessFidelityCost(Cost):
    """ Wraps the BayesianOptimiser class as a qcoptim Cost class to make it
    compatible with the qcoptim Batch submission class.
    """

    def __init__(self, est_circs: EstimateCircuits, ansatz: AnsatzInterface, length: int):
        self.est_circs = est_circs
        self.name = 'circuit_' + gen_random_str(5)
        self.ansatz = ansatz
        self.length = length
        self.instance = self.est_circs.quant_inst
        self.nb_qubits = ansatz.nb_qubits  # may be redundant
        self.dim = np.power(2, ansatz.nb_qubits)
        self.nb_params = ansatz.nb_params  # maybe redundant
        self.fix_transpile = True
        self.verbose = True
        self._keep_res = False
        self._res = []
        # invert fidelity to minimise
        self._wrap_cost = lambda x: 1-x

    def qk_vars(self):
        """ Returns parameter objects in the circuit"""
        return self.ansatz.params

    def evaluate_cost(self, results_obj, name=None, **kwargs):
        """ generates cost from qiskit result objects"""
        q_list = [i for i in range(self.nqubits)][::-1]
        expects = []
        for i, _c in enumerate(settings):
            _ig = []
            for j, _b in enumerate(_c[1]):
                if _b == '0':
                    _ig.append(j)
            _ignore = [q_list[i] for i in _ig]
            expects.append(generate_expectation(results_obj.get_counts(i), _ignore))
        ideal_chi = [self.est_circs.chi_dict[i] for i in self.qutip_settings]
        fidelity = 0
        for i, _chi in enumerate(ideal_chi):
            fidelity += expects[i] / _chi

        fidelity += self.est_circs.length - len(settings)
        fidelity /= self.est_circs.length

        return self._wrap_cost(np.real(fidelity))

    def bind_params_to_meas(self, params=None, param_names=None):
        """ accepts some parameters and returns a list of circuits to be executed
        """
        self.est_circs.length = self.length
        probs = [self.est_circs.prob_dict[key] for key in self.est_circs.prob_dict]
        keys = [key for key in self.est_circs.prob_dict]
        self.settings, self.qutip_settings = self.est_circs.select_settings(probs, keys)
        chosen_circs = [self.est_circs.circuits[_s] for _s in self.settings]
        exec_circs = [qc.populate_circuits(params) for qc in chosen_circs]
        exec_circs = prefix_to_names(exec_circs, param_names)

        return exec_circs


class BayesianOptimiser:

    def __init__(self, circs: EstimateCircuits, init_params: List[float], nb_init: int, nb_iter: int, filename: str, init_len: int, length: int, incr: float = None, domain_default: List[tuple] = None, invert: bool = True):
        self.circs = circs
        self.init_params = init_params
        self.nb_init = nb_init
        self.nb_iter = nb_iter
        self.init_len = init_len
        self.length = length
        self.incr = incr
        if domain_default is None:
            self.domain_default = [(_p - incr, _p + incr) for _p in init_params]
        else:
            self.domain_default = domain_default
        self.invert = invert
        self.bo_args = self.generate_BO_args()
        self.method_bo = MethodBO(args=self.bo_args)
        self.get_filename(filename)
        self.observed_params = []
        self.observed_FOMs = []
        self.predicted_params = []
        self.predicted_FOMs = []

    def get_filename(self, filename):
        """ Ensures that a unique filename is used with consequential numbering
        """
        if not path.exists(filename):
            makedirs(filename)
            self.filename = filename
        else:
            test = False
            idx = 2
            filename += f'_{idx}'
            while not test:
                if not path.exists(filename):
                    makedirs(filename)
                    self.filename = filename
                    test = True
                else:
                    idx += 1
                    filename = filename[:-(len(str(idx-1))+1)] + f'_{idx}'

    def fidelity_cost(self, params):
        """Wraps calculate fidity method to interact with the Bayesian optimiser.
        """
        f = self.circs.calculate_fidelity(params, self.length)
        if self.invert:
            return 1 - f
        else:
            return f

    def generate_BO_args(self):
        bo_args_default = gen_default_argsbo(
            f=lambda x: None, domain=self.domain_default, nb_init=self.nb_init)
        bo_args_default.update({'nb_iter': self.nb_iter})

        return bo_args_default

    def init_obs(self):
        init_params = self.method_bo.next_evaluation_params()
        if self.init_len*self.nb_init <= 900:
            init_fidels = self.circs.calculate_bulk_fidelity(init_params, self.init_len)
        else:
            num_runs = np.int((self.init_len*self.nb_init)/900)
            init_fidels = []
            for i in range(num_runs):
                idx = np.int(len(init_params)/num_runs)
                _p = init_params[idx*i:idx*(i+1)]
                _fid = self.circs.calculate_bulk_fidelity(_p, self.init_len)
                init_fidels += _fid
        if self.invert:
            init_fidels = [1-x for x in init_fidels]
        return init_params, init_fidels

    def run_BO(self):
        init_time = time.time()
        initial_x, initial_y = self.init_obs()
        # print(initial_x)
        initial_y = np.array(initial_y).reshape(len(initial_x), 1)
        self.method_bo.update(initial_x, initial_y)
        self.method_bo.optimiser._compute_results()
        (x_obs, y_obs), (x_pred, y_pred) = get_best_from_bo(self.method_bo.optimiser)
        self.observed_params.append(x_obs)
        self.observed_FOMs.append(y_obs)
        self.predicted_params.append(x_pred)
        self.predicted_FOMs.append(y_pred)

        self.save_results()

        for _step in range(self.nb_iter):
            # Save data every hour
            if np.abs(time.time() - init_time) > 3600:
                self.save_results()
                init_time = time.time()
            x = self.method_bo.next_evaluation_params()
            y = self.fidelity_cost(x[0])
            self.method_bo.update(x, y)
            (x_obs, y_obs), (x_pred, y_pred) = get_best_from_bo(self.method_bo.optimiser)
            self.observed_params.append(x_obs)
            self.observed_FOMs.append(y_obs)
            self.predicted_params.append(x_pred)
            self.predicted_FOMs.append(y_pred)

        self.save_results(final=True)

    def save_results(self, final=False):
        if final:
            (x_obs, y_obs), (x_pred, y_pred) = get_best_from_bo(self.method_bo.optimiser)
            pathends = ['best_obs_params.pickle',
                        'best_obs_FOM.pickle',
                        'best_predicted_params.pickle',
                        'best_predicted_FOM.pickle']

            list_contents = [x_obs, y_obs, x_pred, y_pred]
            for i, _name in enumerate(pathends):
                with open(path.join(self.filename, _name), 'wb') as f:
                    pickle.dump(list_contents[i], f)
            self.method_bo.optimiser._compute_results()
            self.method_bo.optimiser.plot_acquisition(path.join(self.filename,
                                                                'acquisition_plot.png'))
            self.method_bo.optimiser.plot_convergence(path.join(self.filename,
                                                                'convergence_plot.png'))
        else:
            pathends = ['observed_parameters.pickle',
                        'observed_FOMs.pickle',
                        'predicted_parameters.pickle',
                        'predicted_FOMs.pickle']

            list_contents = [self.observed_params,
                             self.observed_FOMs,
                             self.predicted_params,
                             self.predicted_FOMs]

            for i, _name in enumerate(pathends):
                with open(path.join(self.filename, _name), 'wb') as f:
                    pickle.dump(list_contents[i], f)

            # self.method_bo.optimiser.plot_acquisition(path.join(self.filename,
            #                                                     'acquisition_plot.png'))
            # self.method_bo.optimiser.plot_convergence(path.join(self.filename,
            #                                                     'convergence_plot.png'))

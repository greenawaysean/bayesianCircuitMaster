from typing import List
import numpy as np
import time
import pickle
from os import path, getcwd, makedirs
from qcoptim.optimisers import Method, MethodBO
from qcoptim.utilities import gen_default_argsbo, get_best_from_bo
from GPyOpt_fork.GPyOpt import GPyOpt
from circuit_utils import EstimateCircuits


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

            self.method_bo.optimiser.plot_acquisition(path.join(self.filename,
                                                                'acquisition_plot.png'))
            self.method_bo.optimiser.plot_convergence(path.join(self.filename,
                                                                'convergence_plot.png'))

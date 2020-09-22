from typing import List
import numpy as np
from os import path, getcwd, makedirs
import sys
from general_utils import GateObj, U_from_hamiltonian, get_filename
from circuit_utils import qCirc, EstimateCircuits
from probability_distributions_numpy import ChiProbDist


sys.path.insert(0, path.join(sys.path[0], 'bayesianCircuitMaster'))

savepath = path.join(getcwd(), 'data', 'BO', 'real_machine', 'batch_results',
                     '3q_CNOT_batch')
savepath = get_filename(savepath)

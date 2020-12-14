import numpy as np
from circuit_utils import FlammiaEstimateCircuits, EstimateCircuits
from general_utils import GateObj, U_from_hamiltonian
from probability_distributions_numpy import ChiProbDist, FlammiaProbDist
from qiskit import Aer, IBMQ, ClassicalRegister, QuantumRegister, execute, QuantumCircuit
import pickle
from os import path, getcwd, makedirs

nb_iter = 50

f_shots = 144
f_circs = 896
fom_shots = 384
fom_circs = 336

savepath = path.join(getcwd(), 'data', 'BO', "generating_statistics", "ising_target", f"{nb_iter}_iters_{f_shots*f_circs}_experiments")
if not path.exists(savepath):
    makedirs(savepath)

nqubits = 3
t = 0.75
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

flam_prob_dist = FlammiaProbDist(nqubits=nqubits, U=ideal_U)
chi_prob_dist = ChiProbDist(nqubits=nqubits, U=ideal_U)


load = True
if load:
    IBMQ.load_account()
provider = IBMQ.get_provider(group='samsung', project='imperial')
# backend = Aer.get_backend('qasm_simulator')
backend = provider.get_backend('ibmq_toronto')


qreg = QuantumRegister(nqubits, name='qreg')

flam_est_circ = FlammiaEstimateCircuits(flam_prob_dist, ansatz, nqubits=nqubits, length=f_circs, p_length=2**nqubits, num_shots=f_shots, backend=backend, init_layout = {0:qreg[0], 1:qreg[1], 2:qreg[2]})

fom_est_circ = EstimateCircuits(chi_prob_dist, ansatz, nqubits=nqubits, num_shots=fom_shots, backend=backend, init_layout = {0:qreg[0], 1:qreg[1], 2:qreg[2]})

fidels = []
foms = []
for i in range(nb_iter):
    fidels.append(flam_est_circ.calculate_fidelity(params))
    foms.append(fom_est_circ.calculate_fidelity(params, length=fom_circs))

    with open(path.join(savepath, "fidels.pickle"), 'wb') as f:
        pickle.dump(fidels, f)

    with open(path.join(savepath, "foms.pickle"), 'wb') as f:
        pickle.dump(fidels, f)

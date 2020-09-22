import numpy as np
import pickle
from os import path, getcwd, makedirs
from qutip import sigmax, sigmay, sigmaz, qeye, tensor, basis, cnot
import matplotlib.pyplot as plt

nqubits = 4
tsteps = 4
hx = 0.5
hz = 0.0
J = 1
trange = [0.01*i for i in range(450)]
psi_0 = tensor([basis(2,0)]*nqubits)
rho_0 = psi_0*psi_0.dag()

meas_qubits = [2]
_obs = []
for i in range(nqubits):
    if i in meas_qubits:
        _obs.append(sigmaz())
    else:
        _obs.append(qeye(2))
obs = tensor(_obs)

H = None

for i in range(nqubits):
    if H is None:
        H = tensor([qeye(2)]*i + [hx*sigmax()] + [qeye(2)]*(nqubits-i-1))
        H += tensor([qeye(2)]*i + [hz*sigmaz()] + [qeye(2)]*(nqubits-i-1))
    else:
        H += tensor([qeye(2)]*i + [hx*sigmax()] + [qeye(2)]*(nqubits-i-1))
        H += tensor([qeye(2)]*i + [hz*sigmaz()] + [qeye(2)]*(nqubits-i-1))

for i in range(nqubits-1):
    H += tensor([qeye(2)]*i + [J*tensor(sigmaz(),sigmaz())] + [qeye(2)]*(nqubits-i-2))

# ED
ED_results = []
for t in trange:
    U = (-1j*t*H).expm()
    rho_t = U*rho_0*U.dag()
    e_z = (obs*rho_t).tr()
    ED_results.append(e_z)

# Trotter
trott_results = []
for t in trange:
    V = None
    for _ in range(tsteps):
        for i in range(nqubits):
            if V is None:
                _h = tensor([qeye(2)]*i + [(hx/2)*sigmax()] + [qeye(2)]*(nqubits-i-1))
                V = (-1j*t*_h/tsteps).expm()
                _h = tensor([qeye(2)]*i + [(hz/2)*sigmaz()] + [qeye(2)]*(nqubits-i-1))
                V *= (-1j*t*_h/tsteps).expm()
            else:
                _h = tensor([qeye(2)]*i + [(hx/2)*sigmax()] + [qeye(2)]*(nqubits-i-1))
                V *= (-1j*t*_h/tsteps).expm()
                _h = tensor([qeye(2)]*i + [(hz/2)*sigmaz()] + [qeye(2)]*(nqubits-i-1))
                V *= (-1j*t*_h/tsteps).expm()

        for i in range(nqubits-1):
            _h = tensor([qeye(2)]*i + [J*cnot()*tensor(qeye(2),sigmaz())*cnot()] + [qeye(2)]*(nqubits-i-2))
            V *= (-1j*t*_h/tsteps).expm()

        for i in range(nqubits):
            if V is None:
                _h = tensor([qeye(2)]*i + [(hx/2)*sigmax()] + [qeye(2)]*(nqubits-i-1))
                V = (-1j*t*_h/tsteps).expm()
                _h = tensor([qeye(2)]*i + [(hz/2)*sigmaz()] + [qeye(2)]*(nqubits-i-1))
                V *= (-1j*t*_h/tsteps).expm()
            else:
                _h = tensor([qeye(2)]*i + [(hx/2)*sigmax()] + [qeye(2)]*(nqubits-i-1))
                V *= (-1j*t*_h/tsteps).expm()
                _h = tensor([qeye(2)]*i + [(hz/2)*sigmaz()] + [qeye(2)]*(nqubits-i-1))
                V *= (-1j*t*_h/tsteps).expm()

    rho_t = V*rho_0*V.dag()
    e_z = (obs*rho_t).tr()
    trott_results.append(e_z)

dirname = path.join('C:\\', 'Users', 'seang', 'OneDrive - Imperial College London', 'Documents', 'Project', 'PythonModules', 'bayesianCircuitMaster', 'data', 'BO', 'real_machine', 'ising_optimisation')

with open(path.join(dirname, 'ED_results_hx05.pickle'), 'wb') as f:
    pickle.dump((trange, ED_results, trott_results), f)

plt.plot(trange, ED_results)
plt.plot(trange, trott_results)
plt.show()

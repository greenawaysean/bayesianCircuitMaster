import numpy as np
import itertools
import copy
from qutip import basis, qeye, sigmax, sigmay, sigmaz, tensor, gate_expand_1toN, gate_expand_2toN, Qobj
import matplotlib.pyplot as plt

"""Implementation of the process fidelity estimation proposed in
10.1103/PhysRevLett.106.230501
"""


def generate_prob_dict(nqubits, U):
    """The probability of choosing a given state and a given measurment setting is given
    by the expectation value of the state w.r.t that observable.
    """
    d = 2**nqubits
    input_states = generate_input_states(nqubits)
    observables = generate_observables(nqubits)

    _state_str = [''.join(i)
                  for i in itertools.product('0123', repeat=nqubits)]
    _obs_str = [''.join(i) for i in itertools.product('0123', repeat=nqubits)]

    probabilities = {}
    chi_dict = {}
    for _state_idx in _state_str:
        for _obs_idx in _obs_str:
            _state = input_states[_state_idx]
            _obs = observables[_obs_idx]

            chi = (U * _state * U.dag() * _obs / np.sqrt(d)).tr()

            chi_dict[(_state_idx, _obs_idx)] = chi
            probabilities[(_state_idx, _obs_idx)] = np.abs(chi**2 / d**2)

    return probabilities, chi_dict


def generate_input_states(nqubits):
    input_state_str = [''.join(i)
                       for i in itertools.product('0123', repeat=nqubits)]

    init_state = tensor([basis(2, 0)] * nqubits)

    input_states = {}
    for _state in input_state_str:
        _ops = []
        for i in _state:
            if i == '0':
                _ops.append(qeye(2))
            if i == '1':
                _ops.append(sigmax())
            if i == '2':
                _ops.append(sigmay())
            if i == '3':
                _ops.append(sigmaz())
        _op = tensor(_ops)
        _copy = copy.deepcopy(init_state)
        state = _op*_copy
        input_states[_state] = state*state.dag()

    return input_states


def generate_observables(nqubits):
    input_str = [''.join(i) for i in itertools.product('0123', repeat=nqubits)]

    observables = {}
    for _state in input_str:
        _ops = []
        for i in _state:
            if i == '0':
                _ops.append(qeye(2))
            if i == '1':
                _ops.append(sigmax())
            if i == '2':
                _ops.append(sigmay())
            if i == '3':
                _ops.append(sigmaz())
        _op = tensor(_ops)
        observables[_state] = _op

    return observables


def get_length(resolution, failure_rate):
    return np.int(np.ceil(1 / (resolution**2 * failure_rate)))


def select_settings(length, probs, keys):
    choices = np.random.choice([i for i in range(len(keys))], length, p=[
                               np.real(i) for i in probs])
    settings = [keys[i] for i in choices]
    return settings


def get_approx_chi(input_state, observable, approx_V, nqubits, qiskit):
    d = 2**nqubits
    if qiskit:
        qreg = QuantumRegister(nqubits)
        creg = ClassicalRegiter(nqubits)
        qc = QuantumCircuit(qreg, creg)
        pass
    else:
        init_state = tensor([basis(2, 0)] * nqubits)
        _ops = []
        for i in input_state:
            if i == '0':
                _ops.append(qeye(2))
            if i == '1':
                _ops.append(sigmax())
            if i == '2':
                _ops.append(sigmay())
            if i == '3':
                _ops.append(sigmaz())
        _op = tensor(_ops)
        state = _op*init_state
        state = state*state.dag()

        _ops = []
        for i in observable:
            if i == '0':
                _ops.append(qeye(2))
            if i == '1':
                _ops.append(sigmax())
            if i == '2':
                _ops.append(sigmay())
            if i == '3':
                _ops.append(sigmaz())
        obs = tensor(_ops)

        chi = (approx_V * state * approx_V.dag() * obs / np.sqrt(d)).tr()

    return chi


def estimate_fidelity(settings, chi_dict, length, approx_V, nqubits):
    Y = 0
    for setting in settings:
        chi = chi_dict[setting]
        approx_chi = get_approx_chi(
            setting[0], setting[1], approx_V, nqubits, qiskit=False)
        Y += approx_chi/chi
    return np.abs(Y / length)


if __name__ == '__main__':

    t = 100
    nqubits = 3

    hamiltonian = None
    for i in range(nqubits):
        if hamiltonian is None:
            hamiltonian = 0.5*gate_expand_1toN(sigmax(), nqubits, i)
        else:
            hamiltonian += 0.5*gate_expand_1toN(sigmax(), nqubits, i)
        if i < nqubits-2:
            hamiltonian += gate_expand_2toN(tensor(sigmaz(), sigmaz()), nqubits, i, i+1)

    U = (-1j*t*hamiltonian).expm()

    tsteps = 1
    V = None
    for k in range(tsteps):
        for j in range(nqubits):
            if V is None:
                V = ((-1j*0.5*t/tsteps)*gate_expand_1toN(sigmax(), nqubits, j)).expm()
            else:
                V *= ((-1j*0.5*t/tsteps)*gate_expand_1toN(sigmax(), nqubits, j)).expm()
            if j < nqubits-2:
                V *= ((-1j*t/tsteps)*gate_expand_2toN(tensor(sigmaz(), sigmaz()), nqubits, j, j+1)).expm()

    true_fidel = np.abs((1/2**nqubits)*(U*V.dag()).tr())**2

    prob_dict, chi_dict = generate_prob_dict(nqubits, U)

    probs = [prob_dict[key] for key in prob_dict]
    keys = [key for key in prob_dict]

    # length = get_length(0.1, 0.05)
    length = 5000
    print(length)

    settings = select_settings(length, probs, keys)

    print("true fidelity", true_fidel)
    print("fidelity estimate", estimate_fidelity(settings, chi_dict, length, V, nqubits))

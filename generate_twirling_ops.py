from qutip import tensor, sigmax, sigmay, sigmaz, cnot, qeye, csign


def get_cnot_twirls():
    _ops = [0, 1, 2, 3]

    # gate = csign()
    gate = cnot()

    pauli_tensors = {}

    for i in _ops:
        for j in _ops:
            if i == 0:
                a = qeye(2)
            elif i == 1:
                a = sigmax()
            elif i == 2:
                a = sigmay()
            elif i == 3:
                a = sigmaz()

            if j == 0:
                b = qeye(2)
            elif j == 1:
                b = sigmax()
            elif j == 2:
                b = sigmay()
            elif j == 3:
                b = sigmaz()

            pauli_tensors[(i, j)] = tensor(a, b)

    twirl_ops = {}
    for i in _ops:
        for j in _ops:
            if i == 0:
                a = qeye(2)
            elif i == 1:
                a = sigmax()
            elif i == 2:
                a = sigmay()
            elif i == 3:
                a = sigmaz()

            if j == 0:
                b = qeye(2)
            elif j == 1:
                b = sigmax()
            elif j == 2:
                b = sigmay()
            elif j == 3:
                b = sigmaz()

            _op = gate*tensor(a, b)*gate.dag()
            for k in _ops:
                for m in _ops:
                    if pauli_tensors[(k, m)] == _op or pauli_tensors[(k, m)] == - _op:
                        twirl_ops[(i, j)] = [k, m]
            if (i, j) not in twirl_ops:
                print("error", (i, j))
    return twirl_ops


print(get_cnot_twirls())

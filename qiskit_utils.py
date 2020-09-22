from qiskit import IBMQ, QuantumRegister
from typing import List
import numpy as np

def get_list(off_limits: List, item_in: List, connect_map: List):
    _x = item_in[-1]
    appends = [j for j in connect_map[_x] if j > _x and j not in off_limits]
    list_out = []
    for _k in appends:
        _l = item_in[::]
        _l.append(_k)
        list_out.append(_l)
    return list_out

def calculate_errors(q_list: List, cx_errors: dict, read_errors: dict):
    _count = 0
    for i in range(len(q_list)):
        _count += read_errors[q_list[i]]
        if i < len(q_list) - 1:
            _count += cx_errors[(q_list[i], q_list[i+1])]
    return _count

def layout_to_dict(layout, qreg):
    initial_layout = {}
    for i, _q in enumerate(layout):
        initial_layout[_q] = qreg[i]
    return initial_layout

def get_layout(backend: str, n_choice: int, qreg):
    provider = IBMQ.get_provider(group='samsung', project='imperial')
    backend = provider.backends(backend)[0]
    num_qubits = backend.configuration().num_qubits
    properties = backend.properties()

    q_pairs = []
    cx_errors = {}
    # get cnot errors
    for i in range(num_qubits-1):
        for j in range(i, num_qubits):
            try:
                _err = properties.gate_error('cx', (i, j))
                cx_errors[(i,j)] = _err
                q_pairs.append((i,j))
            except:
                continue

    read_errors = {}
    # get_readout errors
    for i in range(num_qubits):
        read_errors[i] = properties.readout_error(i)

    connect_map = {}
    for i in range(num_qubits):
        _in = []
        for k in q_pairs:
            if i in k:
                for j in k:
                    if j != i:
                        _in.append(j)
        connect_map[i] = _in

    list_of_list_of_lists = []
    for i in range(len(connect_map)):
        list_of_lists = []
        _outlist = get_list([], [i], connect_map)
        if len(_outlist) == 0:
            continue
        # print(_outlist)
        list_of_lists += _outlist
        ignore_list = list(dict.fromkeys([item for sublist in _outlist for item in sublist]))
        while len(list_of_lists[0]) < n_choice:
            _running_list = []
            for _list in list_of_lists:
                new_list = get_list(ignore_list, _list, connect_map)
                _running_list += new_list
            # ignore_list = list(dict.fromkeys([item for sublist in _outlist for item in sublist]))
            list_of_lists = _running_list
            if len(list_of_lists) == 0:
                break
        if len(list_of_lists) == 0:
            continue
        list_of_list_of_lists += list_of_lists

    errors = [calculate_errors(_l, cx_errors, read_errors) for _l in list_of_list_of_lists]

    layout = list_of_list_of_lists[np.argmin(errors)]

    initial_layout = layout_to_dict(layout, qreg)

    return initial_layout


if __name__ == '__main__':
    IBMQ.load_account()
    backend = 'ibmq_toronto'
    nqubits = 5
    qreg = QuantumRegister(nqubits)

    init_layout = get_layout(backend, nqubits, qreg)

    print(init_layout)

from .constants import *
from .multiplexers import *


def one_left_shift(circuit, x_list, parent_group=None):
    this_group = circuit.add_group("ONE_LEFT_SHIFT")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    result = []
    result.append(constant_zero(circuit, x_list[0], parent_group=this_group))
    for x in x_list[:-1]:
        result.append(x)
    return result


def one_right_shift(circuit, x_list, parent_group=None):
    this_group = circuit.add_group("ONE_RIGHT_SHIFT")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    result = []
    for x in x_list[1:]:
        result.append(x)
    result.append(
        constant_zero(circuit, x_list[len(x_list) - 1], parent_group=this_group)
    )
    return result


def n_left_shift(circuit, x_list, amount, parent_group=None):
    this_group = circuit.add_group("N_LEFT_SHIFT")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    n = len(x_list) if len(x_list) < len(amount) else len(amount)
    current = x_list
    for i in range(n):
        shift_amount = 2**i
        shifted = []
        if shift_amount <= n:
            for k in range(shift_amount):
                shifted.append(
                    constant_zero(circuit, current[0], parent_group=this_group)
                )
            for j in range(n - shift_amount):
                shifted.append(current[j])
        else:
            for m in range(n):
                shifted.append(
                    constant_zero(circuit, current[0], parent_group=this_group)
                )
        assert len(shifted) == len(current)
        next_current = []
        for l in range(n):
            next_current.append(
                mux2(
                    circuit, amount[i], shifted[l], current[l], parent_group=this_group
                )
            )
        current = next_current
    return current


def n_right_shift(circuit, x_list, amount, parent_group=None):
    this_group = circuit.add_group("N_RIGHT_SHIFT")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    assert len(x_list) == len(
        amount
    ), "x_list and amount must have the same number of bits"
    n = len(x_list)
    current = x_list
    for i in range(n):
        shift_amount = 2**i
        shifted = []
        if shift_amount <= n:
            for j in range(shift_amount, n):
                shifted.append(current[j])
            for k in range(shift_amount):
                shifted.append(
                    constant_zero(circuit, current[0], parent_group=this_group)
                )
        else:
            for m in range(n):
                shifted.append(
                    constant_zero(circuit, current[0], parent_group=this_group)
                )
        assert len(shifted) == len(current)
        next_current = []
        for l in range(n):
            next_current.append(
                mux2(
                    circuit, amount[i], shifted[l], current[l], parent_group=this_group
                )
            )
        current = next_current
    return current

from ..standard.constants import *
from ..standard.multipliers import *
from ..standard.modular import *
from ..standard.multiplexers import *


def mult_and_mod(circuit, x, y, m, parent_group=None):
    n = len(x)
    assert n == len(y) and n == len(m), "All input must have the same bit length"

    this_group = circuit.add_group("MULT_AND_MOD")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    mult = wallace_tree_multiplier(circuit, x, y, parent_group=this_group)
    mult = mult[:n]
    return modulo_circuit(circuit, mult, m, parent_group=this_group)


def montgomery_ladder(circuit, base, exponent, modulus, parent_group=None):
    this_group = circuit.add_group("MONTGOMERY_LADDER")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    n = len(base)
    assert n == len(exponent) and n == len(
        modulus
    ), "All input must have the same bit length"

    zero = constant_zero(circuit, base[0], parent_group=this_group)
    one = constant_one(circuit, base[0], parent_group=this_group)

    r_0 = []
    r_1 = base

    r_0.append(one)
    for i in range(n - 1):
        r_0.append(zero)

    for bit in reversed(exponent):
        # r_0 * r_1 % modulus
        r_0_r_1 = mult_and_mod(circuit, r_0, r_1, modulus)
        r_0_r_0 = mult_and_mod(circuit, r_0, r_0, modulus)
        r_1_r_1 = mult_and_mod(circuit, r_1, r_1, modulus)

        new_r_0 = []
        new_r_1 = []
        for i in range(n):
            new_r_0.append(
                mux2(circuit, bit, r_0_r_1[i], r_0_r_0[i], parent_group=this_group)
            )
            new_r_1.append(
                mux2(circuit, bit, r_1_r_1[i], r_0_r_1[i], parent_group=this_group)
            )
        r_0 = new_r_0
        r_1 = new_r_1

    return r_0

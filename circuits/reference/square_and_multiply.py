from ..constants import *
from ..modular import *
from ..multipliers import *


def square_and_multiply(circuit, base, exponent, modulus, parent_group=None):
    this_group = circuit.add_group("MODULAR_EXPONENTIATION")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    n = len(base)
    assert n == len(exponent) and n == len(
        modulus
    ), "All input must have the same bit length"

    zero = constant_zero(circuit, base[0], parent_group=this_group)
    one = constant_one(circuit, base[0], parent_group=this_group)

    result = [zero] * n
    result[0] = one
    base_mod = modulo_circuit(circuit, base, modulus, parent_group=this_group)
    for i in range(n):
        bit_pos = n - 1 - i
        current_bit = exponent[bit_pos]
        squared = wallace_tree_multiplier(
            circuit, result, result, parent_group=this_group
        )
        squared = squared[: len(base)]
        squared_mod = modulo_circuit(circuit, squared, modulus, parent_group=this_group)
        with_multiply = wallace_tree_multiplier(
            circuit, squared_mod, base_mod, parent_group=this_group
        )
        with_multiply = with_multiply[: len(base)]
        multiply_mod = modulo_circuit(
            circuit, with_multiply, modulus, parent_group=this_group
        )
        new_result = [None] * n
        for j in range(n):
            # not_bit = circuit.add_node(
            #    "not",
            #    f"NOT_BIT_{bit_pos}_{j}",
            #    inputs=[current_bit],
            #    group_id=this_group_id,
            # ).ports[1]
            # and1 = circuit.add_node(
            #    "and",
            #    f"AND_MULT_{bit_pos}_{j}",
            #    inputs=[current_bit, multiply_mod[j]],
            #    group_id=this_group_id,
            # ).ports[2]
            # and2 = circuit.add_node(
            #    "and",
            #    f"AND_SQR_{bit_pos}_{j}",
            #    inputs=[not_bit, squared_mod[j]],
            #    group_id=this_group_id,
            # ).ports[2]
            # new_result[j] = circuit.add_node(
            #    "or",
            #    f"OR_RESULT_{bit_pos}_{j}",
            #    inputs=[and1, and2],
            #    group_id=this_group_id,
            # ).ports[2]
            not_bit = not_gate(circuit, current_bit, parent_group=this_group)
            and1 = and_gate(
                circuit, [current_bit, multiply_mod[j]], parent_group=this_group
            )
            and2 = and_gate(circuit, [not_bit, squared_mod[j]], parent_group=this_group)
            new_result[j] = or_gate(circuit, [and1, and2], parent_group=this_group)
        result = new_result
    return result

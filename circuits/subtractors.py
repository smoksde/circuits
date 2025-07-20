from typing import List

from .constants import *
from .adders import *
from .utils import *


def subtract(circuit, a_bits, b_bits, parent_group=None) -> List[Port]:
    # a - b
    sub_group = circuit.add_group("SUBTRACT")
    sub_group.set_parent(parent_group)
    zero = constant_zero(circuit, a_bits[0], parent_group=sub_group)
    b_complement = two_complement(circuit, b_bits, parent_group=sub_group)
    result, carry = carry_look_ahead_adder(
        circuit, a_bits, b_complement, zero, parent_group=sub_group
    )
    # result, carry = ripple_carry_adder(
    #    circuit, a_bits, b_complement, zero, parent_group=sub_group
    # )
    return result


def conditional_subtract(circuit, x_bits, m_bits, select, parent_group=None):
    cs_group = circuit.add_group("CONDITIONAL_SUBTRACT")
    cs_group.set_parent(parent_group)
    n = len(x_bits)
    assert len(m_bits) == n, "Both inputs must have the same bit length"
    difference = subtract(circuit, x_bits, m_bits, parent_group=cs_group)
    result = [None] * n
    for i in range(n):
        not_select = circuit.add_node(
            "not", f"NOT_SEL_{i}", inputs=[select], group_id=cs_group.id
        ).ports[1]
        and1 = circuit.add_node(
            "and", f"AND_DIFF_{i}", inputs=[select, difference[i]], group_id=cs_group.id
        ).ports[2]
        and2 = circuit.add_node(
            "and", f"AND_X_{i}", inputs=[not_select, x_bits[i]], group_id=cs_group.id
        ).ports[2]
        result[i] = circuit.add_node(
            "or", f"OR_RES_{i}", inputs=[and1, and2], group_id=cs_group.id
        ).ports[2]
    return result

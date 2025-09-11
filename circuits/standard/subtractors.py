from typing import List

from .constants import *
from .adders import *
from ..circuit_utils import *


def subtract(circuit, a_bits, b_bits, parent_group=None) -> List[Port]:
    # a - b
    this_group = circuit.add_group("SUBTRACT")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    zero = constant_zero(circuit, a_bits[0], parent_group=this_group)
    b_complement = two_complement(circuit, b_bits, parent_group=this_group)
    result, carry = carry_look_ahead_adder(
        circuit, a_bits, b_complement, zero, parent_group=this_group
    )
    return result


def conditional_subtract(circuit, x_bits, m_bits, select, parent_group=None):
    this_group = circuit.add_group("CONDITIONAL_SUBTRACT")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    n = len(x_bits)
    assert len(m_bits) == n, "Both inputs must have the same bit length"
    difference = subtract(circuit, x_bits, m_bits, parent_group=this_group)
    result = [None] * n
    for i in range(n):
        not_select = not_gate(circuit, select, parent_group=this_group)
        and1 = and_gate(circuit, [select, difference[i]], parent_group=this_group)
        and2 = and_gate(circuit, [not_select, x_bits[i]], parent_group=this_group)
        result[i] = or_gate(circuit, [and1, and2], parent_group=this_group)
    return result

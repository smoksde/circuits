from .comparators import *
from .subtractors import *
from .shifters import *


def slow_modulo_circuit(circuit, x_bits, m_bits, parent_group=None):
    this_group = circuit.add_group("SLOW_MODULO_CIRCUIT")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    n = len(x_bits)
    m_len = len(m_bits)
    assert m_len <= n, "Modulus must not be wider than dividend"

    zero = constant_zero(circuit, x_bits[0], parent_group=this_group)

    padded_m = m_bits + [zero] * (n - m_len)

    current_remainder = x_bits[:]

    max_subtractions = 2**n

    for step in range(max_subtractions):
        less, equal, greater = n_bit_comparator(
            circuit, current_remainder, padded_m, parent_group=this_group
        )
        can_subtract = circuit.add_node(
            "or", "OR", inputs=[equal, greater], group_id=this_group_id
        ).ports[2]
        current_remainder = conditional_subtract(
            circuit, current_remainder, padded_m, can_subtract, parent_group=this_group
        )

    return current_remainder


def modulo_circuit_optimized(circuit, x_bits, m_bits, parent_group=None):
    """
    Most efficient version: processes multiple bit positions when possible.
    """
    this_group = circuit.add_group("MODULO_CIRCUIT_OPTIMIZED")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(x_bits)
    m_len = len(m_bits)
    assert m_len <= n, "Modulus must not be wider than dividend"

    zero = constant_zero(circuit, x_bits[0])
    one = constant_one(circuit, x_bits[0])
    current_remainder = x_bits[:]

    shifted_moduli = []
    for shift in range(n - m_len + 1):

        shift_bin_list = utils.int2binlist(shift, len(x_bits))
        shift_repr = [one if bit else zero for bit in shift_bin_list]
        shifted_m = n_left_shift(circuit, m_bits, shift_repr, parent_group=this_group)
        shifted_moduli.append(shifted_m)

    for i in range(len(shifted_moduli) - 1, -1, -1):
        shifted_m = shifted_moduli[i]

        less, equal, greater = n_bit_comparator(
            circuit, current_remainder, shifted_m, parent_group=this_group
        )
        can_subtract = or_gate(circuit, [equal, greater], parent_group=this_group)

        current_remainder = conditional_subtract(
            circuit, current_remainder, shifted_m, can_subtract, parent_group=this_group
        )

    return current_remainder


def modulo_circuit(circuit, x_bits, m_bits, parent_group=None):
    this_group = circuit.add_group("MODULO")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    return modulo_circuit_optimized(circuit, x_bits, m_bits, parent_group=this_group)

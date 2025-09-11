from .comparators import *
from .subtractors import *
from .shifters import *

"""
def modulo_circuit(circuit, x_bits, m_bits):
    n = len(x_bits)
    assert len(m_bits) == n, "Both inputs must have the same bit length"
    m_powers = [m_bits]
    for i in range(1, n):
        prev_power = m_powers[i-1]
        next_power = [constant_zero(circuit, prev_power[0])] + prev_power[:-1]
        m_powers.append(next_power)
    current_remainder = x_bits
    for i in range(len(m_powers)-1, -1, -1):
        power_m = m_powers[i]
        less, equals, greater = n_bit_comparator(circuit, current_remainder, power_m)
        can_subtract = circuit.add_node("not", "NOT", inputs=[less]).ports[1]
        current_remainder = conditional_subtract(circuit, current_remainder, power_m, can_subtract)
    return current_remainder
"""

"""
def modulo_circuit(circuit, x_bits, m_bits):
    n = len(x_bits)
    assert len(m_bits) <= n, "Modulus must not be wider than dividend"

    current_remainder = x_bits

    zero = constant_zero(circuit, x_bits[0])
    one = constant_one(circuit, m_bits[0])

    for shift in range(n - len(m_bits), -1, -1):
        shift_bin_list = utils.int2binlist(shift, len(x_bits))
        shift_repr = [one if bit else zero for bit in shift_bin_list]
        
        shifted_m = n_left_shift(circuit, m_bits, shift_repr)
        
        less, _, _ = n_bit_comparator(circuit, current_remainder, shifted_m)
        can_subtract = circuit.add_node("not", "NOT", inputs=[less]).ports[1]

        current_remainder = conditional_subtract(circuit, current_remainder, shifted_m, can_subtract)

    return current_remainder
"""

"""
def modulo_circuit(circuit, x_bits, m_bits):
    
    n = len(x_bits)
    m_len = len(m_bits)
    current_remainder = x_bits
    
    # We need to do at most 2^n iterations, but that's impractical
    # Instead, we'll do a more reasonable number based on bit width

    padded_m = m_bits
    max_iterations = (1 << (n - m_len + 1)) if n >= m_len else 1
    
    for _ in range(max_iterations):
        # Check if current_remainder >= padded_m
        less, equal, greater = n_bit_comparator(circuit, current_remainder, padded_m)
        
        # can_subtract = (current_remainder >= padded_m)
        can_subtract = circuit.add_node("or", "OR", inputs=[equal, greater]).ports[2]
        
        current_remainder = conditional_subtract(circuit, current_remainder, padded_m, can_subtract)

    return current_remainder
"""


# Individual subtractions and by that slow / large
def slow_modulo_circuit(circuit, x_bits, m_bits, parent_group=None):
    this_group = circuit.add_group("SLOW_MODULO_CIRCUIT")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    n = len(x_bits)
    m_len = len(m_bits)
    assert m_len <= n, "Modulus must not be wider than dividend"

    zero = constant_zero(circuit, x_bits[0], parent_group=this_group)

    # Pad modulus to same width as dividend
    padded_m = m_bits + [zero] * (n - m_len)

    current_remainder = x_bits[:]

    # Unroll a fixed number of subtractions (enough to handle worst case)
    # For n-bit numbers, we need at most 2^(n-m_len) subtractions
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


# Subtracts the shifted moduli, loop within log(n)
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

    # Create a table of shifted moduli
    shifted_moduli = []
    for shift in range(n - m_len + 1):

        shift_bin_list = utils.int2binlist(shift, len(x_bits))
        shift_repr = [one if bit else zero for bit in shift_bin_list]
        shifted_m = n_left_shift(circuit, m_bits, shift_repr, parent_group=this_group)
        shifted_moduli.append(shifted_m)

    # Process from largest shift to smallest
    for i in range(len(shifted_moduli) - 1, -1, -1):
        shifted_m = shifted_moduli[i]

        # Check if we can subtract this shifted modulus
        less, equal, greater = n_bit_comparator(
            circuit, current_remainder, shifted_m, parent_group=this_group
        )
        # can_subtract = circuit.add_node("or", "OR", inputs=[equal, greater]).ports[2]
        can_subtract = or_gate(circuit, [equal, greater], parent_group=this_group)

        # Conditionally subtract
        current_remainder = conditional_subtract(
            circuit, current_remainder, shifted_m, can_subtract, parent_group=this_group
        )

    return current_remainder


def modulo_circuit(circuit, x_bits, m_bits, parent_group=None):
    this_group = circuit.add_group("MODULO")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    return modulo_circuit_optimized(circuit, x_bits, m_bits, parent_group=this_group)
    # return slow_modulo_circuit(circuit, x_bits, m_bits, parent_group=m_group)

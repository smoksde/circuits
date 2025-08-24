import math

from .shifters import *
from .constants import *
from .adders import *
from .multipliers import *
from .comparators import *
from .multiplexers import *
from .subtractors import *
from .modular import *
from .montgomery_ladder import montgomery_ladder
from .manipulators import conditional_zeroing, max_tree_iterative

from .beame import lemma_4_1
from .beame import lemma_5_1
from .beame import theorem_4_2
from .beame import theorem_5_2
from .beame import theorem_5_3


def binary_list_to_int(binary_list):
    return sum(bit * (2**i) for i, bit in enumerate(binary_list))


"""
def or_tree_recursive(circuit, input_list):
    if len(input_list) == 1:
        return input_list[0]
    
    if len(input_list) == 2:
        or_node = circuit.add_node("or", "OR", inputs=input_list)
        return or_node.ports[2]
    
    mid = len(input_list) // 2
    left = or_tree_recursive(circuit, input_list[:mid])
    right = or_tree_recursive(circuit, input_list[mid:])
    or_node = circuit.add_node("or", "OR", inputs=[left, right])
    return or_node.ports[2]"""


"""
def precompute_a_i(const_zero, const_one, int_m, n):
    print("int_m ", int_m)
    print("n ", n)
    a_i_lists = []
    for i in range(n):
        print("index ", i)
        calc = (2**i) % int_m
        print("calc ", calc)
        a = []
        for j in range(n):
            if calc % 2 == 0:
                a.append(const_zero)
                print("0")
            else:
                a.append(const_one)
                print("1")
            calc >>= 1
        a_i_lists.append(a)
    return a_i_lists
"""

"""
def small_mod_lemma_4_1(circuit, x_list, m_list, int_m):

    n = len(x_list)

    print("m, n: ", int_m, n)

    input = circuit.add_node("input", "INPUT")
    const_zero = constant_zero(circuit, input.ports[0])
    const_one = constant_one(circuit, input.ports[0])

    # precompute constants: a_im = 2^i mod m values
    a_i_lists = precompute_a_i(const_zero, const_one, int_m, n)

    print("a_i_lists", a_i_lists)

    # compute summands of y
    summands = []
    for ind, x_i in enumerate(x_list):
        not_x_i_node = circuit.add_node("not", "NOT", inputs=[x_i])
        summand = conditional_zeroing(circuit, a_i_lists[ind], not_x_i_node.ports[1])
        summands.append(summand)

    y, carry = adder_tree_recursive(circuit, summands, const_zero)

    print("y: ", y)

    results = []
    for i in range(n):
        bin_i = utils.int2binlist(i, bit_len=len(x_list))
        coef = [const_zero if bit == 0 else const_one for bit in bin_i]
        print("coef len", len(coef))
        print("m_list len", len(m_list))
        mult_m = wallace_tree_multiplier(circuit, m_list, coef)
        mult_m = mult_m[: -(len(mult_m) // 2)]

        # mult_m should not be greater than y since y - mult_m should be in [0, m[
        _, _, greater = n_bit_comparator(circuit, mult_m, y)

        # mult_m_plus_m should not be less than y since y - mult_m should be in [0, m[
        print("len mult_m ", len(mult_m))
        print("len m_list ", len(m_list))
        mult_m_plus_m, _ = ripple_carry_adder(circuit, mult_m, m_list, const_zero)
        print("len mult_m_plus_m ", len(mult_m_plus_m))
        print("len y ", len(y))
        less, _, _ = n_bit_comparator(circuit, mult_m_plus_m, y)

        negative_mult_m = two_complement(circuit, mult_m)
        diff, carry = ripple_carry_adder(circuit, y, negative_mult_m, const_zero)

        result = conditional_zeroing(circuit, diff, greater)
        result = conditional_zeroing(circuit, diff, less)

        # always a zero list gets appended if the conditions are not fullfilled, else the result (x mod m) gets appended as a list
        results.append(result)

        # negative_mult_m = two_complement(circuit, mult_m)

        # sum, carry = ripple_carry_adder(circuit, y, negative_mult_m, const_zero)

        # is_negative = sign_detector(circuit, sum) # if 1 then negative
        # result = conditional_zeroing(circuit, coef, is_negative)
        # less, equals, greater = n_bit_comparator(circuit, sum, m_list)
        # not_less_node = circuit.add_node("not", "NOT", inputs=[less])
        # result = conditional_zeroing(circuit, result, not_less_node.ports[1])
        # results.append(result)

    # final = []
    # for i in range(n):
    #    for j in range(len(results)):
    #        curr_list = []
    #        curr_list.append(results[j][i])
    #    bit = or_tree_recursive(circuit, curr_list)
    #    final.append(bit)
    sums, carry = adder_tree_recursive(circuit, results, const_zero)
    return sums"""


def log2_estimate(circuit, x_list):
    n = len(x_list)

    zero = constant_zero(circuit, x_list[0])

    result_bits = [zero] * n
    found_any_one = zero

    for i in range(n):
        bit_position = n - 1 - i
        current_bit = x_list[bit_position]
        is_first_one = circuit.add_node(
            "and",
            f"IS_FIRST_ONE_{i}",
            inputs=[
                current_bit,
                circuit.add_node("not", f"NOT_FOUND_{i}", inputs=[found_any_one]).ports[
                    1
                ],
            ],
        ).ports[2]
        found_any_one = circuit.add_node(
            "or", f"FOUND_UPDATE_{i}", inputs=[found_any_one, current_bit]
        ).ports[2]

        for j in range(n):
            bit_j_of_position = 1 if (bit_position >> j) & 1 else 0
            if bit_j_of_position == 1:
                result_bits[j] = circuit.add_node(
                    "or",
                    f"RESULT_BIT_{j}_{bit_position}",
                    inputs=[result_bits[j], is_first_one],
                ).ports[2]

    return result_bits


def reciprocal_newton_raphson(circuit, m_bits, n):
    # Computes 1 / m using Newton-Raphson method.
    # x_{i+1} = x_i * (2 - m * x_i)
    # Converges to 1/m if x_0 is a good initial approximation.

    x_0 = initial_approximation(circuit, m_bits, n)
    m_x0 = fixed_point_multiply(circuit, m_bits, x_0, n)
    two_minus_mx0 = fixed_point_subtract_from_two(circuit, m_x0, n)
    x_1 = fixed_point_multiply(circuit, x_0, two_minus_mx0, n)
    return x_1


def scaled_multiply(circuit, a_bits, b_bits, n):
    full_product = wallace_tree_multiplier(circuit, a_bits, b_bits)
    if len(full_product) >= n + n // 2:
        result = full_product[n // 2 : n // 2 + n]
    else:
        zero = constant_zero(circuit, a_bits[0])
        result = [zero] * (n // 2) + full_product[: n // 2]
        result = result[:n]
    return result


def fixed_point_multiply(circuit, a_bits, b_bits, n):
    product = wallace_tree_multiplier(circuit, a_bits, b_bits)
    result = product[n : 3 * n]
    return result


def fixed_point_subtract_from_two(circuit, a_bits, bit_length):
    zero = constant_zero(circuit, a_bits[0])
    one = constant_one(circuit, a_bits[0])
    two_fixed_point = [zero] * bit_length
    two_fixed_point[bit_length // 2 + 1] = one
    result = subtract(circuit, two_fixed_point, a_bits)
    return result


def initial_approximation(circuit, m_bits, n):
    log2_m = log2_estimate(circuit, m_bits)
    zero = constant_zero(circuit, m_bits[0])
    one = constant_one(circuit, m_bits[0])
    n_bits = [zero] * n
    n_binary = bin(n)[2:]
    for i, bit in enumerate(reversed(n_binary)):
        if bit == "1":
            n_bits[i] = one

    shift_amount = subtract(circuit, n_bits, log2_m)
    one_fixed_point = [zero] * n
    one_fixed_point[n // 2] = one
    initial_approx = n_left_shift(circuit, one_fixed_point, shift_amount)
    return initial_approx


def modular_exponentiation(circuit, base, exponent, modulus, parent_group=None):
    me_group = circuit.add_group("MODULAR_EXPONENTIATION")
    me_group.set_parent(parent_group)
    n = len(base)
    assert n == len(exponent) and n == len(
        modulus
    ), "All input must have the same bit length"

    zero = constant_zero(circuit, base[0], parent_group=me_group)
    one = constant_one(circuit, base[0], parent_group=me_group)

    result = [zero] * n
    result[0] = one
    base_mod = modulo_circuit(circuit, base, modulus, parent_group=me_group)
    for i in range(n):
        bit_pos = n - 1 - i
        current_bit = exponent[bit_pos]
        squared = wallace_tree_multiplier(
            circuit, result, result, parent_group=me_group
        )
        squared = squared[: len(base)]
        squared_mod = modulo_circuit(circuit, squared, modulus, parent_group=me_group)
        with_multiply = wallace_tree_multiplier(
            circuit, squared_mod, base_mod, parent_group=me_group
        )
        with_multiply = with_multiply[: len(base)]
        multiply_mod = modulo_circuit(
            circuit, with_multiply, modulus, parent_group=me_group
        )
        new_result = [None] * n
        for j in range(n):
            not_bit = circuit.add_node(
                "not",
                f"NOT_BIT_{bit_pos}_{j}",
                inputs=[current_bit],
                group_id=me_group.id,
            ).ports[1]
            and1 = circuit.add_node(
                "and",
                f"AND_MULT_{bit_pos}_{j}",
                inputs=[current_bit, multiply_mod[j]],
                group_id=me_group.id,
            ).ports[2]
            and2 = circuit.add_node(
                "and",
                f"AND_SQR_{bit_pos}_{j}",
                inputs=[not_bit, squared_mod[j]],
                group_id=me_group.id,
            ).ports[2]
            new_result[j] = circuit.add_node(
                "or",
                f"OR_RESULT_{bit_pos}_{j}",
                inputs=[and1, and2],
                group_id=me_group.id,
            ).ports[2]
        result = new_result
    return result


CIRCUIT_FUNCTIONS = {
    "xnor_gate": lambda cg, bit_len: setup_xnor_gate(cg, bit_len=bit_len),
    "one_bit_comparator": lambda cg, bit_len: setup_one_bit_comparator(
        cg, bit_len=bit_len
    ),
    "n_bit_comparator": lambda cg, bit_len: setup_n_bit_comparator(cg, bit_len=bit_len),
    "constant_zero": lambda cg, bit_len: setup_constant_zero(cg, bit_len=bit_len),
    "constant_one": lambda cg, bit_len: setup_constant_one(cg, bit_len=bit_len),
    "and_tree_iterative": lambda cg, bit_len: setup_and_tree_iterative(
        cg, bit_len=bit_len
    ),
    "or_tree_iterative": lambda cg, bit_len: setup_or_tree_iterative(
        cg, bit_len=bit_len
    ),
    "half_adder": lambda cg, bit_len: setup_half_adder(cg, bit_len=bit_len),
    "full_adder": lambda cg, bit_len: setup_full_adder(cg, bit_len=bit_len),
    "ripple_carry_adder": lambda cg, bit_len: setup_ripple_carry_adder(
        cg, bit_len=bit_len
    ),
    "carry_look_ahead_adder": lambda cg, bit_len: setup_carry_look_ahead_adder(
        cg, bit_len=bit_len
    ),
    "wallace_tree_multiplier": lambda cg, bit_len: setup_wallace_tree_multiplier(
        cg, bit_len=bit_len
    ),
    # "multiplexer": lambda cg, bit_len: setup_multiplexer(cg, bit_len=bit_len),
    "adder_tree_recursive": lambda cg, bit_len: setup_adder_tree_recursive(
        cg, bit_len=bit_len
    ),
    # "small_mod_lemma_4_1": lambda cg, bit_len: setup_small_mod_lemma_4_1(cg, bit_len=bit_len),
    # "precompute_a_i": lambda cg, bit_len: setup_precompute_a_i(cg, bit_len=bit_len),
    "conditional_zeroing": lambda cg, bit_len: setup_conditional_zeroing(
        cg, bit_len=bit_len
    ),
    "conditional_subtract": lambda cg, bit_len: setup_conditional_subtract(
        cg, bit_len=bit_len
    ),
    "next_power_of_two": lambda cg, bit_len: setup_next_power_of_two(
        cg, bit_len=bit_len
    ),
    "one_left_shift": lambda cg, bit_len: setup_one_left_shift(cg, bit_len=bit_len),
    "one_right_shift": lambda cg, bit_len: setup_one_right_shift(cg, bit_len=bit_len),
    "n_left_shift": lambda cg, bit_len: setup_n_left_shift(cg, bit_len=bit_len),
    "n_right_shift": lambda cg, bit_len: setup_n_right_shift(cg, bit_len=bit_len),
    "log2_estimate": lambda cg, bit_len: setup_log2_estimate(cg, bit_len=bit_len),
    "reciprocal_newton_raphson": lambda cg, bit_len: setup_reciprocal_newton_raphson(
        cg, bit_len=bit_len
    ),
    "modulo_circuit": lambda cg, bit_len: setup_modulo_circuit(cg, bit_len=bit_len),
    "slow_modulo_circuit": lambda cg, bit_len: setup_slow_modulo_circuit(
        cg, bit_len=bit_len
    ),
    "optimized_modulo_circuit": lambda cg, bit_len: setup_optimized_modulo_circuit(
        cg, bit_len=bit_len
    ),
    "modular_exponentiation": lambda cg, bit_len: setup_modular_exponentiation(
        cg, bit_len=bit_len
    ),
    "montgomery_ladder": lambda cg, bit_len: setup_montgomery_ladder(
        cg, bit_len=bit_len
    ),
}


def setup_theorem_4_2_step_1(cg: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [cg.add_input_nodes(n, "INPUT") for _ in range(n)]
    P_NODES = cg.add_input_nodes(n, "INPUT")
    PEXPL_NODES = cg.add_input_nodes(n, "INPUT")

    X_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    P_PORTS = cg.get_input_nodes_ports(P_NODES)
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)

    EXPONENTS_PORTS = theorem_4_2.step_1(cg, X_LIST_PORTS, P_PORTS, PEXPL_PORTS)

    EXPONENTS_NODES = [
        cg.generate_output_nodes_from_ports(ports) for ports in EXPONENTS_PORTS
    ]
    return X_LIST_NODES, P_NODES, PEXPL_NODES, EXPONENTS_NODES

def setup_theorem_4_2_step_1_with_lemma_4_1(cg: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [cg.add_input_nodes(n, "INPUT") for _ in range(n)]
    P_NODES = cg.add_input_nodes(n, "INPUT")
    PEXPL_NODES = cg.add_input_nodes(n, "INPUT")

    X_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    P_PORTS = cg.get_input_nodes_ports(P_NODES)
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)

    EXPONENTS_PORTS = theorem_4_2.step_1_with_lemma_4_1(cg, X_LIST_PORTS, P_PORTS, PEXPL_PORTS)

    EXPONENTS_NODES = [
        cg.generate_output_nodes_from_ports(ports) for ports in EXPONENTS_PORTS
    ]
    return X_LIST_NODES, P_NODES, PEXPL_NODES, EXPONENTS_NODES

def setup_theorem_4_2_step_1_with_precompute(cg: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [cg.add_input_nodes(n, "INPUT") for _ in range(n)]
    P_NODES = cg.add_input_nodes(n, "INPUT")
    PEXPL_NODES = cg.add_input_nodes(n, "INPUT")

    X_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    P_PORTS = cg.get_input_nodes_ports(P_NODES)
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)

    EXPONENTS_PORTS = theorem_4_2.step_1_with_precompute(cg, X_LIST_PORTS, P_PORTS, PEXPL_PORTS)

    EXPONENTS_NODES = [
        cg.generate_output_nodes_from_ports(ports) for ports in EXPONENTS_PORTS
    ]
    return X_LIST_NODES, P_NODES, PEXPL_NODES, EXPONENTS_NODES


def setup_theorem_4_2_step_2(cg: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [cg.add_input_nodes(n, "INPUT") for _ in range(n)]
    P_NODES = cg.add_input_nodes(n, "INPUT")
    J_LIST_NODES = [cg.add_input_nodes(n, "INPUT") for _ in range(n)]

    X_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    P_PORTS = cg.get_input_nodes_ports(P_NODES)
    J_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in J_LIST_NODES]

    Y_LIST_PORTS = theorem_4_2.step_2(cg, X_LIST_PORTS, P_PORTS, J_LIST_PORTS)

    Y_LIST_NODES = [
        cg.generate_output_nodes_from_ports(ports) for ports in Y_LIST_PORTS
    ]
    return X_LIST_NODES, P_NODES, J_LIST_NODES, Y_LIST_NODES


def setup_theorem_4_2_compute_sum(cg: CircuitGraph, bit_len=4):
    n = bit_len
    J_LIST_NODES = [cg.add_input_nodes(n, "INPUT") for _ in range(n)]
    J_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in J_LIST_NODES]
    J_PORTS = theorem_4_2.compute_sum(cg, J_LIST_PORTS)
    J_NODES = cg.generate_output_nodes_from_ports(J_PORTS)
    return J_LIST_NODES, J_NODES


def setup_theorem_4_2_step_4(cg: CircuitGraph, bit_len=4):
    n = bit_len
    P_NODES = cg.add_input_nodes(n, "INPUT")
    PEXPL_NODES = cg.add_input_nodes(n, "INPUT")
    P_PORTS = cg.get_input_nodes_ports(P_NODES)
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)
    FLAG = theorem_4_2.step_4(cg, P_PORTS, PEXPL_PORTS)
    FLAG_NODE = cg.generate_output_node_from_port(FLAG)
    return P_NODES, PEXPL_NODES, FLAG_NODE


def setup_theorem_4_2_A_step_5(cg: CircuitGraph, bit_len=4):
    n = bit_len
    Y_LIST_NODES = [cg.add_input_nodes(n, "INPUT") for _ in range(n)]
    PEXPL_NODES = cg.add_input_nodes(n, "INPUT")
    Y_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in Y_LIST_NODES]
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)
    A_LIST_PORTS = theorem_4_2.A_step_5(cg, Y_LIST_PORTS, PEXPL_PORTS)
    A_LIST_NODES = [
        cg.generate_output_nodes_from_ports(ports) for ports in A_LIST_PORTS
    ]
    return Y_LIST_NODES, PEXPL_NODES, A_LIST_NODES


def setup_theorem_4_2_A_step_7(cg: CircuitGraph, bit_len=4):
    n = bit_len
    A_NODES = cg.add_input_nodes(n, "INPUT")
    PEXPL_NODES = cg.add_input_nodes(n, "INPUT")
    A_PORTS = cg.get_input_nodes_ports(A_NODES)
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)
    A_HAT_PORTS = theorem_4_2.A_step_7(cg, A_PORTS, PEXPL_PORTS)
    A_HAT_NODES = cg.generate_output_nodes_from_ports(A_HAT_PORTS)
    return A_NODES, PEXPL_NODES, A_HAT_NODES


def setup_theorem_4_2_A_step_8(cg: CircuitGraph, bit_len=4):
    n = bit_len
    A_HAT_NODES = cg.add_input_nodes(n)
    PEXPL_NODES = cg.add_input_nodes(n)

    A_HAT_PORTS = cg.get_input_nodes_ports(A_HAT_NODES)
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)
    Y_PRODUCT_PORTS = theorem_4_2.A_step_8(
        cg,
        A_HAT_PORTS,
        PEXPL_PORTS,
    )
    Y_PRODUCT_NODES = cg.generate_output_nodes_from_ports(Y_PRODUCT_PORTS)
    return A_HAT_NODES, PEXPL_NODES, Y_PRODUCT_NODES


def setup_theorem_4_2_B_step_5(cg: CircuitGraph, bit_len=4):
    n = bit_len
    Y_LIST_NODES = [cg.add_input_nodes(n) for _ in range(n)]
    L_NODES = cg.add_input_nodes(n)
    Y_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in Y_LIST_NODES]
    L_PORTS = cg.get_input_nodes_ports(L_NODES)
    A_LIST_PORTS, B_LIST_PORTS = theorem_4_2.B_step_5(cg, Y_LIST_PORTS, L_PORTS)
    A_LIST_NODES = [
        cg.generate_output_nodes_from_ports(ports) for ports in A_LIST_PORTS
    ]
    B_LIST_NODES = [
        cg.generate_output_nodes_from_ports(ports) for ports in B_LIST_PORTS
    ]
    return Y_LIST_NODES, L_NODES, A_LIST_NODES, B_LIST_NODES


def setup_theorem_4_2_B_step_7(cg: CircuitGraph, bit_len=4):
    n = bit_len
    A_NODES = cg.add_input_nodes(n)
    B_NODES = cg.add_input_nodes(n)
    L_NODES = cg.add_input_nodes(n)

    A_PORTS = cg.get_input_nodes_ports(A_NODES)
    B_PORTS = cg.get_input_nodes_ports(B_NODES)
    L_PORTS = cg.get_input_nodes_ports(L_NODES)

    A_HAT_PORTS, B_HAT_PORTS = theorem_4_2.B_step_7(cg, A_PORTS, B_PORTS, L_PORTS)

    A_HAT_NODES = cg.generate_output_nodes_from_ports(A_HAT_PORTS)
    B_HAT_NODES = cg.generate_output_nodes_from_ports(B_HAT_PORTS)

    return A_NODES, B_NODES, L_NODES, A_HAT_NODES, B_HAT_NODES


def setup_theorem_4_2_step_9(cg: CircuitGraph, bit_len=4):
    n = bit_len

    P_NODES = cg.add_input_nodes(n)
    J_NODES = cg.add_input_nodes(n)
    PEXPL_NODES = cg.add_input_nodes(n)
    Y_PRODUCT_NODES = cg.add_input_nodes(n)

    P_PORTS = cg.get_input_nodes_ports(P_NODES)
    J_PORTS = cg.get_input_nodes_ports(J_NODES)
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)
    Y_PRODUCT_PORTS = cg.get_input_nodes_ports(Y_PRODUCT_NODES)

    RESULT_PORTS = theorem_4_2.step_9(
        cg, P_PORTS, J_PORTS, PEXPL_PORTS, Y_PRODUCT_PORTS
    )
    RESULT_NODES = cg.generate_output_nodes_from_ports(RESULT_PORTS)

    return P_NODES, J_NODES, PEXPL_NODES, Y_PRODUCT_NODES, RESULT_NODES


def setup_theorem_4_2(cg: CircuitGraph, bit_len=4):
    n = bit_len

    X_LIST_NODES = [cg.add_input_nodes(n) for _ in range(n)]
    PEXPL_NODES = cg.add_input_nodes(n)

    X_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)

    RESULT_PORTS = theorem_4_2.theorem_4_2(cg, X_LIST_PORTS, PEXPL_PORTS)
    RESULT_NODES = cg.generate_output_nodes_from_ports(RESULT_PORTS)

    return X_LIST_NODES, PEXPL_NODES, RESULT_NODES


def setup_theorem_4_2_for_theorem_5_2(cg: CircuitGraph, bit_len=4):
    n = bit_len

    X_LIST_NODES = [cg.add_input_nodes(n) for _ in range(n)]
    PEXPL_NODES = cg.add_input_nodes(n)

    X_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    PEXPL_PORTS = cg.get_input_nodes_ports(PEXPL_NODES)

    RESULT_PORTS = theorem_4_2.theorem_4_2_for_theorem_5_2(
        cg, X_LIST_PORTS, PEXPL_PORTS
    )
    RESULT_NODES = cg.generate_output_nodes_from_ports(RESULT_PORTS)

    return X_LIST_NODES, PEXPL_NODES, RESULT_NODES


def setup_theorem_4_2_precompute_lookup_tables_B(cg: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = cg.add_input_nodes(1, "INPUT")[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    TABLE_ZERO, TABLE_ONE = theorem_4_2.precompute_lookup_tables_B(
        cg, zero_port, one_port, n
    )
    TABLE_ZERO_NODES = []
    for row in TABLE_ZERO:
        nodes = [
            cg.generate_output_nodes_from_ports(entry, label="OUTPUT") for entry in row
        ]
        TABLE_ZERO_NODES.append(nodes)
    TABLE_ONE_NODES = []
    for row in TABLE_ONE:
        nodes = [
            cg.generate_output_nodes_from_ports(entry, label="OUTPUT") for entry in row
        ]
        TABLE_ONE_NODES.append(nodes)
    return TABLE_ZERO_NODES, TABLE_ONE_NODES


def setup_theorem_4_2_precompute_lookup_generator_powers(cg: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = cg.add_input_nodes(1, "INPUT")[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    TABLE = theorem_4_2.precompute_lookup_generator_powers(cg, zero_port, one_port, n)
    TABLE_NODES = []
    for row in TABLE:
        nodes = [
            cg.generate_output_nodes_from_ports(entry, label="OUTPUT") for entry in row
        ]
        TABLE_NODES.append(nodes)
    return TABLE_NODES


def setup_theorem_4_2_precompute_lookup_division(cg: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = cg.add_input_nodes(1, "INPUT")[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    TABLE = theorem_4_2.precompute_lookup_division(cg, zero_port, one_port, n)
    TABLE_NODES = []
    for row in TABLE:
        nodes = [
            cg.generate_output_nodes_from_ports(entry, label="OUTPUT") for entry in row
        ]
        TABLE_NODES.append(nodes)
    return TABLE_NODES


def setup_theorem_4_2_precompute_lookup_powers(cg: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = cg.add_input_nodes(1, "INPUT")[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    O = theorem_4_2.precompute_lookup_powers(cg, zero_port, one_port, n)
    O_NODES = []
    for o in O:
        powers_of_p_nodes = []
        for power in o:
            power_nodes = cg.generate_output_nodes_from_ports(power, label="OUTPUT")
            powers_of_p_nodes.append(power_nodes)
        O_NODES.append(powers_of_p_nodes)
    return O_NODES


def setup_theorem_4_2_precompute_lookup_is_prime_power(cg: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = cg.add_input_nodes(1, "INPUT")[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    O = theorem_4_2.precompute_lookup_is_prime_power(cg, zero_port, one_port, n)
    O_NODES = cg.generate_output_nodes_from_ports(O, label="OUTPUT")
    return O_NODES


def setup_theorem_4_2_precompute_lookup_p_l(cg: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = cg.add_input_nodes(1, "INPUT")[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    P_TABLE, L_TABLE = theorem_4_2.precompute_lookup_p_l(cg, zero_port, one_port, n)
    P_TABLE_NODES = []
    L_TABLE_NODES = []
    for p, l in zip(P_TABLE, L_TABLE):
        p_nodes = cg.generate_output_nodes_from_ports(p)
        l_nodes = cg.generate_output_nodes_from_ports(l)
        P_TABLE_NODES.append(p_nodes)
        L_TABLE_NODES.append(l_nodes)
    return P_TABLE_NODES, L_TABLE_NODES


def setup_theorem_4_2_precompute_lookup_pexpl_minus_pexpl_minus_one(
    cg: CircuitGraph, bit_len=4
):
    n = bit_len
    input_node = cg.add_input_nodes(1, "INPUT")[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    TABLE = theorem_4_2.precompute_lookup_pexpl_minus_pexpl_minus_one(
        cg, zero_port, one_port, n
    )
    TABLE_NODES = []
    for ports in TABLE:
        nodes = cg.generate_output_nodes_from_ports(ports)
        TABLE_NODES.append(nodes)
    return TABLE_NODES


def setup_theorem_5_3_precompute_good_modulus_sequence(cg: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = cg.add_input_nodes(1)[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    PRIMES_PORTS, PRIMES_PRODUCT_PORTS = theorem_5_3.precompute_good_modulus_sequence(
        cg, zero_port, one_port, n
    )
    PRIMES_NODES = []
    for ports in PRIMES_PORTS:
        nodes = cg.generate_output_nodes_from_ports(ports)
        PRIMES_NODES.append(nodes)
    PRIMES_PRODUCT_NODES = cg.generate_output_nodes_from_ports(PRIMES_PRODUCT_PORTS)
    return PRIMES_NODES, PRIMES_PRODUCT_NODES


def setup_lemma_5_1_precompute_u_list(cg: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = cg.add_input_nodes(1)[0]
    input_port = cg.get_input_node_port(input_node)
    zero_port = constant_zero(cg, input_port)
    one_port = constant_one(cg, input_port)
    U_LIST_PORTS = lemma_5_1.precompute_u_list(cg, zero_port, one_port, n)
    U_LIST_NODES = [
        cg.generate_output_nodes_from_ports(ports) for ports in U_LIST_PORTS
    ]
    return U_LIST_NODES


def setup_lemma_5_1(cg: CircuitGraph, bit_len=4):
    n = bit_len
    X_MOD_C_I_LIST_NODES = [cg.add_input_nodes(n * n) for _ in range(n)]
    C_NODES = cg.add_input_nodes(n * n)
    X_MOD_C_I_LIST_PORTS = [
        cg.get_input_nodes_ports(nodes) for nodes in X_MOD_C_I_LIST_NODES
    ]
    C_PORTS = cg.get_input_nodes_ports(C_NODES)
    RESULT_PORTS = lemma_5_1.lemma_5_1(cg, X_MOD_C_I_LIST_PORTS, C_PORTS)
    RESULT_NODES = cg.generate_output_nodes_from_ports(RESULT_PORTS)
    return X_MOD_C_I_LIST_NODES, C_NODES, RESULT_NODES


def setup_theorem_5_2(cg: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [cg.add_input_nodes(n) for _ in range(n)]
    X_LIST_PORTS = [cg.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    RESULT_PORTS = theorem_5_2.theorem_5_2(cg, X_LIST_PORTS)
    RESULT_NODES = cg.generate_output_nodes_from_ports(RESULT_PORTS)
    return X_LIST_NODES, RESULT_NODES


def setup_max_tree_iterative(cg: CircuitGraph, num_amount=4, bit_len=4):
    VALUES_NODES = []
    VALUES_PORTS = []

    for i in range(num_amount):
        X = cg.add_input_nodes(bit_len, label=f"X_{i}")
        X_PORTS = cg.get_input_nodes_ports(X)
        VALUES_NODES.append(X)
        VALUES_PORTS.append(X_PORTS)
    MAX_PORTS = max_tree_iterative(cg, VALUES_PORTS)
    MAX_NODES = cg.generate_output_nodes_from_ports(MAX_PORTS, label="OUTPUT")
    return VALUES_NODES, MAX_NODES


def setup_adder_tree_iterative(cg: CircuitGraph, num_amount=4, bit_len=4):
    SUMMANDS = []
    SUMMANDS_PORTS = []

    for i in range(num_amount):
        X = cg.add_input_nodes(bit_len, label=f"X_{i}")
        X_PORTS = cg.get_input_nodes_ports(X)
        SUMMANDS.append(X)
        SUMMANDS_PORTS.append(X_PORTS)
    zero = constant_zero(cg, SUMMANDS_PORTS[0][0])
    O = adder_tree_iterative(cg, SUMMANDS_PORTS, zero)
    O_NODES = cg.generate_output_nodes_from_ports(O, label="OUTPUT")
    return SUMMANDS, O_NODES


def setup_lemma_4_1(cg: CircuitGraph, bit_len=4):
    X = cg.add_input_nodes(bit_len, label="X")
    X_PORTS = cg.get_input_nodes_ports(X)
    M = cg.add_input_nodes(bit_len, label="M")
    M_PORTS = cg.get_input_nodes_ports(M)
    O = lemma_4_1.lemma_4_1(cg, X_PORTS, M_PORTS)
    O_NODES = cg.generate_output_nodes_from_ports(O, label="OUTPUT")
    return X, M, O_NODES


def setup_lemma_4_1_reduce_in_parallel(cg: CircuitGraph, bit_len=4):
    Y = cg.add_input_nodes(bit_len, label="Y")
    Y_PORTS = cg.get_input_nodes_ports(Y)
    M = cg.add_input_nodes(bit_len, label="M")
    M_PORTS = cg.get_input_nodes_ports(M)
    n = bit_len
    O = lemma_4_1.reduce_in_parallel(cg, Y_PORTS, M_PORTS, n)
    O_NODES = cg.generate_output_nodes_from_ports(O, "OUTPUT")
    return Y, M, O_NODES


# Returns M -> List[Node] and O_NODES -> List[List[Node]]
def setup_lemma_4_1_provide_aims_given_m(cg: CircuitGraph, bit_len=4):
    M = cg.add_input_nodes(bit_len, label="M")
    M_PORTS = cg.get_input_nodes_ports(M)
    O = lemma_4_1.provide_aims_given_m(cg, M_PORTS)
    O_NODES = []
    for num in O:
        num_nodes = cg.generate_output_nodes_from_ports(num)
        O_NODES.append(num_nodes)
    return M, O_NODES


# O_NODES is a List[List[Port]]
def setup_lemma_4_1_compute_summands(cg: CircuitGraph, num_amount=4, bit_len=4):
    X = cg.add_input_nodes(bit_len, label="X")
    X_PORTS = cg.get_input_nodes_ports(X)
    NUMS = []
    NUMS_PORTS = []
    for i in range(num_amount):
        num = cg.add_input_nodes(bit_len, label=f"NUM_{i}")
        num_ports = cg.get_input_nodes_ports(num)
        NUMS.append(num)
        NUMS_PORTS.append(num_ports)
    O = lemma_4_1.compute_summands(cg, X_PORTS, NUMS_PORTS)
    O_NODES = []
    for summand in O:
        O_NODES.append(cg.generate_output_nodes_from_ports(summand, label="OUT"))
    return X, NUMS, O_NODES


def setup_lemma_4_1_compute_y(cg: CircuitGraph, bit_len=4):
    X = cg.add_input_nodes(bit_len, label="X")
    X_PORTS = cg.get_input_nodes_ports(X)
    M = cg.add_input_nodes(bit_len, label="M")
    M_PORTS = cg.get_input_nodes_ports(M)
    Y = lemma_4_1.compute_y(cg, X_PORTS, M_PORTS)
    Y_NODES = cg.generate_output_nodes_from_ports(Y, label="OUTPUT")
    return X, M, Y_NODES


def setup_lemma_4_1_compute_diffs(cg: CircuitGraph, bit_len=4):
    Y = cg.add_input_nodes(bit_len, label="Y")
    Y_PORTS = cg.get_input_nodes_ports(Y)
    M = cg.add_input_nodes(bit_len, label="M")
    M_PORTS = cg.get_input_nodes_ports(M)
    n = bit_len
    D = lemma_4_1.compute_diffs(cg, Y_PORTS, M_PORTS, n)
    D_NODES = []
    for d in D:
        d_nodes = cg.generate_output_nodes_from_ports(d, label="OUTPUT")
        D_NODES.append(d_nodes)
    return Y, M, D_NODES


def setup_bus_multiplexer(cg, num_amount=4, bit_len=4):
    BUS = []
    BUS_PORTS = []
    for i in range(num_amount):
        input = [cg.add_node("input", f"BUS_{i}_{j}") for j in range(bit_len)]
        BUS.append(input)
        num_ports = []
        for nde in input:
            num_ports.append(nde.ports[0])
        BUS_PORTS.append(num_ports)
    selector = [
        cg.add_node("input", f"SELECTOR_{i}") for i in range(int(math.log2(num_amount)))
    ]
    O = bus_multiplexer(cg, BUS_PORTS, [s.ports[0] for s in selector])
    O_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(O)]
    return BUS, selector, O_NODES


def setup_tensor_multiplexer(cg: CircuitGraph, num_amount=4, bit_len=4):
    TENSOR = []
    TENSOR_PORTS = []
    for i in range(num_amount):
        column_nodes = []
        column_ports = []
        for j in range(num_amount):
            num = cg.add_input_nodes(bit_len, label=f"INPUT_{i}_{j}")
            num_ports = cg.get_input_nodes_ports(num)
            column_nodes.append(num)
            column_ports.append(num_ports)
        TENSOR.append(column_nodes)
        TENSOR_PORTS.append(column_ports)
    selector = cg.add_input_nodes(int(math.log2(num_amount)), label="SELECTOR")
    selector_ports = cg.get_input_nodes_ports(selector)
    BUS = tensor_multiplexer(cg, TENSOR_PORTS, selector_ports)
    BUS_NODES = []
    for num in BUS:
        num_nodes = cg.generate_output_nodes_from_ports(num)
        BUS_NODES.append(num_nodes)
    return TENSOR, selector, BUS_NODES


def setup_montgomery_ladder(cg, bit_len=4):
    B = [cg.add_node("input", f"B_{i}") for i in range(bit_len)]
    E = [cg.add_node("input", f"E_{i}") for i in range(bit_len)]
    M = [cg.add_node("input", f"M_{i}") for i in range(bit_len)]
    OUT = montgomery_ladder(
        cg, [b.ports[0] for b in B], [e.ports[0] for e in E], [m.ports[0] for m in M]
    )
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return B, E, M, OUT_NODES


def setup_modular_exponentiation(cg, bit_len=4):
    B = [cg.add_node("input", f"B_{i}") for i in range(bit_len)]
    E = [cg.add_node("input", f"E_{i}") for i in range(bit_len)]
    M = [cg.add_node("input", f"M_{i}") for i in range(bit_len)]
    OUT = modular_exponentiation(
        cg, [b.ports[0] for b in B], [e.ports[0] for e in E], [m.ports[0] for m in M]
    )
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return B, E, M, OUT_NODES


def setup_modulo_circuit(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [cg.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = modulo_circuit(cg, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_slow_modulo_circuit(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [cg.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = slow_modulo_circuit(cg, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_optimized_modulo_circuit(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [cg.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = modulo_circuit_optimized(cg, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_reciprocal_newton_raphson(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    OUT = reciprocal_newton_raphson(cg, [x.ports[0] for x in X], 4)
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, OUT_NODES


def setup_log2_estimate(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    OUT = log2_estimate(cg, [x.ports[0] for x in X])
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, OUT_NODES


def setup_or_tree_recursive(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    o = or_tree_recursive(cg, [x.ports[0] for x in X])
    o_node = cg.add_node("output", "OUT", inputs=[o])
    return X, o_node


def setup_or_tree_iterative(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    o = or_tree_iterative(cg, [x.ports[0] for x in X])
    o_node = cg.add_node("output", "OUT", inputs=[o])
    return X, o_node


def setup_and_tree_iterative(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    o = and_tree_iterative(cg, [x.ports[0] for x in X])
    o_node = cg.add_node("output", "OUT", inputs=[o])
    return X, o_node


def setup_n_left_shift(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [cg.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = n_left_shift(cg, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_n_right_shift(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [cg.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = n_right_shift(cg, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_one_left_shift(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    OUT = one_left_shift(cg, [x.ports[0] for x in X])
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, OUT_NODES


def setup_one_right_shift(cg, bit_len=4):
    X = [cg.add_node("input", f"X_{i}") for i in range(bit_len)]
    OUT = one_right_shift(cg, [x.ports[0] for x in X])
    OUT_NODES = [
        cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, OUT_NODES


"""def setup_small_mod_lemma_4_1(cg, bit_len=4):
    X = [cg.add_node("input", f"X{i}") for i in range(4)]
    M = [cg.add_node("input", f"M{i}") for i in range(4)]
    # in_node = cg.add_node("input", f"IN")
    # build M from int
    # int_m = 3
    # bin_list_m = utils.int2binlist(int_m, 4)
    # print("bin_list_m shape: ", len(bin_list_m))
    # M = [constant_one(cg, in_node.ports[0]) if bit else constant_zero(cg, in_node.ports[0]) for bit in bin_list_m]
    outputs = small_mod_lemma_4_1(cg, [x.ports[0] for x in X], M, 2)
    out_nodes = []
    for out in outputs:
        out_node = cg.add_node("output", "REMAINDER", inputs=[out])
        out_nodes.append(out_node)
    return X, out_nodes"""


def setup_adder_tree_recursive(cg, bit_len=4):
    ports = []
    for k in range(4):
        X = [cg.add_node("input", f"X{i}") for i in range(bit_len)]
        ports.append([x.ports[0] for x in X])
    cin = cg.add_node("input", "CIN")
    outputs, carry = adder_tree_recursive(cg, ports, cin.ports[0])
    for out in outputs:
        cg.add_node("output", "SUM", inputs=[out])
    cg.add_node("output", "CARRY", inputs=[carry])
    return


def setup_multiplexer(cg, bit_len=4):
    X = [cg.add_node("input", f"X{i}") for i in range(bit_len)]
    S = [cg.add_node("input", f"S{i}") for i in range(int(math.log2(bit_len)))]
    mux = multiplexer(cg, [x.ports[0] for x in X], [s.ports[0] for s in S])
    out = cg.add_node("output", "MUX OUT", inputs=[mux])
    return


def setup_one_bit_comparator(cg, bit_len=4):
    less, equals, greater = one_bit_comparator(
        cg, cg.add_node("input", "x").ports[0], cg.add_node("input", "y").ports[0]
    )
    less_node = cg.add_node("output", "LESS", inputs=[less])
    equals_node = cg.add_node("output", "EQUALS", inputs=[equals])
    greater_node = cg.add_node("output", "GREATER", inputs=[greater])
    return


def setup_n_bit_comparator(cg, bit_len=4):
    A = [cg.add_node("input", f"A{i}") for i in range(bit_len)]
    B = [cg.add_node("input", f"B{i}") for i in range(bit_len)]
    less, equals, greater = n_bit_comparator(
        cg, [a.ports[0] for a in A], [b.ports[0] for b in B]
    )
    L = cg.add_node("output", "LESS", inputs=[less])
    E = cg.add_node("output", "EQUALS", inputs=[equals])
    G = cg.add_node("output", "GREATER", inputs=[greater])
    return A, B, L, E, G


def setup_n_bit_equality(cg: CircuitGraph, bit_len=4):
    n = bit_len
    A = cg.add_input_nodes(n)
    B = cg.add_input_nodes(n)
    A_PORTS = cg.get_input_nodes_ports(A)
    B_PORTS = cg.get_input_nodes_ports(B)
    equals = n_bit_equality(cg, A_PORTS, B_PORTS)
    equals_node = cg.generate_output_node_from_port(equals)
    return A, B, equals_node


def setup_xnor_gate(cg, bit_len=4):
    X = [cg.add_node("input", f"X{i}") for i in range(2)]
    out = xnor_gate(cg, X[0].ports[0], X[1].ports[0])
    xnor_output = cg.add_node("output", "XNOR OUTPUT", inputs=[out])
    return


def setup_constant_zero(cg, bit_len=4):
    X = cg.add_node("input", "X")
    zero_port = constant_zero(cg, X.ports[0])
    zero_node = cg.add_node("output", "CONSTANT ZERO", inputs=[zero_port])
    return


def setup_constant_one(cg, bit_len=4):
    X = cg.add_node("input", "X")
    one_port = constant_one(cg, X.ports[0])
    one_node = cg.add_node("output", "CONSTANT ONE", inputs=[one_port])
    return


def setup_half_adder(cg, bit_len=4):
    A = cg.add_node("input", "A")
    B = cg.add_node("input", "B")
    sum_port, carry_port = half_adder(cg, A.ports[0], B.ports[0])
    sum_node = cg.add_node("output", "sum")
    carry_node = cg.add_node("output", "carry")
    cg.add_edge(sum_port, sum_node.ports[0])
    cg.add_edge(carry_port, carry_node.ports[0])

    return A, B, sum_node, carry_node


def setup_full_adder(cg, bit_len=4):
    A = cg.add_node("input", "A")
    B = cg.add_node("input", "B")
    cin = cg.add_node("input", "Cin")
    sum_port, carry_port = full_adder(cg, A.ports[0], B.ports[0], cin.ports[0])
    sum_node = cg.add_node("output", "sum")
    carry_node = cg.add_node("output", "carry")
    cg.add_edge(sum_port, sum_node.ports[0])
    cg.add_edge(carry_port, carry_node.ports[0])
    return A, B, cin, sum_node, carry_node


def setup_ripple_carry_adder(cg: CircuitGraph, bit_len=4):
    A = cg.add_input_nodes(bit_len, label="A")
    B = cg.add_input_nodes(bit_len, label="B")
    # A = [cg.add_node("input", f"A{i}") for i in range(bit_len)]
    # B = [cg.add_node("input", f"B{i}") for i in range(bit_len)]
    cin = cg.add_node("input", "Cin")
    sum_ports, carry_port = ripple_carry_adder(
        cg, [a.ports[0] for a in A], [b.ports[0] for b in B], cin.ports[0]
    )
    for i, s_port in enumerate(sum_ports):
        sum_node = cg.add_node("output", f"sum_{i}")
        cg.add_edge(s_port, sum_node.ports[0])
    carry_node = cg.add_node("output", "carry")
    cg.add_edge(carry_port, carry_node.ports[0])
    return
    A = [cg.add_input(f"A{i}") for i in range(4)]
    B = [cg.add_input(f"B{i}") for i in range(4)]
    Cin = cg.add_input("Cin")
    sum_outputs, carry_out = ripple_carry_adder(cg, A, B, Cin)
    for sum in sum_outputs:
        cg.add_output(sum, "sum")
    cg.add_output(carry_out, "carry")


def setup_carry_look_ahead_adder(cg: CircuitGraph, bit_len=4):
    A = cg.add_input_nodes(bit_len, label="A")
    B = cg.add_input_nodes(bit_len, label="B")
    cin = cg.add_node("input", "Cin")
    sum_outputs, carry_out = carry_look_ahead_adder(
        cg,
        cg.get_input_nodes_ports(A),
        cg.get_input_nodes_ports(B),
        cg.get_input_node_port(cin),
    )
    sum_nodes = []
    for sum in sum_outputs:
        sum_nodes.append(cg.add_node("output", "SUM", inputs=[sum]))
    carry_node = cg.add_node("output", "CARRY", inputs=[carry_out])
    return A, B, cin, sum_nodes, carry_node


def setup_wallace_tree_multiplier(cg, bit_len=4):
    A = [cg.add_node("input", f"A{i}") for i in range(bit_len)]
    B = [cg.add_node("input", f"B{i}") for i in range(bit_len)]
    outputs = wallace_tree_multiplier(
        cg, [a.ports[0] for a in A], [b.ports[0] for b in B]
    )
    output_nodes = []
    for out in outputs:
        output_nodes.append(cg.add_node("output", "PRODUCT", inputs=[out]))
    return A, B, output_nodes


def setup_faulty_wallace_tree_multiplier(cg, bit_len=4):
    A = [cg.add_node("input", f"A{i}") for i in range(bit_len)]
    B = [cg.add_node("input", f"B{i}") for i in range(bit_len)]
    outputs = faulty_wallace_tree_multiplier(
        cg, [a.ports[0] for a in A], [b.ports[0] for b in B]
    )
    output_nodes = []
    for out in outputs:
        output_nodes.append(cg.add_node("output", "PRODUCT", inputs=[out]))
    return A, B, output_nodes


def setup_precompute_aim(cg, bit_len=4):
    input_node = cg.add_node("input", "INPUT")
    input_node_port = input_node.ports[0]
    zero_port = constant_zero(cg, input_node_port)
    one_port = constant_one(cg, input_node_port)
    aims = lemma_4_1.precompute_aim(cg, zero_port, one_port, bit_len, bit_len)
    output_nodes = []
    for m, ais in enumerate(aims):
        m = m + 1
        am_entry = []
        for i, ports in enumerate(ais):
            aim_entry = []
            for port in ports:
                out_node = cg.add_node("output", f"OUT_{m}_{i}", inputs=[port])
                aim_entry.append(out_node)
            am_entry.append(aim_entry)
        output_nodes.append(am_entry)
    return output_nodes


def setup_conditional_zeroing(cg, bit_len=4):
    X = [cg.add_node("input", f"A{i}") for i in range(bit_len)]
    C = cg.add_node("input", f"COND")
    output = conditional_zeroing(cg, [x.ports[0] for x in X], C.ports[0])
    output_nodes = []
    for i, out in enumerate(output):
        output_node = cg.add_node("output", f"OUT_{i}", inputs=[out])
        output_nodes.append(output_node)
    return X, C, output_nodes

def setup_subtract(cg: CircuitGraph, bit_len=4):
    n = bit_len
    A = cg.add_input_nodes(n, label="A")
    B = cg.add_input_nodes(n, label="B")
    diff = subtract(cg, cg.get_input_nodes_ports(A), cg.get_input_nodes_ports(B))
    diff_nodes = cg.generate_output_nodes_from_ports(diff, label="DIFF")
    return A, B, diff_nodes

def setup_conditional_subtract(cg: CircuitGraph, bit_len=4):
    A = [cg.add_node("input", f"A{i}") for i in range(bit_len)]
    B = [cg.add_node("input", f"B{i}") for i in range(bit_len)]
    C = cg.add_node("input", f"COND")
    output = conditional_subtract(
        cg, [a.ports[0] for a in A], [b.ports[0] for b in B], C.ports[0]
    )
    O = []
    for i, out in enumerate(output):
        output_node = cg.add_node("output", f"OUT_{i}", inputs=[out])
        O.append(output_node)
    return A, B, C, O


def setup_next_power_of_two(cg, bit_len=4):
    X = [cg.add_node("input", f"X{i}") for i in range(bit_len)]
    output = next_power_of_two(cg, [x.ports[0] for x in X])
    O = []
    for i, out in enumerate(output):
        output_node = cg.add_node("output", f"OUT_{i}", inputs=[out])
        O.append(output_node)
    return X, O


# def setup_four_bit_wallace_tree_multiplier(cg):
#    A = [cg.add_input(f"A{i}") for i in range(4)]
#    B = [cg.add_input(f"B{i}") for i in range(4)]
#    Cin = cg.add_input("cin")
#    sum_outputs, carry_out = four_bit_wallace_tree_multiplier(cg, A, B, Cin)
#    for i, sum in enumerate(sum_outputs):
#        cg.add_output(sum, f"p_{i}")
#    cg.add_output(carry_out, f"p_{len(sum_outputs)}")

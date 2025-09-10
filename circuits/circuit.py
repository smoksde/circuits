import math

from core.interface import Interface, GraphInterface, DepthInterface

from .shifters import *
from .constants import *
from .adders import *
from .multipliers import *
from .comparators import *
from .multiplexers import *
from .subtractors import *
from .modular import *
from .reference.montgomery_ladder import montgomery_ladder
from .reference.square_and_multiply import square_and_multiply
from .manipulators import conditional_zeroing, max_tree_iterative

from .beame import lemma_4_1
from .beame import lemma_5_1
from .beame import theorem_4_2
from .beame import theorem_5_2
from .beame import theorem_5_3


def binary_list_to_int(binary_list):
    return sum(bit * (2**i) for i, bit in enumerate(binary_list))


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


CIRCUIT_FUNCTIONS = {
    "xnor_gate": lambda circuit, bit_len: setup_xnor_gate(circuit, bit_len=bit_len),
    "one_bit_comparator": lambda circuit, bit_len: setup_one_bit_comparator(
        circuit, bit_len=bit_len
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
    "square_and_multiply": lambda cg, bit_len: setup_square_and_multiply(
        cg, bit_len=bit_len
    ),
    "montgomery_ladder": lambda cg, bit_len: setup_montgomery_ladder(
        cg, bit_len=bit_len
    ),
}


def setup_theorem_4_2_step_1(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [circuit.add_input_nodes(n, "INPUT") for _ in range(n)]
    P_NODES = circuit.add_input_nodes(n, "INPUT")
    PEXPL_NODES = circuit.add_input_nodes(n, "INPUT")

    X_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    P_PORTS = circuit.get_input_nodes_ports(P_NODES)
    PEXPL_PORTS = circuit.get_input_nodes_ports(PEXPL_NODES)

    EXPONENTS_PORTS = theorem_4_2.step_1(circuit, X_LIST_PORTS, P_PORTS, PEXPL_PORTS)

    EXPONENTS_NODES = [
        circuit.generate_output_nodes_from_ports(ports) for ports in EXPONENTS_PORTS
    ]
    return X_LIST_NODES, P_NODES, PEXPL_NODES, EXPONENTS_NODES


def setup_theorem_4_2_step_1_with_lemma_4_1(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [circuit.add_input_nodes(n, "INPUT") for _ in range(n)]
    P_NODES = circuit.add_input_nodes(n, "INPUT")
    PEXPL_NODES = circuit.add_input_nodes(n, "INPUT")

    X_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    P_PORTS = circuit.get_input_nodes_ports(P_NODES)
    PEXPL_PORTS = circuit.get_input_nodes_ports(PEXPL_NODES)

    EXPONENTS_PORTS = theorem_4_2.step_1_with_lemma_4_1(
        circuit, X_LIST_PORTS, P_PORTS, PEXPL_PORTS
    )

    EXPONENTS_NODES = [
        circuit.generate_output_nodes_from_ports(ports) for ports in EXPONENTS_PORTS
    ]
    return X_LIST_NODES, P_NODES, PEXPL_NODES, EXPONENTS_NODES


def setup_theorem_4_2_step_1_with_precompute(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [circuit.add_input_nodes(n, "INPUT") for _ in range(n)]
    P_NODES = circuit.add_input_nodes(n, "INPUT")
    PEXPL_NODES = circuit.add_input_nodes(n, "INPUT")

    X_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    P_PORTS = circuit.get_input_nodes_ports(P_NODES)
    PEXPL_PORTS = circuit.get_input_nodes_ports(PEXPL_NODES)

    EXPONENTS_PORTS = theorem_4_2.step_1_with_precompute(
        circuit, X_LIST_PORTS, P_PORTS, PEXPL_PORTS
    )

    EXPONENTS_NODES = [
        circuit.generate_output_nodes_from_ports(ports) for ports in EXPONENTS_PORTS
    ]
    return X_LIST_NODES, P_NODES, PEXPL_NODES, EXPONENTS_NODES


def setup_theorem_4_2_step_2(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    X_LIST_NODES = [circuit.add_input_nodes(n, "INPUT") for _ in range(n)]
    P_NODES = circuit.add_input_nodes(n, "INPUT")
    J_LIST_NODES = [circuit.add_input_nodes(n, "INPUT") for _ in range(n)]

    X_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES]
    P_PORTS = circuit.get_input_nodes_ports(P_NODES)
    J_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in J_LIST_NODES]

    Y_LIST_PORTS = theorem_4_2.step_2(circuit, X_LIST_PORTS, P_PORTS, J_LIST_PORTS)

    Y_LIST_NODES = [
        circuit.generate_output_nodes_from_ports(ports) for ports in Y_LIST_PORTS
    ]
    return X_LIST_NODES, P_NODES, J_LIST_NODES, Y_LIST_NODES


def setup_theorem_4_2_compute_sum(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    J_LIST_NODES = [circuit.add_input_nodes(n, "INPUT") for _ in range(n)]
    J_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in J_LIST_NODES]
    J_PORTS = theorem_4_2.compute_sum(circuit, J_LIST_PORTS)
    J_NODES = circuit.generate_output_nodes_from_ports(J_PORTS)
    return J_LIST_NODES, J_NODES


def setup_theorem_4_2_step_4(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    P_NODES = circuit.add_input_nodes(n, "INPUT")
    PEXPL_NODES = circuit.add_input_nodes(n, "INPUT")
    P_PORTS = circuit.get_input_nodes_ports(P_NODES)
    PEXPL_PORTS = circuit.get_input_nodes_ports(PEXPL_NODES)
    FLAG = theorem_4_2.step_4(circuit, P_PORTS, PEXPL_PORTS)
    FLAG_NODE = circuit.generate_output_node_from_port(FLAG)
    return P_NODES, PEXPL_NODES, FLAG_NODE


def setup_theorem_4_2_A_step_5(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    Y_LIST_NODES = [circuit.add_input_nodes(n, "INPUT") for _ in range(n)]
    PEXPL_NODES = circuit.add_input_nodes(n, "INPUT")
    Y_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in Y_LIST_NODES]
    PEXPL_PORTS = circuit.get_input_nodes_ports(PEXPL_NODES)
    A_LIST_PORTS = theorem_4_2.A_step_5(circuit, Y_LIST_PORTS, PEXPL_PORTS)
    A_LIST_NODES = [
        circuit.generate_output_nodes_from_ports(ports) for ports in A_LIST_PORTS
    ]
    return Y_LIST_NODES, PEXPL_NODES, A_LIST_NODES


def setup_theorem_4_2_A_step_7(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    A_NODES = circuit.add_input_nodes(n, "INPUT")
    PEXPL_NODES = circuit.add_input_nodes(n, "INPUT")
    A_PORTS = circuit.get_input_nodes_ports(A_NODES)
    PEXPL_PORTS = circuit.get_input_nodes_ports(PEXPL_NODES)
    A_HAT_PORTS = theorem_4_2.A_step_7(circuit, A_PORTS, PEXPL_PORTS)
    A_HAT_NODES = circuit.generate_output_nodes_from_ports(A_HAT_PORTS)
    return A_NODES, PEXPL_NODES, A_HAT_NODES


def setup_theorem_4_2_A_step_8(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    A_HAT_NODES = circuit.add_input_nodes(n)
    PEXPL_NODES = circuit.add_input_nodes(n)

    A_HAT_PORTS = circuit.get_input_nodes_ports(A_HAT_NODES)
    PEXPL_PORTS = circuit.get_input_nodes_ports(PEXPL_NODES)
    Y_PRODUCT_PORTS = theorem_4_2.A_step_8(
        circuit,
        A_HAT_PORTS,
        PEXPL_PORTS,
    )
    Y_PRODUCT_NODES = circuit.generate_output_nodes_from_ports(Y_PRODUCT_PORTS)
    return A_HAT_NODES, PEXPL_NODES, Y_PRODUCT_NODES


def setup_theorem_4_2_B_step_5(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    Y_LIST_NODES = [circuit.add_input_nodes(n) for _ in range(n)]
    L_NODES = circuit.add_input_nodes(n)
    Y_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in Y_LIST_NODES]
    L_PORTS = circuit.get_input_nodes_ports(L_NODES)
    A_LIST_PORTS, B_LIST_PORTS = theorem_4_2.B_step_5(circuit, Y_LIST_PORTS, L_PORTS)
    A_LIST_NODES = [
        circuit.generate_output_nodes_from_ports(ports) for ports in A_LIST_PORTS
    ]
    B_LIST_NODES = [
        circuit.generate_output_nodes_from_ports(ports) for ports in B_LIST_PORTS
    ]
    return Y_LIST_NODES, L_NODES, A_LIST_NODES, B_LIST_NODES


def setup_theorem_4_2_B_step_7(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    A_NODES = circuit.add_input_nodes(n)
    B_NODES = circuit.add_input_nodes(n)
    L_NODES = circuit.add_input_nodes(n)

    A_PORTS = circuit.get_input_nodes_ports(A_NODES)
    B_PORTS = circuit.get_input_nodes_ports(B_NODES)
    L_PORTS = circuit.get_input_nodes_ports(L_NODES)

    A_HAT_PORTS, B_HAT_PORTS = theorem_4_2.B_step_7(circuit, A_PORTS, B_PORTS, L_PORTS)

    A_HAT_NODES = circuit.generate_output_nodes_from_ports(A_HAT_PORTS)
    B_HAT_NODES = circuit.generate_output_nodes_from_ports(B_HAT_PORTS)

    return A_NODES, B_NODES, L_NODES, A_HAT_NODES, B_HAT_NODES


def setup_theorem_4_2_step_9(circuit: CircuitGraph, bit_len=4):
    n = bit_len

    P_NODES = circuit.add_input_nodes(n)
    J_NODES = circuit.add_input_nodes(n)
    PEXPL_NODES = circuit.add_input_nodes(n)
    Y_PRODUCT_NODES = circuit.add_input_nodes(n)

    P_PORTS = circuit.get_input_nodes_ports(P_NODES)
    J_PORTS = circuit.get_input_nodes_ports(J_NODES)
    PEXPL_PORTS = circuit.get_input_nodes_ports(PEXPL_NODES)
    Y_PRODUCT_PORTS = circuit.get_input_nodes_ports(Y_PRODUCT_NODES)

    RESULT_PORTS = theorem_4_2.step_9(
        circuit, P_PORTS, J_PORTS, PEXPL_PORTS, Y_PRODUCT_PORTS
    )
    RESULT_NODES = circuit.generate_output_nodes_from_ports(RESULT_PORTS)

    return P_NODES, J_NODES, PEXPL_NODES, Y_PRODUCT_NODES, RESULT_NODES


def setup_theorem_4_2_not_modified(interface: Interface, bit_len=4):
    n = bit_len
    if isinstance(interface, DepthInterface):
        X_LIST = [[0] * n for _ in range(n)]
        PEXPL = [0] * n
        RESULT_DEPTHS = theorem_4_2.theorem_4_2_not_modified(interface, X_LIST, PEXPL)
        max_depth = max(RESULT_DEPTHS)
        return max_depth
    elif isinstance(interface, GraphInterface):
        X_LIST_NODES = [interface.add_input_nodes(n) for _ in range(n)]
        PEXPL_NODES = interface.add_input_nodes(n)
        X_LIST_PORTS = [
            interface.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES
        ]
        PEXPL_PORTS = interface.get_input_nodes_ports(PEXPL_NODES)
        RESULT_PORTS = theorem_4_2.theorem_4_2(interface, X_LIST_PORTS, PEXPL_PORTS)
        RESULT_NODES = interface.generate_output_nodes_from_ports(RESULT_PORTS)
        return X_LIST_NODES, PEXPL_NODES, RESULT_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_theorem_4_2(interface: Interface, bit_len=4):
    n = bit_len
    if isinstance(interface, DepthInterface):
        X_LIST = [[0] * n for _ in range(n)]
        PEXPL = [0] * n
        RESULT_DEPTHS = theorem_4_2.theorem_4_2_not_modified(interface, X_LIST, PEXPL)
        max_depth = max(RESULT_DEPTHS)
        return max_depth
    elif isinstance(interface, GraphInterface):
        X_LIST_NODES = [interface.add_input_nodes(n) for _ in range(n)]
        PEXPL_NODES = interface.add_input_nodes(n)
        X_LIST_PORTS = [
            interface.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES
        ]
        PEXPL_PORTS = interface.get_input_nodes_ports(PEXPL_NODES)
        RESULT_PORTS = theorem_4_2.theorem_4_2(interface, X_LIST_PORTS, PEXPL_PORTS)
        RESULT_NODES = interface.generate_output_nodes_from_ports(RESULT_PORTS)
        return X_LIST_NODES, PEXPL_NODES, RESULT_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_theorem_4_2_precompute_largest_powers(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = circuit.add_input_nodes(1)[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    MATRIX_PORTS = theorem_4_2.precompute_largest_powers(
        circuit, zero_port, one_port, n
    )
    MATRIX_NODES = []
    for row in MATRIX_PORTS:
        MATRIX_NODES.append(
            [circuit.generate_output_nodes_from_ports(ports) for ports in row]
        )
    return MATRIX_NODES


def setup_theorem_4_2_precompute_lookup_tables_B(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = circuit.add_input_nodes(1, "INPUT")[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    TABLE_ZERO, TABLE_ONE = theorem_4_2.precompute_lookup_tables_B(
        circuit, zero_port, one_port, n
    )
    TABLE_ZERO_NODES = []
    for row in TABLE_ZERO:
        nodes = [
            circuit.generate_output_nodes_from_ports(entry, label="OUTPUT")
            for entry in row
        ]
        TABLE_ZERO_NODES.append(nodes)
    TABLE_ONE_NODES = []
    for row in TABLE_ONE:
        nodes = [
            circuit.generate_output_nodes_from_ports(entry, label="OUTPUT")
            for entry in row
        ]
        TABLE_ONE_NODES.append(nodes)
    return TABLE_ZERO_NODES, TABLE_ONE_NODES


def setup_theorem_4_2_precompute_lookup_generator_powers(
    circuit: CircuitGraph, bit_len=4
):
    n = bit_len
    input_node = circuit.add_input_nodes(1, "INPUT")[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    TABLE = theorem_4_2.precompute_lookup_generator_powers(
        circuit, zero_port, one_port, n
    )
    TABLE_NODES = []
    for row in TABLE:
        nodes = [
            circuit.generate_output_nodes_from_ports(entry, label="OUTPUT")
            for entry in row
        ]
        TABLE_NODES.append(nodes)
    return TABLE_NODES


def setup_theorem_4_2_precompute_lookup_division(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = circuit.add_input_nodes(1, "INPUT")[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    TABLE = theorem_4_2.precompute_lookup_division(circuit, zero_port, one_port, n)
    TABLE_NODES = []
    for row in TABLE:
        nodes = [
            circuit.generate_output_nodes_from_ports(entry, label="OUTPUT")
            for entry in row
        ]
        TABLE_NODES.append(nodes)
    return TABLE_NODES


def setup_theorem_4_2_precompute_lookup_powers(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = circuit.add_input_nodes(1, "INPUT")[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    O = theorem_4_2.precompute_lookup_powers(circuit, zero_port, one_port, n)
    O_NODES = []
    for o in O:
        powers_of_p_nodes = []
        for power in o:
            power_nodes = circuit.generate_output_nodes_from_ports(
                power, label="OUTPUT"
            )
            powers_of_p_nodes.append(power_nodes)
        O_NODES.append(powers_of_p_nodes)
    return O_NODES


def setup_theorem_4_2_precompute_lookup_is_prime_power(
    circuit: CircuitGraph, bit_len=4
):
    n = bit_len
    input_node = circuit.add_input_nodes(1, "INPUT")[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    O = theorem_4_2.precompute_lookup_is_prime_power(circuit, zero_port, one_port, n)
    O_NODES = circuit.generate_output_nodes_from_ports(O, label="OUTPUT")
    return O_NODES


def setup_theorem_4_2_precompute_lookup_p_l(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = circuit.add_input_nodes(1, "INPUT")[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    P_TABLE, L_TABLE = theorem_4_2.precompute_lookup_p_l(
        circuit, zero_port, one_port, n
    )
    P_TABLE_NODES = []
    L_TABLE_NODES = []
    for p, l in zip(P_TABLE, L_TABLE):
        p_nodes = circuit.generate_output_nodes_from_ports(p)
        l_nodes = circuit.generate_output_nodes_from_ports(l)
        P_TABLE_NODES.append(p_nodes)
        L_TABLE_NODES.append(l_nodes)
    return P_TABLE_NODES, L_TABLE_NODES


def setup_theorem_4_2_precompute_lookup_pexpl_minus_pexpl_minus_one(
    circuit: CircuitGraph, bit_len=4
):
    n = bit_len
    input_node = circuit.add_input_nodes(1, "INPUT")[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    TABLE = theorem_4_2.precompute_lookup_pexpl_minus_pexpl_minus_one(
        circuit, zero_port, one_port, n
    )
    TABLE_NODES = []
    for ports in TABLE:
        nodes = circuit.generate_output_nodes_from_ports(ports)
        TABLE_NODES.append(nodes)
    return TABLE_NODES


def setup_theorem_5_3_precompute_good_modulus_sequence(
    circuit: CircuitGraph, bit_len=4
):
    n = bit_len
    input_node = circuit.add_input_nodes(1)[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    PRIMES_PORTS, PRIMES_PRODUCT_PORTS = theorem_5_3.precompute_good_modulus_sequence(
        circuit, zero_port, one_port, n
    )
    PRIMES_NODES = []
    for ports in PRIMES_PORTS:
        nodes = circuit.generate_output_nodes_from_ports(ports)
        PRIMES_NODES.append(nodes)
    PRIMES_PRODUCT_NODES = circuit.generate_output_nodes_from_ports(
        PRIMES_PRODUCT_PORTS
    )
    return PRIMES_NODES, PRIMES_PRODUCT_NODES


def setup_lemma_5_1_precompute_u_list(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = circuit.add_input_nodes(1)[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    U_LIST_PORTS = lemma_5_1.precompute_u_list(circuit, zero_port, one_port, n)
    U_LIST_NODES = [
        circuit.generate_output_nodes_from_ports(ports) for ports in U_LIST_PORTS
    ]
    return U_LIST_NODES


def setup_lemma_5_1_step_5(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    s = n
    X_MOD_C_I_LIST_NODES = [circuit.add_input_nodes(n) for _ in range(s)]
    U_LIST_NODES = [circuit.add_input_nodes(n) for _ in range(n)]
    X_MOD_C_I_LIST_PORTS = [
        circuit.get_input_nodes_ports(nodes) for nodes in X_MOD_C_I_LIST_NODES
    ]
    U_LIST_PORTS = [circuit.get_input_nodes_ports(nodes) for nodes in U_LIST_NODES]
    Y_PORTS = lemma_5_1.step_5(circuit, X_MOD_C_I_LIST_PORTS, U_LIST_PORTS)
    Y_NODES = circuit.generate_output_nodes_from_ports(Y_PORTS)
    return X_MOD_C_I_LIST_NODES, U_LIST_NODES, Y_NODES


def setup_lemma_5_1_step_6_and_7(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = circuit.add_input_nodes(1)[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    Y_NODES = circuit.add_input_nodes(n)
    C_NODES = circuit.add_input_nodes(n)
    Y_PORTS = circuit.get_input_nodes_ports(Y_NODES)
    C_PORTS = circuit.get_input_nodes_ports(C_NODES)
    RESULT_PORTS = lemma_5_1.step_6_and_7(
        circuit, Y_PORTS, C_PORTS, zero_port, one_port
    )
    RESULT_NODES = circuit.generate_output_nodes_from_ports(RESULT_PORTS)
    return Y_NODES, C_NODES, RESULT_NODES


def setup_lemma_5_1(interface: CircuitGraph, bit_len=4):
    n = bit_len
    s = n * n  # n * n
    if isinstance(interface, DepthInterface):
        X_MOD_C_I_LIST = [[0] * s for _ in range(n)]
        C = [0] * s
        RESULT_DEPTHS = lemma_5_1.lemma_5_1(interface, X_MOD_C_I_LIST, C)
        max_depth = max(RESULT_DEPTHS)
        return max_depth
    elif isinstance(interface, GraphInterface):
        X_MOD_C_I_LIST_NODES = [interface.add_input_nodes(s) for _ in range(n)]
        C_NODES = interface.add_input_nodes(s)
        X_MOD_C_I_LIST_PORTS = [
            interface.get_input_nodes_ports(nodes) for nodes in X_MOD_C_I_LIST_NODES
        ]
        C_PORTS = interface.get_input_nodes_ports(C_NODES)
        RESULT_PORTS = lemma_5_1.lemma_5_1(interface, X_MOD_C_I_LIST_PORTS, C_PORTS)
        RESULT_NODES = interface.generate_output_nodes_from_ports(RESULT_PORTS)
        return X_MOD_C_I_LIST_NODES, C_NODES, RESULT_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_theorem_5_2_step_3(interface: Interface, bit_len=4):
    n = bit_len
    s = n * n
    if isinstance(interface, DepthInterface):
        X_LIST = [[0] * n for _ in range(n)]
        C_LIST = [[0] * n for _ in range(s)]
        MATRIX_DEPTHS = theorem_5_2.step_3(interface, X_LIST, C_LIST)
        max_depth = max(x for matrix in MATRIX_DEPTHS for row in matrix for x in row)
        return max_depth
    elif isinstance(interface, GraphInterface):
        X_LIST_NODES = [interface.add_input_nodes(n) for _ in range(n)]
        C_LIST_NODES = [interface.add_input_nodes(n) for _ in range(s)]
        X_LIST_PORTS = [
            interface.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES
        ]
        C_LIST_PORTS = [
            interface.get_input_nodes_ports(nodes) for nodes in C_LIST_NODES
        ]
        MATRIX_PORTS = theorem_5_2.step_3(interface, X_LIST_PORTS, C_LIST_PORTS)
        MATRIX_NODES = []
        for row in MATRIX_PORTS:
            nodes = [interface.generate_output_nodes_from_ports(ports) for ports in row]
            MATRIX_NODES.append(nodes)
        return X_LIST_NODES, C_LIST_NODES, MATRIX_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_theorem_5_2_step_4(interface: Interface, bit_len=4):
    n = bit_len
    s = n  # n * n
    if isinstance(interface, DepthInterface):
        B_J_I_MATRIX = [[[0] * n for _ in range(n)] for _ in range(s)]
        C_LIST = [[0] * n for _ in range(s)]
        B_J_LIST_DEPTHS = theorem_5_2.step_4(interface, B_J_I_MATRIX, C_LIST)
        max_depth = max([v for sub in B_J_LIST_DEPTHS for v in sub])
        return max_depth
    elif isinstance(interface, GraphInterface):
        B_J_I_MATRIX_NODES = []
        B_J_I_MATRIX_PORTS = []
        for _ in range(s):
            nodes_list = [interface.add_input_nodes(n) for _ in range(n)]
            B_J_I_MATRIX_NODES.append(nodes_list)
            B_J_I_MATRIX_PORTS.append(
                [interface.get_input_nodes_ports(nodes) for nodes in nodes_list]
            )
        C_LIST_NODES = [interface.add_input_nodes(n) for _ in range(s)]
        C_LIST_PORTS = [
            interface.get_input_nodes_ports(nodes) for nodes in C_LIST_NODES
        ]
        B_J_LIST_PORTS = theorem_5_2.step_4(interface, B_J_I_MATRIX_PORTS, C_LIST_PORTS)
        B_J_LIST_NODES = [
            interface.generate_output_nodes_from_ports(ports)
            for ports in B_J_LIST_PORTS
        ]
        return B_J_I_MATRIX_NODES, C_LIST_NODES, B_J_LIST_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_theorem_5_2_step_5(interface: Interface, bit_len=4):
    n = bit_len
    if isinstance(interface, DepthInterface):
        X_MOD_C_I_LIST = [[0] * (n * n) for _ in range(n)]
        C = [0] * (n * n)
        RESULT_DEPTHS = theorem_5_2.step_5(interface, X_MOD_C_I_LIST, C)
        max_depth = max(RESULT_DEPTHS)
        return max_depth
    elif isinstance(interface, GraphInterface):
        X_MOD_C_I_LIST_NODES = [interface.add_input_nodes(n * n) for _ in range(n)]
        C_NODES = interface.add_input_nodes(n * n)
        X_MOD_C_I_LIST_PORTS = [
            interface.get_input_nodes_ports(nodes) for nodes in X_MOD_C_I_LIST_NODES
        ]
        C_PORTS = interface.get_input_nodes_ports(C_NODES)
        RESULT_PORTS = theorem_5_2.step_5(interface, X_MOD_C_I_LIST_PORTS, C_PORTS)
        RESULT_NODES = interface.generate_output_nodes_from_ports(RESULT_PORTS)
        return X_MOD_C_I_LIST_NODES, C_NODES, RESULT_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_theorem_5_2(interface: Interface, bit_len=4):
    n = bit_len
    if isinstance(interface, DepthInterface):
        X_LIST = [[0] * n for _ in range(n)]
        RESULT_DEPTHS = theorem_5_2.theorem_5_2(interface, X_LIST)
        max_depth = max(RESULT_DEPTHS)
        return max_depth
    elif isinstance(interface, GraphInterface):
        X_LIST_NODES = [interface.add_input_nodes(n) for _ in range(n)]
        X_LIST_PORTS = [
            interface.get_input_nodes_ports(nodes) for nodes in X_LIST_NODES
        ]
        RESULT_PORTS = theorem_5_2.theorem_5_2(interface, X_LIST_PORTS)
        RESULT_NODES = interface.generate_output_nodes_from_ports(RESULT_PORTS)
        return X_LIST_NODES, RESULT_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_max_tree_iterative(circuit: CircuitGraph, num_amount=4, bit_len=4):
    VALUES_NODES = []
    VALUES_PORTS = []

    for i in range(num_amount):
        X = circuit.add_input_nodes(bit_len, label=f"X_{i}")
        X_PORTS = circuit.get_input_nodes_ports(X)
        VALUES_NODES.append(X)
        VALUES_PORTS.append(X_PORTS)
    MAX_PORTS = max_tree_iterative(circuit, VALUES_PORTS)
    MAX_NODES = circuit.generate_output_nodes_from_ports(MAX_PORTS, label="OUTPUT")
    return VALUES_NODES, MAX_NODES


def setup_adder_tree_iterative(circuit: CircuitGraph, num_amount=4, bit_len=4):
    SUMMANDS = []
    SUMMANDS_PORTS = []

    for i in range(num_amount):
        X = circuit.add_input_nodes(bit_len, label=f"X_{i}")
        X_PORTS = circuit.get_input_nodes_ports(X)
        SUMMANDS.append(X)
        SUMMANDS_PORTS.append(X_PORTS)
    zero = constant_zero(circuit, SUMMANDS_PORTS[0][0])
    O = adder_tree_iterative(circuit, SUMMANDS_PORTS, zero)
    O_NODES = circuit.generate_output_nodes_from_ports(O, label="OUTPUT")
    return SUMMANDS, O_NODES


def setup_lemma_4_1(interface: Interface, bit_len=4):
    if isinstance(interface, DepthInterface):
        X = [0] * bit_len
        M = [0] * bit_len
        result_depths = lemma_4_1.lemma_4_1(interface, X, M)
        max_depth = max(result_depths)
        return max_depth
    elif isinstance(interface, GraphInterface):
        X = interface.add_input_nodes(bit_len, label="X")
        X_PORTS = interface.get_input_nodes_ports(X)
        M = interface.add_input_nodes(bit_len, label="M")
        M_PORTS = interface.get_input_nodes_ports(M)
        O = lemma_4_1.lemma_4_1(interface, X_PORTS, M_PORTS)
        O_NODES = interface.generate_output_nodes_from_ports(O, label="OUTPUT")
        return X, M, O_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_lemma_4_1_reduce_in_parallel(circuit: CircuitGraph, bit_len=4):
    Y = circuit.add_input_nodes(bit_len, label="Y")
    Y_PORTS = circuit.get_input_nodes_ports(Y)
    M = circuit.add_input_nodes(bit_len, label="M")
    M_PORTS = circuit.get_input_nodes_ports(M)
    n = bit_len
    O = lemma_4_1.reduce_in_parallel(circuit, Y_PORTS, M_PORTS, n)
    O_NODES = circuit.generate_output_nodes_from_ports(O, "OUTPUT")
    return Y, M, O_NODES


# Returns M -> List[Node] and O_NODES -> List[List[Node]]
def setup_lemma_4_1_provide_aims_given_m(circuit: CircuitGraph, bit_len=4):
    M = circuit.add_input_nodes(bit_len, label="M")
    M_PORTS = circuit.get_input_nodes_ports(M)
    O = lemma_4_1.provide_aims_given_m(circuit, M_PORTS)
    O_NODES = []
    for num in O:
        num_nodes = circuit.generate_output_nodes_from_ports(num)
        O_NODES.append(num_nodes)
    return M, O_NODES


def setup_lemma_4_1_precompute_aim(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    input_node = circuit.add_input_nodes(1, "INPUT")[0]
    input_port = circuit.get_input_node_port(input_node)
    zero_port = constant_zero(circuit, input_port)
    one_port = constant_one(circuit, input_port)
    MATRIX_PORTS = lemma_4_1.precompute_aim(circuit, zero_port, one_port, n)
    MATRIX_NODES = []
    for row in MATRIX_PORTS:
        MATRIX_NODES.append(
            [circuit.generate_output_nodes_from_ports(ports) for ports in row]
        )
    return MATRIX_NODES


# O_NODES is a List[List[Port]]
def setup_lemma_4_1_compute_summands(circuit: CircuitGraph, num_amount=4, bit_len=4):
    X = circuit.add_input_nodes(bit_len, label="X")
    X_PORTS = circuit.get_input_nodes_ports(X)
    NUMS = []
    NUMS_PORTS = []
    for i in range(num_amount):
        num = circuit.add_input_nodes(bit_len, label=f"NUM_{i}")
        num_ports = circuit.get_input_nodes_ports(num)
        NUMS.append(num)
        NUMS_PORTS.append(num_ports)
    O = lemma_4_1.compute_summands(circuit, X_PORTS, NUMS_PORTS)
    O_NODES = []
    for summand in O:
        O_NODES.append(circuit.generate_output_nodes_from_ports(summand, label="OUT"))
    return X, NUMS, O_NODES


def setup_lemma_4_1_compute_y(circuit: CircuitGraph, bit_len=4):
    X = circuit.add_input_nodes(bit_len, label="X")
    X_PORTS = circuit.get_input_nodes_ports(X)
    M = circuit.add_input_nodes(bit_len, label="M")
    M_PORTS = circuit.get_input_nodes_ports(M)
    Y = lemma_4_1.compute_y(circuit, X_PORTS, M_PORTS)
    Y_NODES = circuit.generate_output_nodes_from_ports(Y, label="OUTPUT")
    return X, M, Y_NODES


def setup_lemma_4_1_compute_diffs(circuit: CircuitGraph, bit_len=4):
    Y = circuit.add_input_nodes(bit_len, label="Y")
    Y_PORTS = circuit.get_input_nodes_ports(Y)
    M = circuit.add_input_nodes(bit_len, label="M")
    M_PORTS = circuit.get_input_nodes_ports(M)
    n = bit_len
    D = lemma_4_1.compute_diffs(circuit, Y_PORTS, M_PORTS, n)
    D_NODES = []
    for d in D:
        d_nodes = circuit.generate_output_nodes_from_ports(d, label="OUTPUT")
        D_NODES.append(d_nodes)
    return Y, M, D_NODES


def setup_bus_multiplexer(circuit, num_amount=4, bit_len=4):
    BUS = []
    BUS_PORTS = []
    for i in range(num_amount):
        input = [circuit.add_node("input", f"BUS_{i}_{j}") for j in range(bit_len)]
        BUS.append(input)
        num_ports = []
        for nde in input:
            num_ports.append(nde.ports[0])
        BUS_PORTS.append(num_ports)
    selector = [
        circuit.add_node("input", f"SELECTOR_{i}")
        for i in range(int(math.log2(num_amount)))
    ]
    O = bus_multiplexer(circuit, BUS_PORTS, [s.ports[0] for s in selector])
    O_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(O)
    ]
    return BUS, selector, O_NODES


def setup_tensor_multiplexer(circuit: CircuitGraph, num_amount=4, bit_len=4):
    TENSOR = []
    TENSOR_PORTS = []
    for i in range(num_amount):
        column_nodes = []
        column_ports = []
        for j in range(num_amount):
            num = circuit.add_input_nodes(bit_len, label=f"INPUT_{i}_{j}")
            num_ports = circuit.get_input_nodes_ports(num)
            column_nodes.append(num)
            column_ports.append(num_ports)
        TENSOR.append(column_nodes)
        TENSOR_PORTS.append(column_ports)
    selector = circuit.add_input_nodes(int(math.log2(num_amount)), label="SELECTOR")
    selector_ports = circuit.get_input_nodes_ports(selector)
    BUS = tensor_multiplexer(circuit, TENSOR_PORTS, selector_ports)
    BUS_NODES = []
    for num in BUS:
        num_nodes = circuit.generate_output_nodes_from_ports(num)
        BUS_NODES.append(num_nodes)
    return TENSOR, selector, BUS_NODES


def setup_montgomery_ladder(interface, bit_len=4):
    n = bit_len
    if isinstance(interface, DepthInterface):
        B = [0] * n
        E = [0] * n
        M = [0] * n
        OUT_DEPTHS = montgomery_ladder(interface, B, E, M)
        max_depth = max(OUT_DEPTHS)
        return max_depth
    elif isinstance(interface, GraphInterface):
        B = [interface.add_node("input", f"B_{i}") for i in range(bit_len)]
        E = [interface.add_node("input", f"E_{i}") for i in range(bit_len)]
        M = [interface.add_node("input", f"M_{i}") for i in range(bit_len)]
        OUT = montgomery_ladder(
            interface,
            [b.ports[0] for b in B],
            [e.ports[0] for e in E],
            [m.ports[0] for m in M],
        )
        OUT_NODES = [
            interface.add_node("output", f"OUT_{i}", inputs=[o])
            for i, o in enumerate(OUT)
        ]
        return B, E, M, OUT_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_square_and_multiply(interface: Interface, bit_len=4):
    n = bit_len
    if isinstance(interface, DepthInterface):
        B = [0] * n
        E = [0] * n
        M = [0] * n
        OUT_DEPTHS = square_and_multiply(interface, B, E, M)
        max_depth = max(OUT_DEPTHS)
        return max_depth
    elif isinstance(interface, GraphInterface):
        B = [interface.add_node("input", f"B_{i}") for i in range(bit_len)]
        E = [interface.add_node("input", f"E_{i}") for i in range(bit_len)]
        M = [interface.add_node("input", f"M_{i}") for i in range(bit_len)]
        OUT = square_and_multiply(
            interface,
            [b.ports[0] for b in B],
            [e.ports[0] for e in E],
            [m.ports[0] for m in M],
        )
        OUT_NODES = [
            interface.add_node("output", f"OUT_{i}", inputs=[o])
            for i, o in enumerate(OUT)
        ]
        return B, E, M, OUT_NODES
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_modulo_circuit(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [circuit.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = modulo_circuit(circuit, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_slow_modulo_circuit(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [circuit.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = slow_modulo_circuit(circuit, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_optimized_modulo_circuit(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [circuit.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = modulo_circuit_optimized(
        circuit, [x.ports[0] for x in X], [a.ports[0] for a in A]
    )
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_reciprocal_newton_raphson(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(4)]
    OUT = reciprocal_newton_raphson(circuit, [x.ports[0] for x in X], 4)
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, OUT_NODES


def setup_log2_estimate(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(4)]
    OUT = log2_estimate(circuit, [x.ports[0] for x in X])
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, OUT_NODES


def setup_or_tree_recursive(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    o = or_tree_recursive(circuit, [x.ports[0] for x in X])
    o_node = circuit.add_node("output", "OUT", inputs=[o])
    return X, o_node


def setup_or_tree_iterative(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    o = or_tree_iterative(circuit, [x.ports[0] for x in X])
    o_node = circuit.add_node("output", "OUT", inputs=[o])
    return X, o_node


def setup_and_tree_iterative(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    o = and_tree_iterative(circuit, [x.ports[0] for x in X])
    o_node = circuit.add_node("output", "OUT", inputs=[o])
    return X, o_node


def setup_n_left_shift(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [circuit.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = n_left_shift(circuit, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_n_right_shift(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    A = [circuit.add_node("input", f"A_{i}") for i in range(bit_len)]
    OUT = n_right_shift(circuit, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, A, OUT_NODES


def setup_one_left_shift(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    OUT = one_left_shift(circuit, [x.ports[0] for x in X])
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
    ]
    return X, OUT_NODES


def setup_one_right_shift(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X_{i}") for i in range(bit_len)]
    OUT = one_right_shift(circuit, [x.ports[0] for x in X])
    OUT_NODES = [
        circuit.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)
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


def setup_adder_tree_recursive(circuit, bit_len=4):
    ports = []
    for k in range(4):
        X = [circuit.add_node("input", f"X{i}") for i in range(bit_len)]
        ports.append([x.ports[0] for x in X])
    cin = circuit.add_node("input", "CIN")
    outputs, carry = adder_tree_recursive(circuit, ports, cin.ports[0])
    for out in outputs:
        circuit.add_node("output", "SUM", inputs=[out])
    circuit.add_node("output", "CARRY", inputs=[carry])
    return


def setup_multiplexer(circuit, bit_len=4):
    X = [circuit.add_node("input", f"X{i}") for i in range(bit_len)]
    S = [circuit.add_node("input", f"S{i}") for i in range(int(math.log2(bit_len)))]
    mux = multiplexer(circuit, [x.ports[0] for x in X], [s.ports[0] for s in S])
    out = circuit.add_node("output", "MUX OUT", inputs=[mux])
    return


def setup_one_bit_comparator(circuit, bit_len=4):
    less, equals, greater = one_bit_comparator(
        circuit,
        circuit.add_node("input", "x").ports[0],
        circuit.add_node("input", "y").ports[0],
    )
    less_node = circuit.add_node("output", "LESS", inputs=[less])
    equals_node = circuit.add_node("output", "EQUALS", inputs=[equals])
    greater_node = circuit.add_node("output", "GREATER", inputs=[greater])
    return


def setup_n_bit_comparator(circuit, bit_len=4):
    A = [circuit.add_node("input", f"A{i}") for i in range(bit_len)]
    B = [circuit.add_node("input", f"B{i}") for i in range(bit_len)]
    less, equals, greater = n_bit_comparator(
        circuit, [a.ports[0] for a in A], [b.ports[0] for b in B]
    )
    L = circuit.add_node("output", "LESS", inputs=[less])
    E = circuit.add_node("output", "EQUALS", inputs=[equals])
    G = circuit.add_node("output", "GREATER", inputs=[greater])
    return A, B, L, E, G


def setup_n_bit_equality(circuit: CircuitGraph, bit_len=4):
    n = bit_len
    A = circuit.add_input_nodes(n)
    B = circuit.add_input_nodes(n)
    A_PORTS = circuit.get_input_nodes_ports(A)
    B_PORTS = circuit.get_input_nodes_ports(B)
    equals = n_bit_equality(circuit, A_PORTS, B_PORTS)
    equals_node = circuit.generate_output_node_from_port(equals)
    return A, B, equals_node


def setup_xnor_gate(circuit: CircuitGraph, bit_len=4):
    X = [circuit.add_node("input", f"X{i}") for i in range(2)]
    out = xnor_gate(circuit, X[0].ports[0], X[1].ports[0])
    xnor_output = circuit.add_node("output", "XNOR OUTPUT", inputs=[out])
    return


def setup_constant_zero(circuit: CircuitGraph, bit_len=4):
    X = circuit.add_node("input", "X")
    zero_port = constant_zero(circuit, X.ports[0])
    zero_node = circuit.add_node("output", "CONSTANT ZERO", inputs=[zero_port])
    return


def setup_constant_one(circuit: CircuitGraph, bit_len=4):
    X = circuit.add_node("input", "X")
    one_port = constant_one(circuit, X.ports[0])
    one_node = circuit.add_node("output", "CONSTANT ONE", inputs=[one_port])
    return


def setup_half_adder(interface: Interface, bit_len=4):
    if isinstance(interface, DepthInterface):
        a, b = 0, 0
        sum_depth, carry_depth = half_adder(interface, a, b)
        max_depth = max(sum_depth, carry_depth)
        return max_depth
    elif isinstance(interface, GraphInterface):
        A = interface.add_node("input", "A")
        B = interface.add_node("input", "B")
        sum_port, carry_port = half_adder(interface, A.ports[0], B.ports[0])
        sum_node = interface.add_node("output", "sum")
        carry_node = interface.add_node("output", "carry")
        interface.add_edge(sum_port, sum_node.ports[0])
        interface.add_edge(carry_port, carry_node.ports[0])
        return A, B, sum_node, carry_node
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_full_adder(interface: Interface, bit_len=4):
    if isinstance(interface, DepthInterface):
        a, b, cin = 0, 0, 0
        sum_depth, carry_depth = full_adder(interface, a, b, cin)
        max_depth = max(sum_depth, carry_depth)
        return max_depth
    elif isinstance(interface, GraphInterface):
        A = interface.add_node("input", "A")
        B = interface.add_node("input", "B")
        cin = interface.add_node("input", "Cin")
        sum_port, carry_port = full_adder(
            interface, A.ports[0], B.ports[0], cin.ports[0]
        )
        sum_node = interface.add_node("output", "sum")
        carry_node = interface.add_node("output", "carry")
        interface.add_edge(sum_port, sum_node.ports[0])
        interface.add_edge(carry_port, carry_node.ports[0])
        return A, B, cin, sum_node, carry_node
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_ripple_carry_adder(interface: Interface, bit_len=4):
    if isinstance(interface, DepthInterface):
        A = [0] * bit_len
        B = [0] * bit_len
        cin = 0
        sum_depths, carry_depth = ripple_carry_adder(interface, A, B, cin)
        max_depth = max(max(sum_depths), carry_depth)
        return max_depth
    elif isinstance(interface, GraphInterface):
        A = interface.add_input_nodes(bit_len, label="A")
        B = interface.add_input_nodes(bit_len, label="B")
        cin = interface.add_node("input", "Cin")
        sum_outputs, carry_out = ripple_carry_adder(
            interface, [a.ports[0] for a in A], [b.ports[0] for b in B], cin.ports[0]
        )
        sum_nodes = []
        for sum in sum_outputs:
            sum_nodes.append(interface.add_node("output", "SUM", inputs=[sum]))
        carry_node = interface.add_node("output", "CARRY", inputs=[carry_out])
        return A, B, cin, sum_nodes, carry_node
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_carry_look_ahead_adder(interface: Interface, bit_len=4):
    if isinstance(interface, DepthInterface):
        A = [0] * bit_len
        B = [0] * bit_len
        cin = 0
        sum_depths, carry_depth = carry_look_ahead_adder(interface, A, B, cin)
        max_depth = max(max(sum_depths), carry_depth)
        return max_depth
    elif isinstance(interface, GraphInterface):
        A = interface.add_input_nodes(bit_len, label="A")
        B = interface.add_input_nodes(bit_len, label="B")
        cin = interface.add_node("input", "Cin")
        sum_outputs, carry_out = carry_look_ahead_adder(
            interface,
            interface.get_input_nodes_ports(A),
            interface.get_input_nodes_ports(B),
            interface.get_input_node_port(cin),
        )
        sum_nodes = []
        for sum in sum_outputs:
            sum_nodes.append(interface.add_node("output", "SUM", inputs=[sum]))
        carry_node = interface.add_node("output", "CARRY", inputs=[carry_out])
        return A, B, cin, sum_nodes, carry_node
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_wallace_tree_multiplier(interface: Interface, bit_len=4):
    if isinstance(interface, DepthInterface):
        A = [0] * bit_len
        B = [0] * bit_len
        out_depths = wallace_tree_multiplier(interface, A, B)
        max_depth = max(out_depths)
        return max_depth
    elif isinstance(interface, GraphInterface):
        A = [interface.add_node("input", f"A{i}") for i in range(bit_len)]
        B = [interface.add_node("input", f"B{i}") for i in range(bit_len)]
        outputs = wallace_tree_multiplier(
            interface, [a.ports[0] for a in A], [b.ports[0] for b in B]
        )
        output_nodes = []
        for out in outputs:
            output_nodes.append(interface.add_node("output", "PRODUCT", inputs=[out]))
        return A, B, output_nodes
    else:
        raise TypeError(f"Unsupported interface type: {type(interface).__name__}")


def setup_subtract(cg: CircuitGraph, bit_len=4):
    n = bit_len
    A_NODES = cg.add_input_nodes(n)
    B_NODES = cg.add_input_nodes(n)
    A_PORTS = cg.get_input_nodes_ports(A_NODES)
    B_PORTS = cg.get_input_nodes_ports(B_NODES)
    DIFF_PORTS = subtract(cg, A_PORTS, B_PORTS)
    DIFF_NODES = cg.generate_output_nodes_from_ports(DIFF_PORTS)
    return A_NODES, B_NODES, DIFF_NODES


def setup_two_complement(cg: CircuitGraph, bit_len=4):
    n = bit_len
    X_NODES = cg.add_input_nodes(n)
    X_PORTS = cg.get_input_nodes_ports(X_NODES)
    RESULT_PORTS = two_complement(cg, X_PORTS)
    RESULT_NODES = cg.generate_output_nodes_from_ports(RESULT_PORTS)
    return X_NODES, RESULT_NODES


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

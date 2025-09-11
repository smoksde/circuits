from typing import Optional, List, Tuple
from core.graph import *
import utils
import math

import sanity.software_beame as sb
from ..standard.constants import *
from ..standard.multiplexers import tensor_multiplexer, bus_multiplexer
from ..standard.comparators import n_bit_comparator, n_bit_equality
from ..standard.manipulators import (
    max_tree_iterative,
    min_tree_iterative,
    conditional_zeroing,
)
from ..standard.trees import or_tree_iterative, adder_tree_iterative
from ..standard.adders import carry_look_ahead_adder
from ..standard.subtractors import subtract
from ..standard.shifters import n_left_shift
from ..standard.multipliers import wallace_tree_multiplier
from .. import circuit_utils

from . import lemma_4_1

import sanity.theorem_4_2_sanity as theorem_4_2_sanity


def precompute_lookup_is_prime_power(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group(label="PRECOMPUTE_LOOKUP_IS_PRIME_POWER")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    result = []
    for i in range(1, n + 1):
        if utils.is_prime_power(i):
            result.append(one)
        else:
            result.append(zero)
    return result


def precompute_lookup_p_l(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group(label="PRECOMPUTE_LOOKUP_P_L")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    p_result = []
    l_result = []
    for i in range(0, n + 1):
        if utils.is_prime_power(i) and i != 0:
            factorization = utils.wheel_factorize(i)
            p = factorization[0]
            l = len(factorization)
            # FILL
            p_bits = utils.int2binlist(p, bit_len=n)
            l_bits = utils.int2binlist(l, bit_len=n)
            P_PORTS = []
            for p_bit in p_bits:
                if p_bit:
                    P_PORTS.append(one)
                else:
                    P_PORTS.append(zero)
            L_PORTS = []
            for l_bit in l_bits:
                if l_bit:
                    L_PORTS.append(one)
                else:
                    L_PORTS.append(zero)
            p_result.append(P_PORTS)
            l_result.append(L_PORTS)
        else:
            P_PORTS = [zero for _ in range(n)]
            L_PORTS = [zero for _ in range(n)]
            p_result.append(P_PORTS)
            l_result.append(L_PORTS)
    return p_result, l_result


def precompute_lookup_powers(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
) -> List[List[List[Port]]]:
    this_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_POWERS")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    result = []
    for p in range(0, n + 1):
        powers_of_p = []
        for e in range(n):
            if e > math.log2(n) or p**e > n or p == 0:
                power = 0
            else:
                power = p**e
            power_bits = utils.int2binlist(power, bit_len=n)
            power_ports = []
            for bit in power_bits:
                if bit:
                    power_ports.append(one)
                else:
                    power_ports.append(zero)
            powers_of_p.append(power_ports)
        result.append(powers_of_p)

    return result


def precompute_lookup_generator_powers(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_GENERATOR_POWERS")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    result = []
    software_primitive_roots = sb.find_primitive_roots(n)
    software_p_l_lookup = sb.theorem_4_2_precompute_lookup_p_l(n)

    for pexpl in range(0, n + 1):
        row = []
        p, l = software_p_l_lookup[pexpl]
        if p == 0 or l == 0:
            thresh = 0
        else:
            thresh = int(math.pow(2, n))
        g = software_primitive_roots[pexpl]
        software_pows_of_g = sb.compute_powers_mod_up_to(g, pexpl, thresh)
        while len(software_pows_of_g) < n:
            software_pows_of_g.append(0)

        for entry in software_pows_of_g:
            bits = utils.int2binlist(entry, bit_len=n)
            num_ports = []
            for bit in bits:
                if bit:
                    num_ports.append(one)
                else:
                    num_ports.append(zero)
            row.append(num_ports)
        result.append(row)
    return result


def precompute_lookup_tables_B(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_TABLES_B")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    TABLE_ZERO = []
    for l in range(0, int(math.log2(n)) + 1):
        row = []
        for b in range(0, n + 1):
            try:
                value = theorem_4_2_sanity.compute_a_b_l_formula(0, b, l)
            except:
                value = 0
            if value > 2**n - 1:
                value = 0
            value_bits = utils.int2binlist(value, bit_len=n)

            num = []
            for bit in value_bits:
                if bit:
                    num.append(one)
                else:
                    num.append(zero)
            row.append(num)
        TABLE_ZERO.append(row)

    TABLE_ONE = []
    for l in range(0, int(math.log2(n)) + 1):
        row = []
        for b in range(0, n + 1):
            try:
                value = theorem_4_2_sanity.compute_a_b_l_formula(1, b, l)
            except:
                value = 0
            if value > (2**n) - 1:
                value = 0
            value_bits = utils.int2binlist(value, bit_len=n)

            num = []
            for bit in value_bits:
                if bit:
                    num.append(one)
                else:
                    num.append(zero)
            row.append(num)
        TABLE_ONE.append(row)

    return TABLE_ZERO, TABLE_ONE


def step_1_with_lemma_4_1(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    p: List[Port],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
) -> List[List[Port]]:
    n = len(x_list)

    this_group = circuit.add_group("THEOREM_4_2_STEP_1_WITH_LEMMA_4_1")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    zero = constant_zero(circuit, p[0], parent_group=this_group)
    one = constant_one(circuit, p[0], parent_group=this_group)

    lookup_powers = precompute_lookup_powers(
        circuit, zero, one, len(x_list), parent_group=this_group
    )

    p_powers = tensor_multiplexer(circuit, lookup_powers, p)

    exponents = []

    for x in x_list:
        res_list = []
        for i in range(int(math.log2(n))):
            less, equal, greater = n_bit_comparator(
                circuit, p_powers[i], pexpl, parent_group=this_group
            )
            goreq = not_gate(circuit, less, parent_group=this_group)

            remainder = lemma_4_1.lemma_4_1(
                circuit, x, p_powers[i], parent_group=this_group
            )

            is_remainder_not_zero = or_tree_iterative(
                circuit, remainder, parent_group=this_group
            )

            p_powers_is_not_zero = or_tree_iterative(
                circuit, p_powers[i], parent_group=this_group
            )
            p_powers_is_zero = not_gate(
                circuit, p_powers_is_not_zero, parent_group=this_group
            )

            cond = or_gate(
                circuit, [is_remainder_not_zero, goreq], parent_group=this_group
            )

            cond = or_gate(circuit, [cond, p_powers_is_zero], parent_group=this_group)

            exponent = circuit_utils.generate_number(i, n, zero, one)

            res = conditional_zeroing(circuit, exponent, cond, parent_group=this_group)
            res_list.append(res)
        exponent = max_tree_iterative(circuit, res_list, parent_group=this_group)
        exponents.append(exponent)

    return exponents


def step_1(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    p: List[Port],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
) -> List[List[Port]]:
    this_group = circuit.add_group("THEOREM_4_2_STEP_1")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    return step_1_with_precompute(circuit, x_list, p, parent_group=this_group)
    # return step_1_with_lemma_4_1(circuit, x_list, p, pexpl, parent_group=this_group)


def precompute_largest_powers(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
) -> List[List[List[Port]]]:
    this_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LARGEST_POWERS")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    result = []

    for p in range(0, n + 1):
        row = []
        for x in range(0, n):
            largest_exp = 0
            j = 0
            while True:
                if p == 0 or p == 1:
                    largest_exp = 0
                    break
                power = p**j
                if power > x:
                    break
                elif x % power == 0:
                    largest_exp = j
                j += 1
            largest_exp_ports = circuit_utils.generate_number(
                largest_exp, n, zero, one, parent_group=this_group
            )
            row.append(largest_exp_ports)
        result.append(row)
    return result


def step_1_with_precompute(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    p: List[Port],
    parent_group: Optional[Group] = None,
) -> List[List[Port]]:

    this_group = circuit.add_group("THEOREM_4_2_STEP_1_WITH_PRECOMPUTE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(x_list)

    zero = constant_zero(circuit, p[0], parent_group=this_group)
    one = constant_one(circuit, p[0], parent_group=this_group)

    lookup_largest_powers = precompute_largest_powers(
        circuit, zero, one, n, parent_group=this_group
    )

    lookup_largest_powers_for_p = tensor_multiplexer(
        circuit, lookup_largest_powers, p, parent_group=this_group
    )

    exponents = []

    for x in x_list:
        lookup_power = bus_multiplexer(
            circuit, lookup_largest_powers_for_p, x, parent_group=this_group
        )
        exponents.append(lookup_power)
    return exponents


def step_2(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    p: List[Port],
    j_list: List[List[Port]],
    parent_group: Optional[Group] = None,
):

    this_group = circuit.add_group("THEOREM_4_2_STEP_2")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(x_list)

    zero = constant_zero(circuit, p[0], parent_group=this_group)
    one = constant_one(circuit, p[0], parent_group=this_group)

    powers_lookup = precompute_lookup_powers(
        circuit, zero, one, n, parent_group=this_group
    )

    division_lookup = precompute_lookup_division(
        circuit, zero, one, n, parent_group=this_group
    )

    powers_of_p_bus = tensor_multiplexer(
        circuit, powers_lookup, p, parent_group=this_group
    )

    y_list = []
    for i in range(n):
        x = x_list[i]

        power = bus_multiplexer(
            circuit, powers_of_p_bus, j_list[i], parent_group=this_group
        )
        quotient_bus = tensor_multiplexer(
            circuit, division_lookup, x, parent_group=this_group
        )
        quotient = bus_multiplexer(
            circuit, quotient_bus, power, parent_group=this_group
        )
        y_list.append(quotient)
    return y_list


def compute_sum(
    circuit: CircuitGraph,
    j_list: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_STEP_3")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    zero = constant_zero(circuit, j_list[0][0], parent_group=this_group)
    j = adder_tree_iterative(circuit, j_list, zero, parent_group=this_group)
    return j


def step_4(
    circuit: CircuitGraph,
    p: List[Port],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
) -> Port:
    this_group = circuit.add_group("THEOREM_4_2_STEP_4")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(p)

    zero = constant_zero(circuit, p[0], parent_group=this_group)
    one = constant_one(circuit, p[0], parent_group=this_group)

    num_two = [zero for _ in range(n)]
    num_two[1] = one

    num_four = [zero for _ in range(n)]
    num_four[2] = one

    _, p_equals_two, _ = n_bit_comparator(circuit, p, num_two, parent_group=this_group)

    p_not_equals_two = not_gate(circuit, p_equals_two, parent_group=this_group)

    _, pexpl_equals_two, _ = n_bit_comparator(
        circuit, pexpl, num_two, parent_group=this_group
    )
    _, pexpl_equals_four, _ = n_bit_comparator(
        circuit, pexpl, num_four, parent_group=this_group
    )

    two_or_four = or_gate(
        circuit, [pexpl_equals_two, pexpl_equals_four], parent_group=this_group
    )

    flag = or_gate(circuit, [p_not_equals_two, two_or_four], parent_group=this_group)

    return flag


def A_step_5(
    circuit: CircuitGraph,
    y_list: List[List[Port]],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
):

    this_group = circuit.add_group("THEOREM_4_2_A_STEP_5")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(y_list[0])

    zero = constant_zero(circuit, y_list[0][0], parent_group=this_group)
    one = constant_one(circuit, y_list[0][0], parent_group=this_group)

    generator_powers_lookup = precompute_lookup_generator_powers(
        circuit, zero, one, n, parent_group=this_group
    )

    generator_powers_bus = tensor_multiplexer(
        circuit, generator_powers_lookup, pexpl, parent_group=this_group
    )

    a_list = []

    supremum_num = len(generator_powers_bus)
    supremum = circuit_utils.generate_number(supremum_num, n, zero, one)

    for y in y_list:
        e_candidates = []
        for idx, num in enumerate(generator_powers_bus):
            equals = n_bit_equality(circuit, y, num, parent_group=this_group)
            e_num = circuit_utils.generate_number(idx, n, zero, one)
            e_num = bus_multiplexer(
                circuit, [supremum, e_num], [equals], parent_group=this_group
            )
            e_candidates.append(e_num)
        a = min_tree_iterative(circuit, e_candidates, parent_group=this_group)
        a_list.append(a)
    return a_list


def A_step_7(
    circuit: CircuitGraph,
    a: List[Port],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("THEOREM_4_2_A_STEP_7")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(pexpl)

    zero_port = constant_zero(circuit, a[0], parent_group=this_group)
    one_port = constant_one(circuit, a[0], parent_group=this_group)

    pexpl_minus_pexpl_minus_one_lookup = precompute_lookup_pexpl_minus_pexpl_minus_one(
        circuit, zero_port, one_port, n, parent_group=this_group
    )

    m = bus_multiplexer(
        circuit, pexpl_minus_pexpl_minus_one_lookup, pexpl, parent_group=this_group
    )

    a_hat = lemma_4_1.lemma_4_1(circuit, a, m, parent_group=this_group)

    return a_hat


def A_step_8(
    circuit: CircuitGraph,
    a_hat: List[Port],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("THEOREM_4_2_A_step_8")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(a_hat)

    zero = constant_zero(circuit, a_hat[0], parent_group=this_group)
    one = constant_one(circuit, a_hat[0], parent_group=this_group)

    disc_log_lookup = precompute_lookup_generator_powers(
        circuit, zero, one, n, parent_group=this_group
    )

    bus = tensor_multiplexer(circuit, disc_log_lookup, pexpl, parent_group=this_group)
    y_product = bus_multiplexer(circuit, bus, a_hat, parent_group=this_group)
    return y_product


def B_step_5(
    circuit: CircuitGraph,
    y_list: List[List[Port]],
    l: List[Port],
    parent_group: Optional[Group] = None,
) -> Tuple[List[List[Port]], List[List[Port]]]:

    this_group = circuit.add_group("THEOREM_4_2_B_STEP_5")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(y_list[0])

    zero = constant_zero(circuit, l[0], parent_group=this_group)
    one = constant_one(circuit, l[0], parent_group=this_group)

    table_a_zero, table_a_one = precompute_lookup_tables_B(
        circuit, zero, one, n, parent_group=this_group
    )

    num_one = [zero for _ in range(n)]
    num_one[0] = one

    table_a_zero_twoexpl_bus = tensor_multiplexer(
        circuit, table_a_zero, l, parent_group=this_group
    )

    table_a_one_twoexpl_bus = tensor_multiplexer(
        circuit, table_a_one, l, parent_group=this_group
    )

    len_a_zero_bus = len(table_a_zero_twoexpl_bus)
    supremum_a_zero = circuit_utils.generate_number(len_a_zero_bus, n, zero, one)

    len_a_one_bus = len(table_a_one_twoexpl_bus)
    supremum_a_one = circuit_utils.generate_number(len_a_one_bus, n, zero, one)

    a_list = []
    b_list = []

    for y_i in y_list:
        index_a_zero = compute_value_index_from_bus(
            circuit,
            y_i,
            table_a_zero_twoexpl_bus,
            supremum_a_zero,
            zero,
            one,
            parent_group=this_group,
        )
        index_a_one = compute_value_index_from_bus(
            circuit,
            y_i,
            table_a_one_twoexpl_bus,
            supremum_a_one,
            zero,
            one,
            parent_group=this_group,
        )

        equals = n_bit_equality(
            circuit, index_a_zero, supremum_a_zero, parent_group=this_group
        )

        index = bus_multiplexer(
            circuit, [index_a_zero, index_a_one], [equals], parent_group=this_group
        )

        b_i = index
        a_i = [zero for _ in range(n)]
        a_i[0] = equals

        a_list.append(a_i)
        b_list.append(b_i)

    return (a_list, b_list)


def compute_value_index_from_bus(
    circuit: CircuitGraph,
    value: List[Port],
    bus: List[List[Port]],
    supremum: List[List[Port]],
    zero: Port,
    one: Port,
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("COMPUTE_VALUE_INDEX_FROM_BUS")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(value)
    index_candidates = []
    for idx, num in enumerate(bus):
        equals = n_bit_equality(circuit, value, num, parent_group=this_group)
        index_num = circuit_utils.generate_number(idx, n, zero, one)
        index_num = bus_multiplexer(
            circuit, [supremum, index_num], [equals], parent_group=this_group
        )
        index_candidates.append(index_num)
    index_value = min_tree_iterative(circuit, index_candidates, parent_group=this_group)
    return index_value


def B_step_7(
    circuit: CircuitGraph,
    a: List[Port],
    b: List[Port],
    l: List[Port],
    parent_group: Optional[Group] = None,
) -> Tuple[List[Port], List[Port]]:
    this_group = circuit.add_group("THEOREM_4_2_B_step_7")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(a)

    zero = constant_zero(circuit, a[0], parent_group=this_group)
    one = constant_one(circuit, a[0], parent_group=this_group)

    a_hat = [zero for _ in range(n)]
    a_hat[0] = a[0]

    two_num = [zero for _ in range(n)]
    two_num[1] = one
    l_minus_two = subtract(circuit, l, two_num, parent_group=this_group)
    one_num = [zero for _ in range(n)]
    one_num[0] = one
    m_for_b = n_left_shift(circuit, one_num, l_minus_two, parent_group=this_group)
    b_hat = lemma_4_1.lemma_4_1(circuit, b, m_for_b, parent_group=this_group)

    return a_hat, b_hat


def B_step_8(
    circuit: CircuitGraph,
    a_hat: List[Port],
    b_hat: List[Port],
    l: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("THEOREM_4_2_B_step_8")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(a_hat)

    zero = constant_zero(circuit, l[0], parent_group=this_group)
    one = constant_one(circuit, l[0], parent_group=this_group)

    num_one = [zero for _ in range(n)]
    num_one[0] = one

    table_a_zero, table_a_one = precompute_lookup_tables_B(
        circuit, zero, one, n, parent_group=this_group
    )

    table_a_zero_twoexpl_bus = tensor_multiplexer(
        circuit, table_a_zero, l, parent_group=this_group
    )

    table_a_one_twoexpl_bus = tensor_multiplexer(
        circuit, table_a_one, l, parent_group=this_group
    )

    table_a_zero_result = bus_multiplexer(
        circuit, table_a_zero_twoexpl_bus, b_hat, parent_group=this_group
    )

    table_a_one_result = bus_multiplexer(
        circuit, table_a_one_twoexpl_bus, b_hat, parent_group=this_group
    )

    equals = n_bit_equality(circuit, a_hat, num_one, parent_group=this_group)

    not_equals = not_gate(circuit, equals, parent_group=this_group)

    r1 = conditional_zeroing(
        circuit, table_a_zero_result, equals, parent_group=this_group
    )
    r2 = conditional_zeroing(
        circuit, table_a_one_result, not_equals, parent_group=this_group
    )

    sum, carry = carry_look_ahead_adder(circuit, r1, r2, zero, parent_group=this_group)

    return sum


def step_9(
    circuit: CircuitGraph,
    p: List[Port],
    j: List[Port],
    pexpl: List[Port],
    y_product: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("THEOREM_4_2_STEP_9")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(p)

    zero_port = constant_zero(circuit, p[0], parent_group=this_group)
    one_port = constant_one(circuit, p[0], parent_group=this_group)

    power_lookup = precompute_lookup_powers(
        circuit, zero_port, one_port, n, parent_group=this_group
    )
    p_powers = tensor_multiplexer(circuit, power_lookup, p, parent_group=this_group)

    pexpj = bus_multiplexer(circuit, p_powers, j, parent_group=this_group)

    product = wallace_tree_multiplier(
        circuit, pexpj, y_product, parent_group=this_group
    )

    product = product[:n]

    result = lemma_4_1.lemma_4_1(circuit, product, pexpl, parent_group=this_group)

    return result


def theorem_4_2_not_modified(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("THEOREM_4_2_NOT_MODIFIED")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(pexpl)

    zero = constant_zero(circuit, pexpl[0], parent_group=this_group)
    one = constant_one(circuit, pexpl[0], parent_group=this_group)

    p_lookup, l_lookup = precompute_lookup_p_l(
        circuit, zero, one, n, parent_group=this_group
    )

    p = bus_multiplexer(circuit, p_lookup, pexpl)
    l = bus_multiplexer(circuit, l_lookup, pexpl)

    do_a = step_4(circuit, p, pexpl, parent_group=this_group)

    j_list = step_1(circuit, x_list, p, pexpl, parent_group=this_group)
    y_list = step_2(circuit, x_list, p, j_list, parent_group=this_group)
    j = compute_sum(circuit, j_list, parent_group=this_group)

    a_list = A_step_5(circuit, y_list, pexpl, parent_group=this_group)
    a = compute_sum(circuit, a_list)

    while len(a) < len(pexpl):
        a.append(zero)
    a_hat = A_step_7(circuit, a, pexpl, parent_group=this_group)
    y_product_part_a = A_step_8(circuit, a_hat, pexpl, parent_group=this_group)

    new_y_list = []
    for _ in range(len(y_list)):
        new_y_list.append([zero for _ in range(len(y_list[0]))])
    y_list = new_y_list

    a_list, b_list = B_step_5(circuit, y_list, l, parent_group=this_group)
    a = compute_sum(circuit, a_list, parent_group=this_group)
    b = compute_sum(circuit, b_list, parent_group=this_group)
    a_hat, b_hat = B_step_7(circuit, a, b, l, parent_group=this_group)

    y_product_part_b = B_step_8(circuit, a_hat, b_hat, l, parent_group=this_group)

    y_product = bus_multiplexer(
        circuit, [y_product_part_b, y_product_part_a], [do_a], parent_group=this_group
    )

    result = step_9(circuit, p, j, pexpl, y_product, parent_group=this_group)

    return result


def theorem_4_2(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("THEOREM_4_2")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(pexpl)

    zero = constant_zero(circuit, pexpl[0], parent_group=this_group)
    one = constant_one(circuit, pexpl[0], parent_group=this_group)

    p_lookup, l_lookup = precompute_lookup_p_l(
        circuit, zero, one, n, parent_group=this_group
    )

    p = bus_multiplexer(circuit, p_lookup, pexpl)
    l = bus_multiplexer(circuit, l_lookup, pexpl)

    j_list = step_1(circuit, x_list, p, pexpl, parent_group=this_group)
    y_list = step_2(circuit, x_list, p, j_list, parent_group=this_group)
    j = compute_sum(circuit, j_list, parent_group=this_group)

    a_list = A_step_5(circuit, y_list, pexpl, parent_group=this_group)
    a = compute_sum(circuit, a_list)
    while len(a) < len(pexpl):
        a.append(zero)
    a_hat = A_step_7(circuit, a, pexpl, parent_group=this_group)
    y_product = A_step_8(circuit, a_hat, pexpl, parent_group=this_group)

    result = step_9(circuit, p, j, pexpl, y_product, parent_group=this_group)

    return result


def precompute_lookup_pexpl_minus_pexpl_minus_one(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group(
        "THEOREM_4_2_PRECOMPUTE_LOOKUP_PEXPL_MINUS_PEXPL_MINUS_ONE"
    )
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    p_l_lookup = sb.theorem_4_2_precompute_lookup_p_l(n)

    table = []
    for pexpl in range(0, n + 1):
        p, l = p_l_lookup[pexpl]
        if l - 1 < 0:
            term = 0
        else:
            term = pexpl - p ** (l - 1)
        term_bits = utils.int2binlist(term, bit_len=n)
        num = []
        for bit in term_bits:
            if bit:
                num.append(one)
            else:
                num.append(zero)
        table.append(num)
    return table


def precompute_lookup_division(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_DIVISION")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    table = []
    for x in range(n + 1):
        row = []
        for y in range(n + 1):
            ports = []
            if x == 0 or y == 0:
                ports = [zero for _ in range(n)]
            else:
                value = x // y
                value_bits = utils.int2binlist(value, bit_len=n)
                for bit in value_bits:
                    if bit:
                        ports.append(one)
                    else:
                        ports.append(zero)
            row.append(ports)
        table.append(row)
    return table

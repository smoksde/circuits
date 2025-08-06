from typing import List, Optional, Tuple
from graph import *
from .constants import *
from utils import int2binlist, is_prime_power, wheel_factorize
from .utils import generate_number
from .multipliers import *
from .manipulators import (
    conditional_zeroing,
    max_tree_iterative,
    smallest_non_zero_tree_iterative,
    min_tree_iterative,
)
from .multiplexers import tensor_multiplexer, bus_multiplexer
from .adders import carry_look_ahead_adder
from .trees import adder_tree_iterative, or_tree_iterative
from .subtractors import subtract
from .comparators import n_bit_comparator, n_bit_equality
from .shifters import one_left_shift
import math

import software_beame as sb
import sanity

from asserts import *


# Lemma 4.1
# Creates a list of lists with first index m from 1 to n and second index i from 0 to n - 1
# m <= n
def precompute_aim(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    pa_group = circuit.add_group("PRE_AIM")
    pa_group.set_parent(parent_group)

    lis = []

    for m in range(0, n + 1):
        fix_m_entries = []
        for i in range(n):
            aim_entry = []
            if m == 0:
                aim_value = 0
            else:
                aim_value = int((2**i) % m)
            b_list = int2binlist(aim_value, bit_len=n)
            for bit in b_list:
                if bit:
                    aim_entry.append(one)
                else:
                    aim_entry.append(zero)
            fix_m_entries.append(aim_entry)
        lis.append(fix_m_entries)

    assert_tensor_of_ports(lis)
    return lis


def lemma_4_1_provide_aims_given_m(
    circuit: CircuitGraph, m: List[Port], parent_group: Optional[Group] = None
):
    pagm_group = circuit.add_group("LEMMA_4_1_PROVIDE_AIMS_GIVEN_M")
    pagm_group.set_parent(parent_group)
    zero = constant_zero(circuit, m[0], parent_group=pagm_group)
    one = constant_one(circuit, m[0], parent_group=pagm_group)
    n = len(m)
    aims_ports = precompute_aim(circuit, zero, one, n, parent_group=pagm_group)
    # figure out if aims_ports needs to be reformated
    ais = tensor_multiplexer(circuit, aims_ports, m)
    return ais  # List[List[Port]]


# computes x_i * a_im terms in the big summation
def lemma_4_1_compute_summands(
    circuit: CircuitGraph,
    x: List[Port],
    aims: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    cs_group = circuit.add_group("LEMMA_4_1_COMPUTE_SUMMANDS")
    cs_group.set_parent(parent_group)
    n = len(x)
    result = []
    for p1, p2 in zip(x, aims):
        not_p1 = circuit.add_node(
            "not", "NOT", inputs=[p1], group_id=cs_group.id
        ).ports[1]
        product = conditional_zeroing(circuit, p2, not_p1, parent_group=cs_group)
        result.append(product)
    return result


def lemma_4_1_compute_y(
    circuit: CircuitGraph,
    x: List[Port],
    m: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:
    cy_group = circuit.add_group("LEMMA_4_1_COMPUTE_Y")
    cy_group.set_parent(parent_group)

    zero = constant_zero(circuit, x[0], parent_group=cy_group)

    aims = lemma_4_1_provide_aims_given_m(circuit, m, parent_group=cy_group)
    summands = lemma_4_1_compute_summands(circuit, x, aims, parent_group=cy_group)
    sum = adder_tree_iterative(circuit, summands, zero, parent_group=cy_group)

    assert_list_of_ports(sum)
    return sum


def lemma_4_1_compute_diffs(
    circuit: CircuitGraph,
    y: List[Port],
    m: List[Port],
    n: int,
    parent_group: Optional[Group] = None,
) -> List[List[Port]]:

    cd_group = circuit.add_group("LEMMA_4_1_COMPUTE_DIFFS")
    cd_group.set_parent(parent_group)

    zero = constant_zero(circuit, y[0], parent_group=cd_group)

    diff_list = []

    # m_powers = [m]

    # for i in range(int(math.log2(n))):
    #    current_len = len(m_powers)
    #    current = m_powers[current_len - 1]
    #    next = one_left_shift(circuit, current, parent_group=cd_group)
    #    m_powers.append(next)

    for i in range(n):
        if i == 0:
            acc = [zero for k in range(n)]
        else:
            # summands = []
            # i_bits = int2binlist(i, bit_len=n)
            # for idx, bit in enumerate(i_bits):
            #    if bit:
            #        summands.append(m_powers[idx])
            summands = [m for _ in range(i)]
            acc = adder_tree_iterative(circuit, summands, zero, parent_group=cd_group)
        # acc = [zero for k in range(n)]
        # for _ in range(i):
        #    acc, _ = carry_look_ahead_adder(
        #        circuit, acc, m, zero, parent_group=cd_group
        #    )
        diff = subtract(circuit, y, acc, parent_group=cd_group)
        diff_list.append(diff)

    return diff_list


# check if its correct that we can generate the multiples via additions due to the fact that m <= n where n is the bit width of the number x in the lemma
def lemma_4_1_reduce_in_parallel(
    circuit: CircuitGraph,
    y: List[Port],
    m: List[Port],
    n: int,
    parent_group: Optional[Group] = None,
) -> List[Port]:

    rip_group = circuit.add_group("LEMMA_4_1_REDUCE_IN_PARALLEL")
    rip_group.set_parent(parent_group)

    zero = constant_zero(circuit, y[0], parent_group=rip_group)

    inter_list = []

    diff_list = lemma_4_1_compute_diffs(circuit, y, m, n, parent_group=rip_group)

    # is parallel in the circuit
    for i in range(n):
        # build acc starting at zero
        diff = diff_list[i]
        # check diff if its in the range of [0, m - 1]
        is_negative = diff[len(diff) - 1]
        is_positive = circuit.add_node(
            "not", "NOT", inputs=[is_negative], group_id=rip_group.id
        ).ports[1]
        less, equal, greater = n_bit_comparator(
            circuit, diff, m, parent_group=rip_group
        )
        # has to be less
        desired = circuit.add_node(
            "and", "DESIRED_AND", inputs=[is_positive, less], group_id=rip_group.id
        ).ports[2]
        not_desired = circuit.add_node(
            "not", "NOT", inputs=[desired], group_id=rip_group.id
        ).ports[1]
        inter = conditional_zeroing(circuit, diff, not_desired, parent_group=rip_group)
        inter_list.append(inter)

    result = []
    for i in range(n):
        reformat = []
        for entry in inter_list:
            reformat.append(entry[i])
        tree_res = or_tree_iterative(circuit, reformat, parent_group=rip_group)
        result.append(tree_res)
    return result


def lemma_4_1(
    circuit: CircuitGraph,
    x: List[Port],
    m: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:
    lemma_group = circuit.add_group("LEMMA_4_1")
    lemma_group.set_parent(parent_group)

    n = len(x)
    y = lemma_4_1_compute_y(circuit, x, m, parent_group=lemma_group)
    result = lemma_4_1_reduce_in_parallel(circuit, y, m, n, parent_group=lemma_group)
    return result


# compute y = sum i from 0 to n - 1: x_i * a_im; since x_i is one bit no multiplier needed
"""def compute_y_lemma_4_1(
    circuit: CircuitGraph,
    x: List[Port],
    aim: List[List[List[Port]]],
    parent_group: Optional[Group] = None,
):
    cy_group = circuit.add_group("COMPUTE_Y_FOR_LEMMA_4_1")
    cy_group.set_parent(parent_group)
    n = len(x)
    for i in range(n):
        not_x_i = circuit.add_node("not", "not_x_i", inputs=[x[i]], group_id=)
        aim_entry = 
        conditional_zeroing(cirucit, )
    return
"""


def theorem_4_2_precompute_lookup_is_prime_power(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    plipp_group = circuit.add_group(label="PRECOMPUTE_LOOKUP_IS_PRIME_POWER")
    plipp_group.set_parent(parent_group)

    result = []
    for i in range(1, n + 1):
        if is_prime_power(i):
            result.append(one)
        else:
            result.append(zero)
    return result


def theorem_4_2_precompute_lookup_p_l(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    plpl_group = circuit.add_group(label="PRECOMPUTE_LOOKUP_P_L")
    plpl_group.set_parent(parent_group)

    p_result = []
    l_result = []
    for i in range(0, n + 1):
        if is_prime_power(i) and i != 0:
            factorization = wheel_factorize(i)
            p = factorization[0]
            l = len(factorization)
            # FILL
            p_bits = int2binlist(p, bit_len=n)
            l_bits = int2binlist(l, bit_len=n)
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


def theorem_4_2_precompute_lookup_powers(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
) -> List[List[List[Port]]]:
    plp_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_POWERS")
    plp_group.set_parent(parent_group)

    result = []
    for p in range(1, n + 1):
        powers_of_p = []
        for e in range(n):
            if e > math.log2(n) or p**e > n:
                power = 0
            else:
                power = p**e
            power_bits = int2binlist(power, bit_len=n)
            power_ports = []
            for bit in power_bits:
                if bit:
                    power_ports.append(one)
                else:
                    power_ports.append(zero)
            powers_of_p.append(power_ports)
        result.append(powers_of_p)

    return result


def theorem_4_2_precompute_lookup_powers_decr(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    plp_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_POWERS_DECR")
    plp_group.set_parent(parent_group)

    result = []
    for p in range(1, n + 1):
        powers_of_p = []
        for e in range(n):
            if e > math.log2(n) or p**e > n:
                power = 0
            else:
                power = (p**e) - 1
            power_bits = int2binlist(power, bit_len=n)
            power_ports = []
            for bit in power_bits:
                if bit:
                    power_ports.append(one)
                else:
                    power_ports.append(zero)
            powers_of_p.append(power_ports)
        result.append(powers_of_p)

    return result


# WITH DUMMY ROW FOR SIMPLER INDEXING
def theorem_4_2_precompute_lookup_generator_powers(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_GENERATOR_POWERS")
    this_group.set_parent(parent_group)

    result = []
    software_primitive_roots = sb.find_primitive_roots(n)
    software_p_l_lookup = sb.theorem_4_2_precompute_lookup_p_l(n)

    dummy_row = []
    for i in range(n):
        dummy_row.append([zero for _ in range(n)])
    result.append(dummy_row)

    for pexpl_idx, pexpl in enumerate(range(1, n + 1)):
        row = []
        p, l = software_p_l_lookup[pexpl]
        if p == 0 or l == 0:
            thresh = 0
        else:
            thresh = int(math.pow(p, l)) - int(math.pow(p, l - 1))
        g = software_primitive_roots[pexpl_idx]
        software_pows_of_g = sb.compute_powers_mod_up_to(g, pexpl, thresh)
        while len(software_pows_of_g) < n:
            software_pows_of_g.append(0)

        # FILL ROW
        for entry in software_pows_of_g:
            bits = int2binlist(entry, bit_len=n)
            num_ports = []
            for bit in bits:
                if bit:
                    num_ports.append(one)
                else:
                    num_ports.append(zero)
            row.append(num_ports)
        result.append(row)
    return result


def theorem_4_2_precompute_lookup_tables_B(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_TABLES_B")
    this_group.set_parent(parent_group)
    # TABLE for a == 0
    TABLE_ZERO = []
    for l in range(0, int(math.log2(n)) + 1):
        row = []
        for b in range(0, n + 1):
            try:
                value = sanity.compute_a_b_l_formula(0, b, l)
            except:
                value = 0
            if value > 2**n - 1:
                value = 0
            value_bits = int2binlist(value, bit_len=n)

            num = []
            for bit in value_bits:
                if bit:
                    num.append(one)
                else:
                    num.append(zero)
            row.append(num)
        TABLE_ZERO.append(row)

    # TABLE for a == 1
    TABLE_ONE = []
    for l in range(0, int(math.log2(n)) + 1):
        row = []
        for b in range(0, n + 1):
            try:
                value = sanity.compute_a_b_l_formula(1, b, l)
            except:
                value = 0
            if value > (2**n) - 1:
                value = 0
            value_bits = int2binlist(value, bit_len=n)

            num = []
            for bit in value_bits:
                if bit:
                    num.append(one)
                else:
                    num.append(zero)
            row.append(num)
        TABLE_ONE.append(row)

    return TABLE_ZERO, TABLE_ONE


def theorem_4_2_step_1(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    p: List[Port],
    p_decr: List[Port],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
) -> List[List[Port]]:
    n = len(x_list)

    this_group = circuit.add_group("THEOREM_4_2_STEP_1")
    this_group.set_parent(parent_group)

    zero = constant_zero(circuit, p[0], parent_group=this_group)
    one = constant_one(circuit, p[0], parent_group=this_group)

    lookup_powers = theorem_4_2_precompute_lookup_powers(
        circuit, zero, one, len(x_list), parent_group=this_group
    )

    lookup_powers_decr = theorem_4_2_precompute_lookup_powers_decr(
        circuit, zero, one, len(x_list), parent_group=this_group
    )

    p_powers = tensor_multiplexer(circuit, lookup_powers, p_decr)
    p_powers_decr = tensor_multiplexer(circuit, lookup_powers_decr, p_decr)

    exponents = []

    for x in x_list:
        res_list = []
        for i in range(int(math.log2(n))):
            less, equal, greater = n_bit_comparator(
                circuit, p_powers[i], pexpl, parent_group=this_group
            )

            goreq = circuit.add_node(
                "not", "NOT", inputs=[less], group_id=this_group.id
            ).ports[1]

            remainder = lemma_4_1(circuit, x, p_powers[i], parent_group=this_group)

            is_remainder_not_zero = or_tree_iterative(
                circuit, remainder, parent_group=this_group
            )

            p_powers_is_not_zero = or_tree_iterative(
                circuit, p_powers[i], parent_group=this_group
            )
            p_powers_is_zero = circuit.add_node(
                "not", "NOT", inputs=[p_powers_is_not_zero], group_id=this_group.id
            ).ports[1]

            cond = circuit.add_node(
                "or",
                "OR",
                inputs=[is_remainder_not_zero, goreq],
                group_id=this_group.id,
            ).ports[2]

            cond = circuit.add_node(
                "or", "OR", inputs=[cond, p_powers_is_zero], group_id=this_group.id
            ).ports[2]

            exponent = generate_number(i, n, zero, one)

            res = conditional_zeroing(circuit, exponent, cond, parent_group=this_group)
            res_list.append(res)
        exponent = max_tree_iterative(circuit, res_list, parent_group=this_group)
        exponents.append(exponent)

    return exponents


def theorem_4_2_step_2(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    p: List[Port],
    p_decr: List[Port],
    j_list: List[List[Port]],
    parent_group: Optional[Group] = None,
):

    this_group = circuit.add_group("THEOREM_4_2_STEP_2")
    this_group.set_parent(parent_group)

    n = len(x_list)

    zero = constant_zero(circuit, p[0], parent_group=this_group)
    one = constant_one(circuit, p[0], parent_group=this_group)

    powers_lookup = theorem_4_2_precompute_lookup_powers(
        circuit, zero, one, n, parent_group=this_group
    )

    division_lookup = theorem_4_2_precompute_lookup_division(
        circuit, zero, one, n, parent_group=this_group
    )

    powers_of_p_bus = tensor_multiplexer(
        circuit, powers_lookup, p_decr, parent_group=this_group
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


# This is used for Theorem 4.2 Step 3, 6
def theorem_4_2_compute_sum(
    circuit: CircuitGraph,
    j_list: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_STEP_3")
    this_group.set_parent(parent_group)

    zero = constant_zero(circuit, j_list[0][0], parent_group=this_group)
    j = adder_tree_iterative(circuit, j_list, zero, parent_group=this_group)
    return j


def theorem_4_2_step_4(
    circuit: CircuitGraph,
    p: List[Port],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_STEP_4")
    this_group.set_parent(parent_group)

    n = len(p)

    zero = constant_zero(circuit, p[0], parent_group=this_group)
    one = constant_one(circuit, p[0], parent_group=this_group)

    # BUILD NUM TWO
    num_two = [zero for _ in range(n)]
    num_two[1] = one

    # BUILD NUM FOUR
    num_four = [zero for _ in range(n)]
    num_four[2] = one

    _, p_equals_two, _ = n_bit_comparator(circuit, p, num_two, parent_group=this_group)
    p_not_equals_two = circuit.add_node(
        "not", "NOT", inputs=[p_equals_two], group_id=this_group.id
    ).ports[1]

    _, pexpl_equals_two, _ = n_bit_comparator(
        circuit, pexpl, num_two, parent_group=this_group
    )
    _, pexpl_equals_four, _ = n_bit_comparator(
        circuit, pexpl, num_four, parent_group=this_group
    )

    two_or_four = circuit.add_node(
        "or", "OR", inputs=[pexpl_equals_two, pexpl_equals_four], group_id=this_group.id
    ).ports[2]

    flag = circuit.add_node(
        "or", "OR", inputs=[p_not_equals_two, two_or_four], group_id=this_group.id
    ).ports[2]

    return flag


def theorem_4_2_A_step_5(
    circuit: CircuitGraph,
    y_list: List[List[Port]],
    pexpl: List[Port],
    parent_group: Optional[Group] = None,
):

    this_group = circuit.add_group("THEOREM_4_2_A_STEP_5")
    this_group.set_parent(parent_group)

    n = len(y_list[0])

    zero = constant_zero(circuit, y_list[0][0], parent_group=this_group)
    one = constant_one(circuit, y_list[0][0], parent_group=this_group)

    generator_powers_lookup = theorem_4_2_precompute_lookup_generator_powers(
        circuit, zero, one, n, parent_group=this_group
    )

    generator_powers_bus = tensor_multiplexer(
        circuit, generator_powers_lookup, pexpl, parent_group=this_group
    )

    a_list = []

    supremum_num = len(generator_powers_bus)
    supremum = generate_number(supremum_num, n, zero, one)

    for y in y_list:
        e_candidates = []
        for idx, num in enumerate(generator_powers_bus):
            # check for equality with y
            # generate number from idx
            # pass only the correct idx number via conditional zeroing, max_tree_iterative
            equals = n_bit_equality(circuit, y, num, parent_group=this_group)
            # not_equals = circuit.add_node(
            #    "not", "NOT", inputs=[equals], group_id=this_group.id
            # ).ports[1]
            e_num = generate_number(idx, n, zero, one)
            # e_num = conditional_zeroing(
            #    circuit, e_num, not_equals, parent_group=this_group
            # )
            e_num = bus_multiplexer(
                circuit, [supremum, e_num], [equals], parent_group=this_group
            )
            e_candidates.append(e_num)
        a = min_tree_iterative(circuit, e_candidates, parent_group=this_group)
        a_list.append(a)
    return a_list


"""
# Circuit for a_hat = a mod (p^l - p^(l-1))
def theorem_4_2_A_step_7(
    circuit: CircuitGraph,
    a: List[Port],
    pexpl: List[Port],
    p_lookup: List[List[Port]],
    l_lookup: List[List[Port]],
    m_lookup: List[List[Port]],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("THEOREM_4_2_A_STEP_7")
    this_group.set_parent(parent_group)

    a_hat = lemma_4_1(circuit, a, m, m_decr, parent_group=this_group)
"""


def theorem_4_2_precompute_lookup_pexpl_minus_pexpl_minus_one(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group(
        "THEOREM_4_2_PRECOMPUTE_LOOKUP_PEXPL_MINUS_PEXPL_MINUS_ONE"
    )
    this_group.set_parent(parent_group)

    p_l_lookup = sb.theorem_4_2_precompute_lookup_p_l(n)

    table = []
    for pexpl in range(0, n + 1):
        p, l = p_l_lookup[pexpl]
        if l - 1 < 0:
            term = 0
        else:
            term = pexpl - p ** (l - 1)
        term_bits = int2binlist(term, bit_len=n)
        num = []
        for bit in term_bits:
            if bit:
                num.append(one)
            else:
                num.append(zero)
        table.append(num)
    return table


# WITH INDEX PADDING -> NO NEED FOR VARIABLE_DECR
def theorem_4_2_precompute_lookup_division(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_4_2_PRECOMPUTE_LOOKUP_DIVISION")
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
                value_bits = int2binlist(value, bit_len=n)
                for bit in value_bits:
                    if bit:
                        ports.append(one)
                    else:
                        ports.append(zero)
            row.append(ports)
        table.append(row)
    return table

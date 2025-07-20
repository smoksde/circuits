from typing import List, Optional
from graph import *
from .constants import *
from utils import int2binlist
from .multipliers import *
from .manipulators import conditional_zeroing
from .multiplexers import tensor_multiplexer
from .adders import carry_look_ahead_adder
from .trees import adder_tree_iterative, or_tree_iterative
from .subtractors import subtract
from .comparators import n_bit_comparator


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
) -> Port:
    pa_group = circuit.add_group("PRE_AIM")
    pa_group.set_parent(parent_group)

    lis = []

    for m in range(1, n + 1):
        fix_m_entries = []
        for i in range(n):
            aim_entry = []
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

    for i in range(n):
        if i == 0:
            acc = [zero for k in range(n)]
        else:
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
    m_decr: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:
    lemma_group = circuit.add_group("LEMMA_4_1")
    lemma_group.set_parent(parent_group)

    n = len(x)
    print("n: ", n)
    y = lemma_4_1_compute_y(circuit, x, m_decr, parent_group=lemma_group)
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


def wheel_factorize(n: int):
    factorization = []
    while n % 2 == 0:
        factorization.append(2)
        n //= 2
    d = 3
    while d * d <= n:
        while n % d == 0:
            factorization.append(d)
            n //= d
        d += 2
    if n > 1:
        factorization.append(n)
    return factorization


def is_prime_power(n: int):
    factors = wheel_factorize(n)
    return len(set(factors)) == 1

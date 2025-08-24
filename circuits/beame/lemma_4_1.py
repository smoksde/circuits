from typing import List, Optional, Tuple
from graph import *
from utils import int2binlist
from asserts import *

from ..constants import *
from ..multiplexers import tensor_multiplexer
from ..manipulators import conditional_zeroing
from ..trees import adder_tree_iterative, or_tree_iterative
from ..comparators import n_bit_comparator
from ..subtractors import subtract


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
    this_group = circuit.add_group("PRE_AIM")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

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


def provide_aims_given_m(
    circuit: CircuitGraph, m: List[Port], parent_group: Optional[Group] = None
):
    this_group = circuit.add_group("LEMMA_4_1_PROVIDE_AIMS_GIVEN_M")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    zero = constant_zero(circuit, m[0], parent_group=this_group)
    one = constant_one(circuit, m[0], parent_group=this_group)
    n = len(m)
    aims_ports = precompute_aim(circuit, zero, one, n, parent_group=this_group)
    # figure out if aims_ports needs to be reformated
    ais = tensor_multiplexer(circuit, aims_ports, m, parent_group=this_group)
    return ais  # List[List[Port]]


# computes x_i * a_im terms in the big summation
def compute_summands(
    circuit: CircuitGraph,
    x: List[Port],
    aims: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("LEMMA_4_1_COMPUTE_SUMMANDS")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    n = len(x)
    result = []
    for p1, p2 in zip(x, aims):
        not_p1 = circuit.add_node(
            "not", "NOT", inputs=[p1], group_id=this_group_id
        ).ports[1]
        product = conditional_zeroing(circuit, p2, not_p1, parent_group=this_group)
        result.append(product)
    return result


def compute_y(
    circuit: CircuitGraph,
    x: List[Port],
    m: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:
    this_group = circuit.add_group("LEMMA_4_1_COMPUTE_Y")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    zero = constant_zero(circuit, x[0], parent_group=this_group)

    aims = provide_aims_given_m(circuit, m, parent_group=this_group)
    summands = compute_summands(circuit, x, aims, parent_group=this_group)
    sum = adder_tree_iterative(circuit, summands, zero, parent_group=this_group)

    assert_list_of_ports(sum)
    return sum


def compute_diffs(
    circuit: CircuitGraph,
    y: List[Port],
    m: List[Port],
    n: int,
    parent_group: Optional[Group] = None,
) -> List[List[Port]]:

    this_group = circuit.add_group("LEMMA_4_1_COMPUTE_DIFFS")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    zero = constant_zero(circuit, y[0], parent_group=this_group)

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
            acc = adder_tree_iterative(circuit, summands, zero, parent_group=this_group)
        # acc = [zero for k in range(n)]
        # for _ in range(i):
        #    acc, _ = carry_look_ahead_adder(
        #        circuit, acc, m, zero, parent_group=cd_group
        #    )
        diff = subtract(circuit, y, acc, parent_group=this_group)
        diff_list.append(diff)

    return diff_list


# check if its correct that we can generate the multiples via additions due to the fact that m <= n where n is the bit width of the number x in the lemma
def reduce_in_parallel(
    circuit: CircuitGraph,
    y: List[Port],
    m: List[Port],
    n: int,
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("LEMMA_4_1_REDUCE_IN_PARALLEL")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    zero = constant_zero(circuit, y[0], parent_group=this_group)

    inter_list = []

    diff_list = compute_diffs(circuit, y, m, n, parent_group=this_group)

    # is parallel in the circuit
    for i in range(n):
        # build acc starting at zero
        diff = diff_list[i]
        # check diff if its in the range of [0, m - 1]
        is_negative = diff[len(diff) - 1]
        is_positive = circuit.add_node(
            "not", "NOT", inputs=[is_negative], group_id=this_group_id
        ).ports[1]
        less, equal, greater = n_bit_comparator(
            circuit, diff, m, parent_group=this_group
        )
        # has to be less
        desired = circuit.add_node(
            "and", "DESIRED_AND", inputs=[is_positive, less], group_id=this_group_id
        ).ports[2]
        not_desired = circuit.add_node(
            "not", "NOT", inputs=[desired], group_id=this_group_id
        ).ports[1]
        inter = conditional_zeroing(circuit, diff, not_desired, parent_group=this_group)
        inter_list.append(inter)

    result = []
    for i in range(n):
        reformat = []
        for entry in inter_list:
            reformat.append(entry[i])
        tree_res = or_tree_iterative(circuit, reformat, parent_group=this_group)
        result.append(tree_res)
    return result


def lemma_4_1(
    circuit: CircuitGraph,
    x: List[Port],
    m: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:
    this_group = circuit.add_group("LEMMA_4_1")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(x)
    y = compute_y(circuit, x, m, parent_group=this_group)
    result = reduce_in_parallel(circuit, y, m, n, parent_group=this_group)
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

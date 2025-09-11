from typing import List, Optional, Tuple
from core.graph import *
from utils import int2binlist

from ..standard.constants import *
from ..standard.multiplexers import tensor_multiplexer
from ..standard.manipulators import conditional_zeroing
from ..standard.trees import adder_tree_iterative, or_tree_iterative
from ..standard.comparators import n_bit_comparator
from ..standard.subtractors import subtract


# Lookup Table generation for Lemma 4.1
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
    ais = tensor_multiplexer(circuit, aims_ports, m, parent_group=this_group)
    return ais


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
        not_p1 = not_gate(circuit, p1, parent_group=this_group)
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

    for i in range(n):
        if i == 0:
            acc = [zero for k in range(n)]
        else:
            summands = [m for _ in range(i)]
            acc = adder_tree_iterative(circuit, summands, zero, parent_group=this_group)
        diff = subtract(circuit, y, acc, parent_group=this_group)
        diff_list.append(diff)

    return diff_list


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

    for i in range(n):
        diff = diff_list[i]
        is_negative = diff[len(diff) - 1]
        is_positive = not_gate(circuit, is_negative, parent_group=this_group)
        less, equal, greater = n_bit_comparator(
            circuit, diff, m, parent_group=this_group
        )
        desired = and_gate(circuit, [is_positive, less], parent_group=this_group)
        not_desired = not_gate(circuit, desired, parent_group=this_group)
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

from typing import Tuple, Optional, List

from graph import *
from . import lemma_4_1
from . import theorem_4_2
from . import lemma_5_1
from . import theorem_5_3

from ..constants import constant_zero, constant_one


def step_3(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    c_list: List[List[Port]],
    parent_group: Optional[Group] = None,
):

    this_group = circuit.add_group("THEOREM_5_2_STEP_3")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(x_list)
    s = len(c_list)

    matrix = []
    for j in range(s):
        row = []
        for i in range(n):
            b_j_i = lemma_4_1.lemma_4_1(
                circuit, x_list[i], c_list[j], parent_group=this_group
            )
            row.append(b_j_i)
        matrix.append(row)
    return matrix


def step_4(
    circuit: CircuitGraph,
    b_j_i_matrix: List[List[List[Port]]],
    c_list: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_5_2_STEP_4")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    b_j_list = []
    for idx, x_list in enumerate(b_j_i_matrix):
        b_j = theorem_4_2.theorem_4_2_for_theorem_5_2(
            circuit, x_list, c_list[idx], parent_group=this_group
        )
        b_j_list.append(b_j)
    return b_j_list


def step_5(
    circuit: CircuitGraph,
    b_j_list: List[List[Port]],
    c: List,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_5_2_STEP_5")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    result = lemma_5_1.lemma_5_1(circuit, b_j_list, c, parent_group=this_group)
    return result


def theorem_5_2(
    circuit: CircuitGraph,
    x_list: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("THEOREM_5_2")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(x_list)
    big_n = n * n

    zero = constant_zero(circuit, x_list[0][0], parent_group=this_group)
    one = constant_one(circuit, x_list[0][0], parent_group=this_group)

    # For step 3 x_list has to be extended to big_n
    for x in x_list:
        while len(x) < big_n:
            x.append(zero)

    c_list, c = theorem_5_3.precompute_good_modulus_sequence(
        circuit, zero, one, big_n, parent_group=this_group
    )

    # does a lot of lemma 4.1 in parallel
    b_j_i_matrix = step_3(circuit, x_list, c_list, parent_group=this_group)

    # does theorem 4.2 in parallel
    b_j_list = step_4(circuit, b_j_i_matrix, c_list, parent_group=this_group)

    # does lemma 5.1
    result = step_5(circuit, b_j_list, c, parent_group=this_group)
    return result

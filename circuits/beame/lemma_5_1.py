from typing import List, Optional, Tuple

from core.graph import *

from ..gates import *
from ..multipliers import wallace_tree_multiplier
from ..trees import adder_tree_iterative
from ..constants import constant_zero, constant_one
from ..subtractors import subtract
from ..manipulators import conditional_zeroing
from ..comparators import n_bit_comparator
import sanity.theorem_5_3_sanity as theorem_5_3_sanity
import sanity.lemma_5_1_sanity as lemma_5_1_sanity

from .. import circuit_utils

import math

# Usually Lemma 5.1 also consists of steps 1 - 4.
# Though assuming only contextual usage of Lemma 5.1
# in Theorem 5.2 with the predefined good modulus
# sequences (first n prime numbers) steps 1 - 4
# can be collapsed to a precomputation of u_i for i = 1, ..., n


# Heavy use of software precomputation that is already verified via tests
def precompute_u_list(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
) -> List[List[Port]]:

    this_group = circuit.add_group("LEMMA_5_1_PRECOMPUTE_U_LIST")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    c_list, c = theorem_5_3_sanity.compute_good_modulus_sequence(n)
    v_list = lemma_5_1_sanity.step_2(c_list, c)
    w_list = lemma_5_1_sanity.step_3(v_list, c_list)
    u_list = lemma_5_1_sanity.step_4(v_list, w_list)

    U_LIST = []
    for u_i in u_list:
        U_I = circuit_utils.generate_number(
            u_i, n * n, zero, one, parent_group=this_group
        )
        U_LIST.append(U_I)
    return U_LIST


# Expects inputs having already n = n^2 bits
def step_5(
    circuit: CircuitGraph,
    x_mod_c_i_list: List[List[Port]],
    u_list: List[List[Port]],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("LEMMA_5_1_STEP_5")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(u_list[0])

    zero = constant_zero(circuit, u_list[0][0], parent_group=this_group)

    summands = []
    for x_mod_c_i, u_i in zip(x_mod_c_i_list, u_list):
        summand = wallace_tree_multiplier(
            circuit, x_mod_c_i, u_i, parent_group=this_group
        )
        summand = summand[:n]
        summands.append(summand)
    y = adder_tree_iterative(circuit, summands, zero, parent_group=this_group)
    return y


# Expects inputs having already n = n^2 bits
def step_6_and_7(
    circuit: CircuitGraph,
    y: List[Port],
    c: List[Port],
    zero: Port,
    one: Port,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("LEMMA_5_1_STEP_6")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n = len(y)

    # Compute c_n given the strict assumptions
    c_list, _ = theorem_5_3_sanity.compute_good_modulus_sequence(int(math.sqrt(n)))

    inter_list = []

    for t in range(0, n * c_list[-1] + 1):
        t_ports = circuit_utils.generate_number(
            t, n, zero, one, parent_group=this_group
        )
        prod = wallace_tree_multiplier(circuit, t_ports, c, parent_group=this_group)
        prod = prod[:n]
        y_t = subtract(circuit, y, prod, parent_group=this_group)

        # check negativity
        is_negative = y_t[len(y_t) - 1]

        less, _, _ = n_bit_comparator(circuit, y_t, c, parent_group=this_group)

        #not_less = circuit.add_node(
        #    "not", "NOT", inputs=[less], group_id=this_group_id
        #).ports[1]

        #not_desired = circuit.add_node(
        #    "or", label="OR", inputs=[not_less, is_negative], group_id=this_group_id
        #).ports[2]

        not_less = not_gate(circuit, less, parent_group=this_group)
        not_desired = or_gate(circuit, [not_less, is_negative], parent_group=this_group)

        # conditional subtract
        inter = conditional_zeroing(circuit, y_t, not_desired, parent_group=this_group)
        inter_list.append(inter)

    result = adder_tree_iterative(circuit, inter_list, zero, parent_group=this_group)
    return result


def lemma_5_1(
    circuit: CircuitGraph,
    x_mod_c_i_list: List[List[Port]],
    c: List[Port],
    parent_group: Optional[Group] = None,
) -> List[Port]:

    this_group = circuit.add_group("LEMMA_5_1")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    big_n = len(c)
    n = int(math.sqrt(big_n))

    zero = constant_zero(circuit, c[0], parent_group=this_group)
    one = constant_one(circuit, c[0], parent_group=this_group)

    u_list = precompute_u_list(circuit, zero, one, n, parent_group=this_group)

    #print(len(x_mod_c_i_list[0]))
    #print(len(u_list[0]))
    y = step_5(circuit, x_mod_c_i_list, u_list, parent_group=this_group)

    result = step_6_and_7(circuit, y, c, zero, one, parent_group=this_group)
    return result

from typing import List, Optional, Tuple
from graph import *

from ..multipliers import wallace_tree_multiplier
from ..trees import adder_tree_iterative
from ..constants import constant_zero
import theorem_5_3_sanity
import lemma_5_1_sanity

from .. import circuit_utils

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
    this_group.set_parent(parent_group)

    c_list, c = theorem_5_3_sanity.compute_good_modulus_sequence(n)
    v_list = lemma_5_1_sanity.step_2(c_list, c)
    w_list = lemma_5_1_sanity.step_3(v_list, c_list)
    u_list = lemma_5_1_sanity.step_4(v_list, w_list)

    U_LIST = []
    for u_i in u_list:
        U_I = circuit_utils.generate_number(u_i, n, zero, one, parent_group=this_group)
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

from typing import List, Optional

from core.graph import *
from .gates import *
from .comparators import n_bit_comparator, n_bit_equality
from .multiplexers import multiplexer, bus_multiplexer
from ..circuit_utils import generate_number
from .constants import constant_zero, constant_one


def conditional_zeroing(circuit, x_list, cond, parent_group=None):
    this_group = circuit.add_group("CONDITIONAL_ZEROING")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    ports = []
    not_cond_node = not_gate(circuit, cond, parent_group=this_group)
    not_cond_port = not_cond_node
    for x in x_list:
        and_out = and_gate(circuit, [x, not_cond_port], parent_group=this_group)
        ports.append(and_out)
    return ports


def max_tree_iterative(
    circuit: CircuitGraph,
    values: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("MAX_TREE_ITERATIVE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    current = values
    while len(current) > 1:
        next = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                a = current[i]
                b = current[i + 1]
                less, _, _ = n_bit_comparator(circuit, a, b, parent_group=this_group)
                result = bus_multiplexer(circuit, [a, b], [less])
                next.append(result)
            else:
                next.append(current[i])
        current = next
    return current[0]


def min_tree_iterative(
    circuit: CircuitGraph,
    values: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("MIN_TREE_ITERATIVE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    current = values
    while len(current) > 1:
        next = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                a = current[i]
                b = current[i + 1]
                _, _, greater = n_bit_comparator(circuit, a, b, parent_group=this_group)
                result = bus_multiplexer(circuit, [a, b], [greater])
                next.append(result)
            else:
                next.append(current[i])
        current = next
    return current[0]


def smallest_non_zero_tree_iterative(
    circuit: CircuitGraph,
    values: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("SMALLEST_NON_ZERO_TREE_ITERATIVE")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    zero = constant_zero(circuit, values[0][0], parent_group=this_group)
    one = constant_one(circuit, values[0][0], parent_group=this_group)

    zero_num = generate_number(0, len(values[0]), zero, one)

    current = values

    while len(current) > 1:
        next = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                a = current[i]
                b = current[i + 1]
                _, _, greater = n_bit_comparator(circuit, a, b, parent_group=this_group)
                # not_greater = circuit.add_node(
                #    "not", "NOT", inputs=[greater], group_id=this_group_id
                # ).ports[1]
                not_greater = not_gate(circuit, greater, parent_group=this_group)
                smaller_one = bus_multiplexer(circuit, [a, b], [greater])
                equals_zero = n_bit_equality(
                    circuit,
                    smaller_one,
                    zero_num,
                    parent_group=this_group,
                )
                selector = multiplexer(
                    circuit,
                    [greater, not_greater],
                    [equals_zero],
                    parent_group=this_group,
                )
                chosen_one = bus_multiplexer(circuit, [a, b], [selector])
                next.append(chosen_one)
            else:
                next.append(current[i])
        current = next
    return current[0]

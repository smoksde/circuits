from typing import List, Optional

from core.graph import *
from .adders import ripple_carry_adder, carry_look_ahead_adder
from circuits.standard.gates import *


def or_tree_recursive(
    circuit: CircuitGraph, input_list: List[Port], parent_group: Optional[Group] = None
) -> Port:
    this_group = circuit.add_group("OR_TREE_RECURSIVE")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    if len(input_list) == 1:
        return input_list[0]
    if len(input_list) == 2:
        sec_or_out = or_gate(circuit, input_list, parent_group=this_group)
        return sec_or_out
    mid = len(input_list) // 2
    left = or_tree_recursive(circuit, input_list[:mid], parent_group=this_group)
    right = or_tree_recursive(circuit, input_list[mid:], parent_group=this_group)
    or_out = or_gate(circuit, [left, right], parent_group=this_group)
    return or_out


def and_tree_iterative(
    circuit: CircuitGraph, input_list: List[Port], parent_group: Optional[Group] = None
) -> Port:
    this_group = circuit.add_group("AND_TREE_ITERATIVE")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    current = input_list
    while len(current) > 1:
        next = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                m = and_gate(
                    circuit, [current[i], current[i + 1]], parent_group=this_group
                )
                next.append(m)
            else:
                next.append(current[i])
        current = next
    return current[0]


def or_tree_iterative(
    circuit: CircuitGraph, input_list: List[Port], parent_group: Optional[Group] = None
) -> Port:
    this_group = circuit.add_group("OR_TREE_ITERATIVE")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    current = input_list
    while len(current) > 1:
        next = []
        for i in range(0, len(current), 2):

            if i + 1 < len(current):
                m = or_gate(
                    circuit, [current[i], current[i + 1]], parent_group=this_group
                )
                next.append(m)
            else:
                next.append(current[i])
        current = next
    return current[0]


def adder_tree_iterative(
    circuit: CircuitGraph,
    summands: List[List[Port]],
    zero: Port,
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("ADDER_TREE_ITERATIVE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    current = summands
    while len(current) > 1:
        next = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                sum, _ = carry_look_ahead_adder(
                    circuit, current[i], current[i + 1], zero, parent_group=this_group
                )
                next.append(sum)
            else:
                next.append(current[i])
        current = next
    return current[0]


def adder_tree_recursive(
    circuit, summand_lists, zero, parent_group: Optional[Group] = None
):
    this_group = circuit.add_group("ADDER_TREE_RECURSIVE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    if len(summand_lists) == 1:
        return summand_lists[0], zero

    if len(summand_lists) == 2:
        sums, carry = ripple_carry_adder(
            circuit, summand_lists[0], summand_lists[1], zero
        )
        return sums, carry

    mid = len(summand_lists) // 2
    left_sums, left_carry = adder_tree_recursive(circuit, summand_lists[:mid], zero)
    right_sums, right_carry = adder_tree_recursive(circuit, summand_lists[mid:], zero)
    sums, carry = ripple_carry_adder(circuit, left_sums, right_sums, zero)
    return sums, carry

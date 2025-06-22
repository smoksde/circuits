from typing import List, Optional

from graph import *


def and_tree_recursive(circuit, input_list, parent_group=None):
    atr_group = circuit.add_group("AND_TREE_RECURSIVE")
    atr_group.set_parent(parent_group)
    if len(input_list) == 1:
        return input_list[0]

    if len(input_list) == 2:
        and_node = circuit.add_node(
            "and", "AND", inputs=input_list, group_id=atr_group.id
        )
        return and_node.ports[2]

    mid = len(input_list) // 2
    left = and_tree_recursive(circuit, input_list[:mid], parent_group=atr_group)
    right = and_tree_recursive(circuit, input_list[mid:], parent_group=atr_group)
    and_node = circuit.add_node(
        "and", "AND", inputs=[left, right], group_id=atr_group.id
    )
    return and_node.ports[2]


def or_tree_recursive(
    circuit: CircuitGraph, input_list: List[Port], parent_group: Optional[Group] = None
) -> Port:
    otr_group = circuit.add_group("OR_TREE_RECURSIVE")
    otr_group.set_parent(parent_group)
    if len(input_list) == 1:
        return input_list[0]
    if len(input_list) == 2:
        or_node = circuit.add_node("or", "OR", inputs=input_list, group_id=otr_group.id)
        return or_node.ports[2]
    mid = len(input_list) // 2
    left = or_tree_recursive(circuit, input_list[:mid], parent_group=otr_group)
    right = or_tree_recursive(circuit, input_list[mid:], parent_group=otr_group)
    or_node = circuit.add_node("or", "OR", inputs=[left, right], group_id=otr_group.id)
    return or_node.ports[2]


def and_tree_iterative(
    circuit: CircuitGraph, input_list: List[Port], parent_group: Optional[Group] = None
) -> Port:
    ati_group = circuit.add_group("AND_TREE_ITERATIVE")
    ati_group.set_parent(parent_group)
    current = input_list
    while len(current) > 1:
        next = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                m = circuit.add_node(
                    "and",
                    "AND",
                    inputs=[current[i], current[i + 1]],
                    group_id=ati_group.id,
                )
                next.append(m.ports[2])
            else:
                next.append(current[i])
        current = next
    return current[0]


def or_tree_iterative(
    circuit: CircuitGraph, input_list: List[Port], parent_group: Optional[Group] = None
) -> Port:
    oti_group = circuit.add_group("OR_TREE_ITERATIVE")
    oti_group.set_parent(parent_group)
    current = input_list
    while len(current) > 1:
        next = []
        for i in range(0, len(current), 2):

            if i + 1 < len(current):
                m = circuit.add_node(
                    "or",
                    "OR",
                    inputs=[current[i], current[i + 1]],
                    group_id=oti_group.id,
                )
                next.append(m.ports[2])
            else:
                next.append(current[i])
        current = next
    return current[0]

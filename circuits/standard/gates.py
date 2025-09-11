from typing import Optional
from core.interface import DepthInterface
from core.graph import Group


def and_gate(circuit, inputs, label=None, parent_group: Optional[Group] = None):

    this_group = circuit.add_group("AND_GATE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    this_group_id = this_group.id if this_group is not None else -1

    node = circuit.add_node(
        "and", label or "AND", inputs=inputs, group_id=this_group_id
    )

    if isinstance(circuit, DepthInterface):
        return node
    else:
        return node.ports[2]


def or_gate(circuit, inputs, label=None, parent_group: Optional[Group] = None):

    this_group = circuit.add_group("OR_GATE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    this_group_id = this_group.id if this_group is not None else -1

    node = circuit.add_node("or", label or "OR", inputs=inputs, group_id=this_group_id)
    if isinstance(circuit, DepthInterface):
        return node
    else:
        return node.ports[2]


def xor_gate(circuit, inputs, label=None, parent_group: Optional[Group] = None):

    this_group = circuit.add_group("XOR_GATE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    this_group_id = this_group.id if this_group is not None else -1

    node = circuit.add_node(
        "xor", label or "XOR", inputs=inputs, group_id=this_group_id
    )
    if isinstance(circuit, DepthInterface):
        return node
    else:
        return node.ports[2]


def not_gate(circuit, input_port, label=None, parent_group: Optional[Group] = None):

    this_group = circuit.add_group("NOT_GATE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    this_group_id = this_group.id if this_group is not None else -1

    node = circuit.add_node(
        "not", label or "NOT", inputs=[input_port], group_id=this_group_id
    )
    if isinstance(circuit, DepthInterface):
        return node
    else:
        return node.ports[1]

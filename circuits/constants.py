from graph import *
from typing import Optional


def constant_zero(
    circuit: CircuitGraph, in_port: Port, parent_group: Optional[Group] = None
) -> Port:
    cz_group = circuit.add_group("CONSTANT_ZERO")
    cz_group.set_parent(parent_group)
    not_in = circuit.add_node("not", "NOT", inputs=[in_port], group_id=cz_group.id)
    not_in_port = not_in.ports[1]
    zero_node = circuit.add_node(
        "and", "ZERO_AND", inputs=[in_port, not_in_port], group_id=cz_group.id
    )
    zero_port = zero_node.ports[2]
    return zero_port


def constant_one(
    circuit: CircuitGraph, in_port: Port, parent_group: Optional[Group] = None
) -> Port:
    co_group = circuit.add_group("CONSTANT_ONE")
    co_group.set_parent(parent_group)
    not_in = circuit.add_node("not", "NOT", inputs=[in_port], group_id=co_group.id)
    not_in_port = not_in.ports[1]
    one_node = circuit.add_node(
        "or", "ONE_OR", inputs=[in_port, not_in_port], group_id=co_group.id
    )
    one_port = one_node.ports[2]
    return one_port

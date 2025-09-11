from core.graph import *
from typing import Optional
from circuits.standard.gates import *


def constant_zero(
    circuit: CircuitGraph, in_port: Port, parent_group: Optional[Group] = None
) -> Port:
    this_group = circuit.add_group("CONSTANT_ZERO")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    not_in_port = not_gate(circuit, in_port, parent_group=this_group)
    zero_port = and_gate(circuit, [in_port, not_in_port], parent_group=this_group)
    return zero_port


def constant_one(
    circuit: CircuitGraph, in_port: Port, parent_group: Optional[Group] = None
) -> Port:
    this_group = circuit.add_group("CONSTANT_ONE")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    not_in_port = not_gate(circuit, in_port, parent_group=this_group)
    one_port = or_gate(circuit, [in_port, not_in_port], parent_group=this_group)
    return one_port

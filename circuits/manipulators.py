from typing import List, Optional

from graph import *
from .comparators import n_bit_comparator
from .multiplexers import bus_multiplexer


# if condition is one then zero it
def conditional_zeroing(circuit, x_list, cond, parent_group=None):
    cz_group = circuit.add_group("CONDITIONAL_ZEROING")
    cz_group.set_parent(parent_group)
    ports = []
    not_cond_node = circuit.add_node("not", "NOT", inputs=[cond], group_id=cz_group.id)
    not_cond_port = not_cond_node.ports[1]
    for x in x_list:
        and_node = circuit.add_node(
            "and", "AND", inputs=[x, not_cond_port], group_id=cz_group.id
        )
        ports.append(and_node.ports[2])
    return ports


def max_tree_iterative(
    circuit: CircuitGraph,
    values: List[List[Port]],
    parent_group: Optional[Group] = None,
):
    this_group = circuit.add_group("MAX_TREE_ITERATIVE")
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

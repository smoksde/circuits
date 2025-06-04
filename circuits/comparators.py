from .trees import *
from .utils import *


def one_bit_comparator(circuit, x, y, parent_group=None):
    obc_group = circuit.add_group("ONE_BIT_COMPARATOR")
    obc_group.set_parent(parent_group)
    not_x = circuit.add_node("not", "NOT", inputs=[x], group_id=obc_group.id)
    not_y = circuit.add_node("not", "NOT", inputs=[y], group_id=obc_group.id)
    x_less_y = circuit.add_node(
        "and", "AND", inputs=[not_x.ports[1], y], group_id=obc_group.id
    )
    x_greater_y = circuit.add_node(
        "and", "AND", inputs=[x, not_y.ports[1]], group_id=obc_group.id
    )
    x_equals_y = xnor_gate(
        circuit, x_less_y.ports[2], x_greater_y.ports[2], parent_group=obc_group
    )
    return x_less_y.ports[2], x_equals_y, x_greater_y.ports[2]


def n_bit_comparator(circuit, x_list, y_list, parent_group=None):
    assert len(x_list) == len(y_list), "Both inputs have to be equally long"
    n = len(x_list)
    nbc_group = circuit.add_group("N_BIT_COMPARATOR")
    nbc_group.set_parent(parent_group)
    # bit wise results
    bit_wise = []
    for i in range(n):
        less, equals, greater = one_bit_comparator(
            circuit, x_list[i], y_list[i], parent_group=nbc_group
        )
        bit_wise.append((less, equals, greater))

    # and gates for n bit equals
    index = n - 1
    pre_port = bit_wise[index][1]  # msb equals port
    build_equals = []
    build_equals.append(pre_port)
    while index > 0:
        curr_equals = circuit.add_node(
            "and",
            "AND",
            inputs=[pre_port, bit_wise[index - 1][1]],
            group_id=nbc_group.id,
        )
        build_equals.append(curr_equals.ports[2])
        pre_port = curr_equals.ports[2]
        index -= 1
    n_equals = pre_port

    index = n - 1
    build_less = []
    build_less.append(bit_wise[index][0])  # append msb less port
    while index > 0:
        curr_less_node = circuit.add_node(
            "and",
            "AND",
            inputs=[build_equals[n - 1 - index], bit_wise[index - 1][0]],
            group_id=nbc_group.id,
        )
        build_less.append(curr_less_node.ports[2])
        index -= 1

    # build or tree for n bit less
    n_less = or_tree_recursive(circuit, build_less, parent_group=nbc_group)

    index = n - 1
    build_greater = []
    build_greater.append(bit_wise[index][2])
    while index > 0:
        curr_greater_node = circuit.add_node(
            "and",
            "AND",
            inputs=[build_equals[n - 1 - index], bit_wise[index - 1][2]],
            group_id=nbc_group.id,
        )
        build_greater.append(curr_greater_node.ports[2])
        index -= 1

    n_greater = or_tree_recursive(circuit, build_greater, parent_group=nbc_group)

    return n_less, n_equals, n_greater

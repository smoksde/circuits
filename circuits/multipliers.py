from .adders import *
from .constants import *


def wallace_tree_multiplier(circuit, x_list, y_list, parent_group=None):
    wtm_group = circuit.add_group("WALLACE_TREE_MULTIPLIER")
    wtm_group.set_parent(parent_group)
    assert len(x_list) == len(y_list), "Both inputs have to be equally long"
    n = len(x_list)
    partial_products = [[] for _ in range(2 * n)]

    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            index = i + j
            node = circuit.add_node("and", "AND", inputs=[x, y], group_id=wtm_group.id)
            partial_products[index].append(node.ports[2])

    while any(len(col) > 2 for col in partial_products):
        new_products = [[] for _ in range(2 * n)]
        for i in range(2 * n):
            col = partial_products[i]
            j = 0
            while (
                len(col) > j + 2
            ):  # as long as 3 or more partial products remain in the current column
                sum, cout = full_adder(
                    circuit, col[j], col[j + 1], col[j + 2], parent_group=wtm_group
                )
                new_products[i].append(sum)
                new_products[i + 1].append(cout)
                j += 3

            if (
                len(col) > j + 1 and j > 0
            ):  # if 2 partial products remain in the current column and j > 0
                sum, cout = half_adder(
                    circuit, col[j], col[j + 1], parent_group=wtm_group
                )
                new_products[i].append(sum)
                new_products[i + 1].append(cout)
                j += 2

            while len(col) > j:
                new_products[i].append(col[j])
                j += 1

        partial_products = new_products

    zero_port = constant_zero(circuit, x_list[0], parent_group=wtm_group)

    # Applying fast adder since all columns have at most 2 partial products
    x_addend = []
    y_addend = []

    for i in range(2 * n):
        if len(partial_products[i]) == 2:
            x_addend.append(partial_products[i][0])
            y_addend.append(partial_products[i][1])
        if len(partial_products[i]) == 1:
            x_addend.append(partial_products[i][0])
            y_addend.append(zero_port)

    sum_outputs, carry = carry_look_ahead_adder(
        circuit, x_addend, y_addend, zero_port, parent_group=wtm_group
    )

    outputs = sum_outputs
    outputs.append(carry)

    return outputs

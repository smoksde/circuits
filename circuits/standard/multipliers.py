from .adders import *
from .constants import *


def wallace_tree_multiplier(circuit, x_list, y_list, parent_group=None):
    this_group = circuit.add_group("WALLACE_TREE_MULTIPLIER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    assert len(x_list) == len(y_list), "Both inputs have to be equally long"
    n = len(x_list)

    partial_products = [[] for _ in range(2 * n + 1)]

    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            index = i + j
            and_out = and_gate(circuit, inputs=[x, y], parent_group=this_group)
            partial_products[index].append(and_out)

    while any(len(col) > 2 for col in partial_products):

        m = len(partial_products)
        new_products = [[] for _ in range(m + 1)]

        for i in range(m):
            col = partial_products[i]
            j = 0

            while len(col) - j >= 3:
                s, cout = full_adder(
                    circuit, col[j], col[j + 1], col[j + 2], parent_group=this_group
                )
                new_products[i].append(s)
                new_products[i + 1].append(cout)
                j += 3

            if len(col) - j == 2:
                s, cout = half_adder(
                    circuit, col[j], col[j + 1], parent_group=this_group
                )
                new_products[i].append(s)
                new_products[i + 1].append(cout)
                j += 2

            if len(col) - j == 1:
                new_products[i].append(col[j])

        while len(new_products) > 1 and not new_products[-1]:
            new_products.pop()

        partial_products = new_products

    if len(partial_products) > 2 * n:
        for k in range(2 * n, len(partial_products)):
            partial_products[2 * n - 1].extend(partial_products[k])

    partial_products = partial_products[: 2 * n]
    while len(partial_products) < 2 * n:
        partial_products.append([])

    zero_port = constant_zero(circuit, x_list[0], parent_group=this_group)

    x_addend = []
    y_addend = []

    for i in range(2 * n):
        if len(partial_products[i]) == 2:
            x_addend.append(partial_products[i][0])
            y_addend.append(partial_products[i][1])
        elif len(partial_products[i]) == 1:
            x_addend.append(partial_products[i][0])
            y_addend.append(zero_port)
        else:
            x_addend.append(zero_port)
            y_addend.append(zero_port)

    sum_outputs, carry = carry_look_ahead_adder(
        circuit, x_addend, y_addend, zero_port, parent_group=this_group
    )

    outputs = list(sum_outputs)
    outputs.append(carry)

    return outputs


def faulty_wallace_tree_multiplier(circuit, x_list, y_list, parent_group=None):
    this_group = circuit.add_group("WALLACE_TREE_MULTIPLIER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    assert len(x_list) == len(y_list), "Both inputs have to be equally long"
    n = len(x_list)
    partial_products = [[] for _ in range(2 * n)]

    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            index = i + j
            node = circuit.add_node("and", "AND", inputs=[x, y], group_id=this_group_id)
            partial_products[index].append(node.ports[2])

    loop_count = 0
    while any(len(col) > 2 for col in partial_products):
        loop_count += 1
        print(f"loop_count: {loop_count}")
        new_products = [[] for _ in range(2 * n)]
        for i in range(2 * n):
            col = partial_products[i]
            j = 0
            while len(col) > j + 2:
                sum, cout = full_adder(
                    circuit, col[j], col[j + 1], col[j + 2], parent_group=this_group
                )
                new_products[i].append(sum)
                new_products[i + 1].append(cout)
                j += 3

            if len(col) > j + 1 and j > 0:
                sum, cout = half_adder(
                    circuit, col[j], col[j + 1], parent_group=this_group
                )
                new_products[i].append(sum)
                new_products[i + 1].append(cout)
                j += 2

            while len(col) > j:
                new_products[i].append(col[j])
                j += 1

        partial_products = new_products

    zero_port = constant_zero(circuit, x_list[0], parent_group=this_group)

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
        circuit, x_addend, y_addend, zero_port, parent_group=this_group
    )

    outputs = sum_outputs
    outputs.append(carry)

    return outputs

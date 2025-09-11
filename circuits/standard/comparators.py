from .trees import *
from ..circuit_utils import *


def one_bit_comparator(circuit, x, y, parent_group=None):
    this_group = circuit.add_group("ONE_BIT_COMPARATOR")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    not_x = not_gate(circuit, x, parent_group=this_group)
    not_y = not_gate(circuit, y, parent_group=this_group)
    x_less_y = and_gate(circuit, [not_x, y], parent_group=this_group)
    x_greater_y = and_gate(circuit, [x, not_y], parent_group=this_group)

    x_equals_y = xnor_gate(circuit, x_less_y, x_greater_y, parent_group=this_group)

    return x_less_y, x_equals_y, x_greater_y


def log_depth_and_tree(circuit, inputs, parent_group=None):
    """Returns a single output that is the AND of all inputs, built with log-depth."""
    if len(inputs) == 1:
        return inputs[0]
    this_group = circuit.add_group("AND_TREE")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    def build(inputs):
        if len(inputs) == 1:
            return inputs[0]
        next_level = []
        for i in range(0, len(inputs), 2):
            if i + 1 < len(inputs):
                and_out = and_gate(
                    circuit, [inputs[i], inputs[i + 1]], parent_group=this_group
                )
                next_level.append(and_out)
            else:
                next_level.append(inputs[i])
        return build(next_level)

    return build(inputs)


def n_bit_comparator(circuit, x_list, y_list, parent_group=None):
    assert len(x_list) == len(y_list), "Both inputs must be the same length"
    n = len(x_list)

    this_group = circuit.add_group("N_BIT_COMPARATOR")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    bit_wise = []
    for i in range(n):
        less, equals, greater = one_bit_comparator(
            circuit, x_list[i], y_list[i], parent_group=this_group
        )
        bit_wise.append((less, equals, greater))

    eq_list = [bit_wise[i][1] for i in reversed(range(n))]

    eq_prefixes = []

    def build_eq_prefixes(eq_inputs):
        if len(eq_inputs) == 0:
            return []
        if len(eq_inputs) == 1:
            return [eq_inputs[0]]
        mid = len(eq_inputs) // 2
        left = build_eq_prefixes(eq_inputs[:mid])
        right = build_eq_prefixes(eq_inputs[mid:])

        if left:
            left_last = left[-1]
            new_right = []
            for r in right:
                and_out = and_gate(circuit, [left_last, r], parent_group=this_group)
                new_right.append(and_out)
            return left + new_right
        return right

    eq_prefixes = build_eq_prefixes(eq_list)

    n_equals = log_depth_and_tree(circuit, eq_list, parent_group=this_group)

    build_less = [bit_wise[n - 1][0]]
    for i in range(n - 2, -1, -1):
        and_out = and_gate(
            circuit, [eq_prefixes[n - 2 - i], bit_wise[i][0]], parent_group=this_group
        )
        build_less.append(and_out)
    n_less = or_tree_recursive(circuit, build_less, parent_group=this_group)

    build_greater = [bit_wise[n - 1][2]]
    for i in range(n - 2, -1, -1):
        and_out = and_gate(
            circuit, [eq_prefixes[n - 2 - i], bit_wise[i][2]], parent_group=this_group
        )
        build_greater.append(and_out)
    n_greater = or_tree_recursive(circuit, build_greater, parent_group=this_group)

    return n_less, n_equals, n_greater


def n_bit_equality(
    circuit: CircuitGraph,
    a: List[Port],
    b: List[Port],
    parent_group: Optional[Group] = None,
) -> Port:

    this_group = circuit.add_group("N_BIT_EQUALITY")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    xnor_bits = []
    for a_bit, b_bit in zip(a, b):
        xor = xor_gate(circuit, [a_bit, b_bit], parent_group=this_group)
        not_xor = not_gate(circuit, xor, parent_group=this_group)
        xnor_bits.append(not_xor)

    return and_tree_iterative(circuit, xnor_bits, parent_group=this_group)

from core.interface import DepthInterface
from circuits.standard.gates import *


def half_adder(circuit, x, y, parent_group=None):
    this_group = circuit.add_group("HALF_ADDER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    xor_out = xor_gate(circuit, [x, y], parent_group=this_group)
    and_out = and_gate(circuit, [x, y], parent_group=this_group)
    return xor_out, and_out


def full_adder(circuit, x, y, cin, parent_group=None):
    this_group = circuit.add_group("FULL_ADDER")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    sum1, carry1 = half_adder(circuit, x, y, parent_group=this_group)
    sum2, carry2 = half_adder(circuit, sum1, cin, parent_group=this_group)
    cout = or_gate(circuit, [carry1, carry2], parent_group=this_group)
    return sum2, cout


def ripple_carry_adder(circuit, x_list, y_list, cin, parent_group=None):
    assert len(x_list) == len(y_list), "x, y must have the same number of bits"
    this_group = circuit.add_group("RIPPLE_CARRY_ADDER")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    sum_outputs = []
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        sum, cin = full_adder(circuit, x, y, cin, parent_group=this_group)
        sum_outputs.append(sum)
    return sum_outputs, cin


def carry_look_ahead_adder(circuit, x_list, y_list, cin, parent_group=None):
    assert len(x_list) == len(y_list), "x, y must have the same number of bits"
    n = len(x_list)
    this_group = circuit.add_group("CARRY_LOOK_AHEAD_ADDER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    propagate = []
    generate = []
    for x, y in zip(x_list, y_list):
        p = xor_gate(circuit, [x, y], parent_group=this_group)
        g = and_gate(circuit, [x, y], parent_group=this_group)
        propagate.append(p)
        generate.append(g)

    def build_group_pg(start, end):
        if start == end:
            return propagate[start], generate[start]
        else:
            mid = (start + end) // 2
            p_low, g_low = build_group_pg(start, mid)
            p_high, g_high = build_group_pg(mid + 1, end)
            p_combined = and_gate(circuit, [p_high, p_low], parent_group=this_group)
            p_high_and_g_low = and_gate(
                circuit, [p_high, g_low], parent_group=this_group
            )
            g_combined = or_gate(
                circuit, [g_high, p_high_and_g_low], parent_group=this_group
            )

            return p_combined, g_combined

    carries = [cin]

    if n > 1:
        for i in range(n):
            if i == 0:
                p0_and_c0 = and_gate(
                    circuit, [propagate[0], cin], parent_group=this_group
                )
                c1 = or_gate(circuit, [generate[0], p0_and_c0], parent_group=this_group)
                carries.append(c1)
            else:
                p_group, g_group = build_group_pg(0, i - 1)

                p_group_and_cin = and_gate(
                    circuit, [p_group, cin], parent_group=this_group
                )
                carry_term = or_gate(
                    circuit, [g_group, p_group_and_cin], parent_group=this_group
                )
                pi_and_carry = and_gate(
                    circuit, [propagate[i], carry_term], parent_group=this_group
                )
                ci_plus_1 = or_gate(
                    circuit, [generate[i], pi_and_carry], parent_group=this_group
                )

                carries.append(ci_plus_1)
    elif n == 1:

        p0_and_c0 = and_gate(circuit, [propagate[0], cin], parent_group=this_group)
        c1 = or_gate(circuit, [generate[0], p0_and_c0], parent_group=this_group)
        carries.append(c1)

    sum_outputs = []
    for i in range(n):

        sum_out = xor_gate(circuit, [propagate[i], carries[i]], parent_group=this_group)
        sum_outputs.append(sum_out)

    return sum_outputs, carries[-1]

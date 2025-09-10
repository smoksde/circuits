from core.interface import DepthInterface
from circuits.gates import *


def half_adder(circuit, x, y, parent_group=None):
    this_group = circuit.add_group("HALF_ADDER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    xor_out = xor_gate(circuit, [x, y], parent_group=this_group)
    and_out = and_gate(circuit, [x, y], parent_group=this_group)
    return xor_out, and_out
    # xor_gate = circuit.add_node("xor", "HA_XOR", inputs=[x, y], group_id=this_group_id)
    # and_gate = circuit.add_node("and", "HA_AND", inputs=[x, y], group_id=this_group_id)
    # return xor_gate.ports[2], and_gate.ports[2]


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
        # p = circuit.add_node("xor", "XOR", inputs=[x, y], group_id=this_group_id)
        # g = circuit.add_node("and", "AND", inputs=[x, y], group_id=this_group_id)
        propagate.append(p)
        generate.append(g)

    def build_group_pg(start, end):
        if start == end:
            return propagate[start], generate[start]
        else:
            mid = (start + end) // 2
            p_low, g_low = build_group_pg(start, mid)
            p_high, g_high = build_group_pg(mid + 1, end)

            # p_combined = circuit.add_node(
            #    "and", "AND", inputs=[p_high, p_low], group_id=this_group_id
            # )

            # p_high_and_g_low = circuit.add_node(
            #    "and", "AND", inputs=[p_high, g_low], group_id=this_group_id
            # )
            # g_combined = circuit.add_node(
            #    "or",
            #    "OR",
            #    inputs=[g_high, p_high_and_g_low.ports[2]],
            #    group_id=this_group_id,
            # )

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
                # p0_and_c0 = circuit.add_node(
                #    "and", "AND", inputs=[propagate[0], cin], group_id=this_group_id
                # )
                # c1 = circuit.add_node(
                #    "or",
                #    "OR",
                #    inputs=[generate[0], p0_and_c0.ports[2]],
                #    group_id=this_group_id,
                # )
                p0_and_c0 = and_gate(
                    circuit, [propagate[0], cin], parent_group=this_group
                )
                c1 = or_gate(circuit, [generate[0], p0_and_c0], parent_group=this_group)
                carries.append(c1)
            else:
                p_group, g_group = build_group_pg(0, i - 1)

                # p_group_and_cin = circuit.add_node(
                #    "and", "AND", inputs=[p_group, cin], group_id=this_group_id
                # )

                # carry_term = circuit.add_node(
                #    "or",
                #    "OR",
                #    inputs=[g_group, p_group_and_cin.ports[2]],
                #    group_id=this_group_id,
                # )

                # pi_and_carry = circuit.add_node(
                #    "and",
                #    "AND",
                #    inputs=[propagate[i], carry_term.ports[2]],
                #    group_id=this_group_id,
                # )
                # ci_plus_1 = circuit.add_node(
                #    "or",
                #    "OR",
                #    inputs=[generate[i], pi_and_carry.ports[2]],
                #    group_id=this_group_id,
                # )

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
        # p0_and_c0 = circuit.add_node(
        #    "and", "AND", inputs=[propagate[0], cin], group_id=this_group_id
        # )
        # c1 = circuit.add_node(
        #    "or", "OR", inputs=[generate[0], p0_and_c0.ports[2]], group_id=this_group_id
        # )
        p0_and_c0 = and_gate(circuit, [propagate[0], cin], parent_group=this_group)
        c1 = or_gate(circuit, [generate[0], p0_and_c0], parent_group=this_group)
        carries.append(c1)

    sum_outputs = []
    for i in range(n):
        # sum_out = circuit.add_node(
        #    "xor", "XOR", inputs=[propagate[i], carries[i]], group_id=this_group_id
        # )
        sum_out = xor_gate(circuit, [propagate[i], carries[i]], parent_group=this_group)
        sum_outputs.append(sum_out)

    return sum_outputs, carries[-1]

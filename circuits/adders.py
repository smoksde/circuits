def half_adder(circuit, x, y, parent_group=None):
    this_group = circuit.add_group("HALF_ADDER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    xor_gate = circuit.add_node("xor", "HA_XOR", inputs=[x, y], group_id=this_group_id)
    and_gate = circuit.add_node("and", "HA_AND", inputs=[x, y], group_id=this_group_id)
    return xor_gate.ports[2], and_gate.ports[2]


def full_adder(circuit, x, y, cin, parent_group=None):
    this_group = circuit.add_group("FULL_ADDER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    sum1, carry1 = half_adder(circuit, x, y, parent_group=this_group)
    sum2, carry2 = half_adder(circuit, sum1, cin, parent_group=this_group)
    cout = circuit.add_node(
        "or", "FA_OR", inputs=[carry1, carry2], group_id=this_group_id
    )
    return sum2, cout.ports[2]


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
        p = circuit.add_node("xor", "XOR", inputs=[x, y], group_id=this_group_id)
        g = circuit.add_node("and", "AND", inputs=[x, y], group_id=this_group_id)
        propagate.append(p.ports[2])
        generate.append(g.ports[2])

    def build_group_pg(start, end):
        if start == end:
            return propagate[start], generate[start]
        else:
            mid = (start + end) // 2
            p_low, g_low = build_group_pg(start, mid)
            p_high, g_high = build_group_pg(mid + 1, end)

            p_combined = circuit.add_node(
                "and", "AND", inputs=[p_high, p_low], group_id=this_group_id
            )

            p_high_and_g_low = circuit.add_node(
                "and", "AND", inputs=[p_high, g_low], group_id=this_group_id
            )
            g_combined = circuit.add_node(
                "or",
                "OR",
                inputs=[g_high, p_high_and_g_low.ports[2]],
                group_id=this_group_id,
            )

            return p_combined.ports[2], g_combined.ports[2]

    carries = [cin]

    if n > 1:
        for i in range(n):
            if i == 0:
                p0_and_c0 = circuit.add_node(
                    "and", "AND", inputs=[propagate[0], cin], group_id=this_group_id
                )
                c1 = circuit.add_node(
                    "or",
                    "OR",
                    inputs=[generate[0], p0_and_c0.ports[2]],
                    group_id=this_group_id,
                )
                carries.append(c1.ports[2])
            else:
                p_group, g_group = build_group_pg(0, i - 1)

                p_group_and_cin = circuit.add_node(
                    "and", "AND", inputs=[p_group, cin], group_id=this_group_id
                )

                carry_term = circuit.add_node(
                    "or",
                    "OR",
                    inputs=[g_group, p_group_and_cin.ports[2]],
                    group_id=this_group_id,
                )

                pi_and_carry = circuit.add_node(
                    "and",
                    "AND",
                    inputs=[propagate[i], carry_term.ports[2]],
                    group_id=this_group_id,
                )
                ci_plus_1 = circuit.add_node(
                    "or",
                    "OR",
                    inputs=[generate[i], pi_and_carry.ports[2]],
                    group_id=this_group_id,
                )

                carries.append(ci_plus_1.ports[2])
    elif n == 1:
        p0_and_c0 = circuit.add_node(
            "and", "AND", inputs=[propagate[0], cin], group_id=this_group_id
        )
        c1 = circuit.add_node(
            "or", "OR", inputs=[generate[0], p0_and_c0.ports[2]], group_id=this_group_id
        )
        carries.append(c1.ports[2])

    sum_outputs = []
    for i in range(n):
        sum_out = circuit.add_node(
            "xor", "XOR", inputs=[propagate[i], carries[i]], group_id=this_group_id
        )
        sum_outputs.append(sum_out.ports[2])

    return sum_outputs, carries[-1]

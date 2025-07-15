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

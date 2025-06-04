from .trees import *
import utils


def multiplexer(circuit, inputs_list, selector_list):
    # inputs should be ascending order
    n_bits = len(selector_list)
    not_selector_list = []
    for sel in selector_list:
        not_sel = circuit.add_node("not", "NOT", inputs=[sel])
        not_selector_list.append(not_sel.ports[1])

    and_ports = []
    for i in range(len(inputs_list)):
        bin_list = utils.int2binlist(i, bit_len=n_bits)
        curr_sel = []
        for j in range(len(bin_list)):
            if bin_list[j]:
                curr_sel.append(selector_list[j])
            else:
                curr_sel.append(not_selector_list[j])
        and_ins = [inputs_list[i]]
        and_ins.extend(curr_sel)
        and_port = and_tree_recursive(circuit, and_ins)
        and_ports.append(and_port)

    or_port = or_tree_recursive(circuit, and_ports)
    return or_port


# if signal then a else b
def mux2(circuit, signal, a, b, parent_group=None):
    mux_group = circuit.add_group("MUX")
    mux_group.set_parent(parent_group)
    not_signal = circuit.add_node(
        "not", "MUX_NOT", inputs=[signal], group_id=mux_group.id
    )
    not_signal_port = not_signal.ports[1]
    first_and = circuit.add_node(
        "and", "MUX_AND", inputs=[signal, a], group_id=mux_group.id
    )
    second_and = circuit.add_node(
        "and", "MUX_AND", inputs=[not_signal_port, b], group_id=mux_group.id
    )
    first_and_port = first_and.ports[2]
    second_and_port = second_and.ports[2]
    out_node = circuit.add_node(
        "or", "MUX_OR", inputs=[first_and_port, second_and_port], group_id=mux_group.id
    )
    out_port = out_node.ports[2]
    return out_port

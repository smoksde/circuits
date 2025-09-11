from core.graph import *

from .trees import *

import utils


def multiplexer(
    circuit, inputs_list, selector_list, parent_group: Optional[Group] = None
):

    this_group = circuit.add_group("MULTIPLEXER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    n_bits = len(selector_list)
    not_selector_list = []
    for sel in selector_list:
        not_sel = not_gate(circuit, sel, parent_group=this_group)
        not_selector_list.append(not_sel)

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
        and_port = and_tree_iterative(circuit, and_ins, parent_group=this_group)
        and_ports.append(and_port)

    or_port = or_tree_iterative(circuit, and_ports, parent_group=this_group)
    return or_port


def bus_multiplexer(circuit, bus, selector, parent_group: Optional[Group] = None):

    this_group = circuit.add_group("BUS_MULTIPLEXER")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    bit_width = len(bus[0])
    num_amount = len(bus)
    num_out = []
    for i in range(bit_width):
        in_list = []
        for j in range(num_amount):
            p = bus[j][i]
            in_list.append(p)
        sig = multiplexer(circuit, in_list, selector, parent_group=this_group)
        num_out.append(sig)

    return num_out


def tensor_multiplexer(
    circuit: CircuitGraph, tensor, selector, parent_group: Optional[Group] = None
):

    this_group = circuit.add_group("TENSOR_MULTIPLEXER")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    dim_one = len(tensor)
    dim_two = len(tensor[0])
    bit_width = len(tensor[0][0])

    result = []
    for i in range(dim_two):
        bus = [tensor[k][i] for k in range(dim_one)]
        selected_word = bus_multiplexer(circuit, bus, selector, parent_group=this_group)
        result.append(selected_word)
    return result


def mux2(circuit, signal, a, b, parent_group: Optional[Group] = None):

    this_group = circuit.add_group("MUX")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    not_signal = not_gate(circuit, signal, parent_group=this_group)
    not_signal_port = not_signal
    first_and = and_gate(circuit, [signal, a], parent_group=this_group)
    second_and = and_gate(circuit, [not_signal_port, b], parent_group=this_group)
    first_and_port = first_and
    second_and_port = second_and
    out_node = or_gate(
        circuit, [first_and_port, second_and_port], parent_group=this_group
    )
    out_port = out_node
    return out_port

from core.graph import *

from .trees import *

import utils


# expects selector with log num of inputs bits?
def multiplexer(
    circuit, inputs_list, selector_list, parent_group: Optional[Group] = None
):

    this_group = circuit.add_group("MULTIPLEXER")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    # inputs should be ascending order
    n_bits = len(selector_list)
    not_selector_list = []
    for sel in selector_list:
        #not_sel = circuit.add_node("not", "NOT", inputs=[sel], group_id=this_group_id)
        #not_selector_list.append(not_sel.ports[1])
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


# selects between multi bit numbers
# bus = [[a1,a2,...],[b1,b2,...],[c1,c2,...],[d1,d2,...]]
# selector = [s1,s2]
# result = [b1,b2,b3,b4] if selector selects the second number, in this case b
# selector bit width is log of num of inputs to choose from
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


# gets a 3-dim input and reduces it to 2-dims
# selects one row, so one entry of the first dim
"""def tensor_multiplexer(circuit, tensor, selector):
    bit_width = len(tensor[0][0])
    dim_one = len(tensor)
    dim_two = len(tensor[0])
    result = []
    for i in range(dim_two):
        outer_list = []
        for j in range(bit_width):
            inner_list = []
            for k in range(dim_one):
                p = tensor[k][i][j]
                inner_list.append(p)
            outer_list.append(inner_list)
        sig = bus_multiplexer(circuit, outer_list, selector)
        result.append(sig)
    return result"""


def tensor_multiplexer(
    circuit: CircuitGraph, tensor, selector, parent_group: Optional[Group] = None
):

    this_group = circuit.add_group("TENSOR_MULTIPLEXER")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    dim_one = len(tensor)  # number of words to choose from
    dim_two = len(tensor[0])  # rows (i.e., how many outputs you want)
    bit_width = len(tensor[0][0])  # width of each number

    result = []
    for i in range(dim_two):  # for each row
        # gather all words at this row index i
        bus = [tensor[k][i] for k in range(dim_one)]  # each tensor[k][i] is a word
        selected_word = bus_multiplexer(circuit, bus, selector, parent_group=this_group)
        result.append(selected_word)
    return result


# if signal then a else b
def mux2(circuit, signal, a, b, parent_group: Optional[Group] = None):

    this_group = circuit.add_group("MUX")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    #not_signal = circuit.add_node(
    #    "not", "MUX_NOT", inputs=[signal], group_id=this_group_id
    #)
    not_signal = not_gate(circuit, signal, parent_group=this_group)
    not_signal_port = not_signal
    #first_and = circuit.add_node(
    #    "and", "MUX_AND", inputs=[signal, a], group_id=this_group_id
    #)
    #second_and = circuit.add_node(
    #    "and", "MUX_AND", inputs=[not_signal_port, b], group_id=this_group_id
    #)
    first_and = and_gate(circuit, [signal, a], parent_group=this_group)
    second_and = and_gate(circuit, [not_signal_port, b], parent_group=this_group)
    first_and_port = first_and
    second_and_port = second_and
    #out_node = circuit.add_node(
    #    "or", "MUX_OR", inputs=[first_and_port, second_and_port], group_id=this_group_id
    #)
    out_node = or_gate(circuit, [first_and_port, second_and_port], parent_group=this_group)
    out_port = out_node
    return out_port

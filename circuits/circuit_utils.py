from .constants import *
from .adders import *
from .trees import *

from utils import int2binlist


def sign_detector(circuit, x_list):
    msb = x_list[-1]
    return msb


def two_complement(circuit, x_list, parent_group=None):
    this_group = circuit.add_group("TWO_COMPLEMENT")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    inverted_list = []
    for x in x_list:
        #not_node = circuit.add_node("not", "NOT", inputs=[x], group_id=this_group_id)
        not_node = not_gate(circuit, x, parent_group=this_group)
        inverted_list.append(not_node)
    one = constant_one(circuit, x_list[0], parent_group=this_group)
    zero = constant_zero(circuit, x_list[0], parent_group=this_group)
    one_number = []
    for i in range(len(x_list)):
        one_number.append(zero)
    one_number[0] = one
    two_comp_list, _ = carry_look_ahead_adder(
        circuit, inverted_list, one_number, zero, parent_group=this_group
    )
    # two_comp_list, _ = ripple_carry_adder(
    #    circuit, inverted_list, one_number, zero, parent_group=tc_group
    # )
    return two_comp_list


def xnor_gate(circuit, x, y, parent_group=None):
    this_group = circuit.add_group("XNOR")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)
    #or_node = circuit.add_node("or", "OR", inputs=[x, y], group_id=this_group_id)
    or_out = or_gate(circuit, [x, y], parent_group=this_group)
    or_node_port = or_out
    #and_node = circuit.add_node("and", "AND", inputs=[x, y], group_id=this_group_id)
    and_out = and_gate(circuit, [x, y], parent_group=this_group)
    #not_or_node = circuit.add_node(
    #    "not", "NOT", inputs=[or_node_port], group_id=this_group_id
    #)
    not_or_out = not_gate(circuit, or_out, parent_group=this_group)
    #xnor_node = circuit.add_node(
    #    "or", "OR", inputs=[not_or_node_port, and_node.ports[2]], group_id=this_group_id
    #)
    xnor_out = or_gate(circuit, [not_or_out, and_out], parent_group=this_group)
    return xnor_out


def next_power_of_two(circuit, x, parent_group=None):
    this_group = circuit.add_group("NEXT_POWER_OF_TWO")
    this_group_id = this_group.id if this_group is not None else -1
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    next_power = []

    for idx, bit in enumerate(x):
        if idx == 0:
            hi = x
            ln = [constant_zero(circuit, in_port=x[0], parent_group=this_group)]
            lo = []
        elif idx == len(x) - 1:
            hi = [bit]
            ln = [x[idx - 1]]
            if idx - 2 >= 0:
                lo = x[: idx - 2]
            else:
                lo = []
        else:
            hi = x[idx:]
            ln = [x[idx - 1]]
            if idx - 2 >= 0:
                lo = x[: idx - 2]
            else:
                lo = []  # empty list -> later constant value

        # build or trees
        hi_ot = or_tree_iterative(circuit, input_list=hi, parent_group=this_group)
        #not_hi_ot = circuit.add_node(
        #    "not", "NOT", inputs=[hi_ot], group_id=this_group_id
        #).ports[1]
        not_hi_ot = not_gate(circuit, hi_ot, parent_group=this_group)
        ln_ot = or_tree_iterative(circuit, input_list=ln, parent_group=this_group)
        # lo_ot = or_tree_iterative(circuit, input_list=lo, parent_group=npot_group)

        #not_hi_and_ln = circuit.add_node(
        #    "and", "AND", inputs=[not_hi_ot, ln_ot], group_id=this_group_id
        #).ports[2]
        not_hi_and_ln = and_gate(circuit, [not_hi_ot, ln_ot], parent_group=this_group)

        next_power.append(not_hi_and_ln)

    return next_power


"""def next_power_of_two(circuit: CircuitGraph, x, parent_group=None):
    npot_group = circuit.add_group("NEXT_POWER_OF_TWO")
    npot_group.set_parent(parent_group)
    next_power = []

    next_power.append(constant_zero(circuit, x[0], parent_group=npot_group))
    for idx, bit in enumerate(x[:-1]):
        found_one = or_tree_iterative(circuit, x[idx:], parent_group=npot_group)
        not_found_one = circuit.add_node(
            "not", "NOT", inputs=[found_one], group_id=npot_group.id
        ).ports[1]
        and_port = circuit.add_node(
            "and", "AND", inputs=[not_found_one, bit], group_id=npot_group.id
        )
        next_power.append(and_port.ports[2])

    return next_power"""


def generate_number(
    n: int, bit_len: int, zero: Port, one: Port, parent_group: Optional[Group] = None
) -> List[Port]:
    #this_group = circuit.add_group("GENERATE_NUMBER")
    #if circuit.enable_groups and this_group is not None:
    #    this_group.set_parent(parent_group)
    bits = int2binlist(n, bit_len=bit_len)
    ports = []
    for bit in bits:
        if bit:
            ports.append(one)
        else:
            ports.append(zero)
    return ports

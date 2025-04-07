from node import Node
from port import Port
from edge import Edge

class CircuitGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.node_count = 0
        self.node_values = {}
        self.port_count = 0

    def add_node(self, node_type, label, inputs=[]):
        node_id = self.node_count
        self.node_count += 1
        node = Node(node_id, node_type, label)
        
        if node_type in ["xor", "and", "or"]:
            input_port_1 = node.add_port(Port(self.port_count, "input", node_id, 0, 10))
            self.port_count += 1
            input_port_2 = node.add_port(Port(self.port_count, "input", node_id, 0, 30))
            self.port_count += 1
            _ = node.add_port(Port(self.port_count, "output", node_id, 40, 20))
            self.port_count += 1
            if len(inputs) == 2:
                self.add_edge(inputs[0], input_port_1)
                self.add_edge(inputs[1], input_port_2)
        elif node_type == "not":
            input_port = node.add_port(Port(self.port_count, "input", node_id, 0, 20))
            self.port_count += 1
            _ = node.add_port(Port(self.port_count, "output", node_id, 40, 20))
            self.port_count += 1
            if len(inputs) == 1:
                self.add_edge(inputs[0], input_port)
        elif node_type == "input":
            _ = node.add_port(Port(self.port_count, "output", node_id, 40, 20))
            self.port_count += 1
        elif node_type == "output":
            input_port_1 = node.add_port(Port(self.port_count, "input", node_id, 0, 20))
            self.port_count += 1
            if len(inputs) == 1:
                self.add_edge(inputs[0], input_port_1)
        else:
            print("Unknown node type!")
            exit()

        self.nodes[str(node_id)] = node
        self.node_values[str(node_id)] = 0
        return node
    
    def add_edge(self, source_port, target_port):
        edge = Edge(source_port.id, target_port.id)
        self.edges.append(edge)
        return edge

    def to_json(self):
        nodes = [node.to_dict() for node in self.nodes.values()]
        edges = [edge.to_dict() for edge in self.edges]       
        return {"nodes": nodes, "edges": edges, "values": self.node_values}
    
    def simulate(self):
        port_values = {}
        port_sources = {}
        
        for edge in self.edges:
            port_sources[edge.target_port_id] = edge.source_port_id
        
        for node_id, node in self.nodes.items():
            if node.type == "input":
                val = self.node_values[node_id]
                output_port = node.ports[0]
                port_values[output_port.id] = val

        remaining = set(self.nodes.keys()) - {nid for nid, n in self.nodes.items() if n.type == "input"}
        resolved = set()

        while remaining:
            progress = False
            for node_id in list(remaining):
                node = self.nodes[node_id]

                input_ports = [p for p in node.ports if p.type == "input"]
                try:
                    inputs = [port_values[port_sources[p.id]] for p in input_ports]
                except KeyError:
                    continue  # not all inputs are ready

                output_val = None
                if node.type == "and":
                    output_val = inputs[0] & inputs[1]
                elif node.type == "or":
                    output_val = inputs[0] | inputs[1]
                elif node.type == "xor":
                    output_val = inputs[0] ^ inputs[1]
                elif node.type == "not":
                    output_val = 0 if inputs[0] else 1
                elif node.type == "output":
                    # For output nodes, we don’t compute anything; just store value
                    port_values[node.ports[0].id] = inputs[0]
                    resolved.add(node_id)
                    remaining.remove(node_id)
                    progress = True
                    continue
                else:
                    continue  # skip unknown nodes

                output_port = [p for p in node.ports if p.type == "output"][0]
                port_values[output_port.id] = output_val
                resolved.add(node_id)
                remaining.remove(node_id)
                progress = True

            if not progress:
                raise RuntimeError("Simulation stalled; possible cycle or unconnected inputs.")

        self.port_values = port_values

    def get_port_value(self, port):
        return self.port_values.get(port.id, None)

#def half_adder(circuit, x, y):
#    sum = circuit.add_gate("XOR", [x, y])
#    carry = circuit.add_gate("AND", [x, y])
#    return sum, carry

def constant_zero(circuit, in_port):
    not_in = circuit.add_node("not", "NOT", inputs=[in_port])
    not_in_port = not_in.ports[1]
    zero_node = circuit.add_node("and", "ZERO_AND", inputs=[in_port, not_in_port])
    zero_port = zero_node.ports[2]
    return zero_port

def constant_one(circuit, in_port):
    not_in = circuit.add_node("not", "NOT", inputs=[in_port])
    not_in_port = not_in.ports[1]
    one_node = circuit.add_node("or", "ONE_OR", inputs=[in_port, not_in_port])
    one_port = one_node.ports[2]
    return one_port

def and_tree_recursive(circuit, input_list):
    if len(input_list) == 1:
        return input_list[0]
    
    if len(input_list) == 2:
        and_node = circuit.add_node("and", "AND", inputs=input_list)
        return and_node.ports[2]
    
    mid = len(input_list) // 2
    left = and_tree_recursive(circuit, input_list[:mid])
    right = and_tree_recursive(circuit, input_list[mid:])
    and_node = circuit.add_node("and", "AND", inputs=[left, right])
    return and_node.ports[2]

def or_tree_recursive(circuit, input_list):
    if len(input_list) == 1:
        return input_list[0]
    
    if len(input_list) == 2:
        or_node = circuit.add_node("or", "OR", inputs=input_list)
        return or_node.ports[2]
    
    mid = len(input_list) // 2
    left = or_tree_recursive(circuit, input_list[:mid])
    right = or_tree_recursive(circuit, input_list[mid:])
    or_node = circuit.add_node("or", "OR", inputs=[left, right])
    return or_node.ports[2]

def half_adder(circuit, x, y):
    xor_gate = circuit.add_node("xor", "HA_XOR", inputs=[x, y])
    and_gate = circuit.add_node("and", "HA_AND", inputs=[x, y])
    return xor_gate.ports[2], and_gate.ports[2]

def full_adder(circuit, x, y, cin):
    sum1, carry1 = half_adder(circuit, x, y)
    sum2, carry2 = half_adder(circuit, sum1, cin)
    cout = circuit.add_node("or", "FA_OR", inputs=[carry1, carry2])
    return sum2, cout.ports[2]

def ripple_carry_adder(circuit, x_list, y_list, cin):
    assert len(x_list) == len(y_list), "x, y must have the same number of bits"
    
    sum_outputs = []
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        sum, cin = full_adder(circuit, x, y, cin)
        sum_outputs.append(sum)
    return sum_outputs, cin
    carry = cin
    sum_outputs = []
    for x, y in zip(x_list, y_list):
        sum, carry = full_adder(circuit, x, y, carry)
        sum_outputs.append(sum)
    return sum_outputs, carry

def carry_look_ahead_adder(circuit, x_list, y_list, cin):
    assert len(x_list) == len(y_list), "x, y must have the same number of bits"
    n = len(x_list)
    
    propagate = []
    generate = []
    for x, y in zip(x_list, y_list):
        p = circuit.add_node("xor", "XOR", inputs=[x, y])
        g = circuit.add_node("and", "AND", inputs=[x, y])
        propagate.append(p.ports[2])
        generate.append(g.ports[2])
    
    def build_group_pg(start, end):
        if start == end:
            return propagate[start], generate[start]
        else:
            mid = (start + end) // 2
            p_low, g_low = build_group_pg(start, mid)
            p_high, g_high = build_group_pg(mid + 1, end)
            
            p_combined = circuit.add_node("and", "AND", inputs=[p_high, p_low])
            
            p_high_and_g_low = circuit.add_node("and", "AND", inputs=[p_high, g_low])
            g_combined = circuit.add_node("or", "OR", inputs=[g_high, p_high_and_g_low.ports[2]])
            
            return p_combined.ports[2], g_combined.ports[2]
    
    carries = [cin]
    
    if n > 1:
        for i in range(n):
            if i == 0:
                p0_and_c0 = circuit.add_node("and", "AND", inputs=[propagate[0], cin])
                c1 = circuit.add_node("or", "OR", inputs=[generate[0], p0_and_c0.ports[2]])
                carries.append(c1.ports[2])
            else:
                p_group, g_group = build_group_pg(0, i-1)
                
                p_group_and_cin = circuit.add_node("and", "AND", inputs=[p_group, cin])
                
                carry_term = circuit.add_node("or", "OR", inputs=[g_group, p_group_and_cin.ports[2]])
                
                pi_and_carry = circuit.add_node("and", "AND", inputs=[propagate[i], carry_term.ports[2]])
                ci_plus_1 = circuit.add_node("or", "OR", inputs=[generate[i], pi_and_carry.ports[2]])
                
                carries.append(ci_plus_1.ports[2])
    elif n == 1:
        p0_and_c0 = circuit.add_node("and", "AND", inputs=[propagate[0], cin])
        c1 = circuit.add_node("or", "OR", inputs=[generate[0], p0_and_c0.ports[2]])
        carries.append(c1.ports[2])
    
    sum_outputs = []
    for i in range(n):
        sum_out = circuit.add_node("xor", "XOR", inputs=[propagate[i], carries[i]])
        sum_outputs.append(sum_out.ports[2])
    
    return sum_outputs, carries[-1]

    propagate = []
    generate = []
    
    for x, y in zip(x_list, y_list):
        p = circuit.add_gate("OR", [x, y])
        g = circuit.add_gate("AND", [x, y])
        propagate.append(p)
        generate.append(g)

    carries = [cin]
    for i in range(len(x_list) - 1):
        p_and_c = circuit.add_gate("AND", [propagate[i], carries[i]])
        
        carry = circuit.add_gate("OR", [generate[i], p_and_c])
        carries.append(carry)

    sum_outputs = []
    for i in range(len(x_list)):
        sum_out = circuit.add_gate("XOR", [x_list[i], y_list[i]])
        sum_out = circuit.add_gate("XOR", [sum_out, carries[i]])
        sum_outputs.append(sum_out)

    return sum_outputs, carries[-1]

def xnor_gate(circuit, x, y):
    or_node = circuit.add_node("or", "OR", inputs=[x, y])
    or_node_port = or_node.ports[2]
    and_node = circuit.add_node("and", "AND", inputs=[x, y])
    not_or_node = circuit.add_node("not", "NOT", inputs=[or_node_port])
    not_or_node_port = not_or_node.ports[1]
    xnor_node = circuit.add_node("or", "OR", inputs=[not_or_node_port, and_node.ports[2]])
    return xnor_node.ports[2]

def one_bit_comparator(circuit, x, y):
    not_x = circuit.add_node("not", "NOT", inputs=[x])
    not_y = circuit.add_node("not", "NOT", inputs=[y])
    x_less_y = circuit.add_node("and", "AND", inputs=[not_x.ports[1], y])
    x_greater_y = circuit.add_node("and", "AND", inputs=[x, not_y.ports[1]])
    x_equals_y = xnor_gate(circuit, x_less_y.ports[2], x_greater_y.ports[2])
    return x_less_y.ports[2], x_equals_y, x_greater_y.ports[2]

def n_bit_comparator(circuit, x_list, y_list):
    assert len(x_list) == len(y_list), "Both inputs have to be equally long"
    n = len(x_list)

    # bit wise results
    bit_wise = []
    for i in range(n):
        less, equals, greater = one_bit_comparator(circuit, x_list[i], y_list[i])
        bit_wise.append((less, equals, greater))

    # and gates for n bit equals
    index = n - 1
    pre_port = bit_wise[index][1] # msb equals port
    build_equals = []
    build_equals.append(pre_port)
    while index > 0:
        curr_equals = circuit.add_node("and", "AND", inputs=[pre_port, bit_wise[index-1][1]])
        build_equals.append(curr_equals.ports[2])
        pre_port = curr_equals.ports[2]
        index -= 1
    n_equals = pre_port

    index = n - 1
    build_less = []
    build_less.append(bit_wise[index][0]) # append msb less port
    while index > 0:
        curr_less_node = circuit.add_node("and", "AND", inputs=[build_equals[n-1-index], bit_wise[index-1][0]])
        build_less.append(curr_less_node.ports[2])
        index -= 1

    # build or tree for n bit less
    n_less = or_tree_recursive(circuit, build_less)

    index = n - 1
    build_greater = []
    build_greater.append(bit_wise[index][2])
    while index > 0:
        curr_greater_node = circuit.add_node("and", "AND", inputs=[build_equals[n-1-index], bit_wise[index-1][2]])
        build_greater.append(curr_greater_node.ports[2])
        index -= 1
    
    n_greater = or_tree_recursive(circuit, build_greater)

    return n_less, n_equals, n_greater

def wallace_tree_multiplier(circuit, x_list, y_list):
    assert len(x_list) == len(y_list), "Both inputs have to be equally long"
    n = len(x_list)
    partial_products = [[] for _ in range(2*n)]

    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            index = i + j
            node = circuit.add_node("and", "AND", inputs=[x, y])
            partial_products[index].append(node.ports[2])

    while any(len(col) > 2 for col in partial_products):
        new_products = [[] for _ in range(2*n)]
        for i in range(2*n):
            col = partial_products[i]
            j = 0
            while len(col) > j + 2: # as long as 3 or more partial products remain in the current column
                sum, cout = full_adder(circuit, col[j], col[j+1], col[j+2])
                new_products[i].append(sum)
                new_products[i+1].append(cout)
                j += 3

            if len(col) > j + 1 and j > 0: # if 2 partial products remain in the current column and j > 0
                sum, cout = half_adder(circuit, col[j], col[j+1])
                new_products[i].append(sum)
                new_products[i+1].append(cout)
                j += 2

            while len(col) > j:
                new_products[i].append(col[j])
                j += 1

        partial_products = new_products

    zero_port = constant_zero(circuit, x_list[0])

    # Applying fast adder since all columns have at most 2 partial products
    x_addend = []
    y_addend = []
    
    for i in range(2*n):
        if len(partial_products[i]) == 2:
            x_addend.append(partial_products[i][0])
            y_addend.append(partial_products[i][1])
        if len(partial_products[i]) == 1:
            x_addend.append(partial_products[i][0])
            y_addend.append(zero_port)

    sum_outputs, carry = carry_look_ahead_adder(circuit, x_addend, y_addend, zero_port)

    outputs = sum_outputs
    outputs.append(carry)

    return outputs

def int_to_binary_list(n, bit_length=None):
    binary_str = bin(n)[2:]
    binary_list = [int(bit) for bit in reversed(binary_str)]
    if bit_length:
        binary_list += [0] * (bit_length - len(binary_list))
    return binary_list

def binary_list_to_int(binary_list):
    return sum(bit * (2 ** i) for i, bit in enumerate(binary_list))

def multiplexer(circuit, inputs_list, selector_list):
    # inputs should be ascending order
    n_bits = len(selector_list)
    not_selector_list = []
    for sel in selector_list:
        not_sel = circuit.add_node("not", "NOT", inputs=[sel])
        not_selector_list.append(not_sel.ports[1])

    and_ports = []
    for i in range(len(inputs_list)):
        bin_list = int_to_binary_list(i, bit_length=n_bits)
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

def adder_tree_recursive(circuit, summand_lists, cin):
    if len(summand_lists) == 1:
        return summand_lists[0], cin
    
    if len(summand_lists) == 2:
        sums, carry= ripple_carry_adder(circuit, summand_lists[0], summand_lists[1], cin)
        return sums, carry
    
    mid = len(summand_lists) // 2
    left_sums, left_carry = adder_tree_recursive(circuit, summand_lists[:mid], cin)
    right_sums, right_carry = adder_tree_recursive(circuit, summand_lists[mid:], cin)
    sums, carry = ripple_carry_adder(circuit, left_sums, right_sums, cin)
    return sums, carry

def or_tree_recursive(circuit, input_list):
    if len(input_list) == 1:
        return input_list[0]
    
    if len(input_list) == 2:
        or_node = circuit.add_node("or", "OR", inputs=input_list)
        return or_node.ports[2]
    
    mid = len(input_list) // 2
    left = or_tree_recursive(circuit, input_list[:mid])
    right = or_tree_recursive(circuit, input_list[mid:])
    or_node = circuit.add_node("or", "OR", inputs=[left, right])
    return or_node.ports[2]

def conditional_zeroing(circuit, x_list, cond):
    ports = []
    for x in x_list:
        and_node = circuit.add_node("and", "AND", inputs=[x, cond])
        ports.append(and_node.ports[2])
    return ports

def sign_detector(circuit, x_list):
    msb = x_list[-1]
    return msb

def two_complement(circuit, x_list):
    inverted_list = []
    for x in x_list:
        not_node = circuit.add_node("not", "NOT", inputs=[x])
        inverted_list.append(not_node.ports[1])
    one = constant_one(circuit, x_list[0])
    zero = constant_zero(circuit, x_list[0])
    one_number = []
    for i in range(len(x_list)):
        one_number.append(zero)
    one_number[0] = one
    two_comp_list, _ = ripple_carry_adder(circuit, inverted_list, one_number, zero)
    return two_comp_list

def small_mod_lemma_4_1(circuit, x_list, m_list, int_m):

    n = len(x_list)

    input = circuit.add_node("input", "INPUT")
    const_zero = constant_zero(circuit, input.ports[0])
    const_one = constant_one(circuit, input.ports[0])

    # precompute constants: a_im = 2^i mod m values
    a_i_lists = []
    for i in range(n):
        calc = (2**i) % int_m
        a = []
        for j in range(n):
            if calc % 2 == 0:
                a.append(const_zero)
            else:
                a.append(const_one)
            calc >>= 1
        a_i_lists.append(a)
    
    summands = []
    for ind, x_i in enumerate(x_list):
        summand = conditional_zeroing(circuit, a_i_lists[ind], x_i)
        summands.append(summand)

    y, carry = adder_tree_recursive(circuit, summands, const_zero)

    results = []
    for i in range(n):
        bin_i = int_to_binary_list(n, len(x_list))
        coef = [const_zero if bit == 0 else const_one for bit in bin_i]
        mult_m = wallace_tree_multiplier(circuit, m_list, coef)
        mult_m = mult_m[:-(len(mult_m)//2)]
        negative_mult_m = two_complement(circuit, mult_m)
        print(len(y))
        print(len(negative_mult_m))
        
        sum, carry = ripple_carry_adder(circuit, y, negative_mult_m, const_zero)


        is_negative = sign_detector(circuit, sum) # if 1 then negative
        result = conditional_zeroing(circuit, coef, is_negative)
        less, equals, greater = n_bit_comparator(circuit, sum, m_list)
        not_less_node = circuit.add_node("not", "NOT", inputs=[less])
        result = conditional_zeroing(circuit, result, not_less_node.ports[1])
        results.append(result)

    final = []
    for i in range(len(result)):
        for j in range(len(results)):
            curr_list = []
            curr_list.append(results[j][i])
        bit = or_tree_recursive(circuit, curr_list)
        final.append(bit)
    return final

CIRCUIT_FUNCTIONS = {
    "xnor_gate": lambda cg: setup_xnor_gate(cg),
    "one_bit_comparator": lambda cg: setup_one_bit_comparator(cg),
    "n_bit_comparator": lambda cg: setup_n_bit_comparator(cg),
    "constant_zero": lambda cg: setup_constant_zero(cg),
    "constant_one": lambda cg: setup_constant_one(cg),
    "and_tree_recursive": lambda cg: setup_and_tree_recursive(cg),
    "or_tree_recursive": lambda cg: setup_or_tree_recursive(cg),
    "half_adder": lambda cg: setup_half_adder(cg),
    "full_adder": lambda cg: setup_full_adder(cg),
    "ripple_carry_adder": lambda cg: setup_ripple_carry_adder(cg),
    "carry_look_ahead_adder": lambda cg: setup_carry_look_ahead_adder(cg),
    "wallace_tree_multiplier": lambda cg: setup_wallace_tree_multiplier(cg),
    "multiplexer": lambda cg: setup_multiplexer(cg),
    "adder_tree_recursive": lambda cg: setup_adder_tree_recursive(cg),
    "small_mod_lemma_4_1": lambda cg: setup_small_mod_lemma_4_1(cg),
}

def setup_small_mod_lemma_4_1(cg):
    X = [cg.add_node("input", f"X{i}") for i in range(4)]
    M = [cg.add_node("input", f"M{i}") for i in range(4)]
    outputs = small_mod_lemma_4_1(cg, [x.ports[0] for x in X], [m.ports[0] for m in M], 2)
    for out in outputs:
        cg.add_node("output", "REMAINDER", inputs=[out])
    return

def setup_adder_tree_recursive(cg):
    ports = []
    for k in range(4):
        X = [cg.add_node("input", f"X{i}") for i in range(4)]
        ports.append([x.ports[0] for x in X])
    cin = cg.add_node("input", "CIN")
    outputs, carry = adder_tree_recursive(cg, ports, cin.ports[0])
    for out in outputs:
        cg.add_node("output", "SUM", inputs=[out])
    cg.add_node("output", "CARRY", inputs=[carry])
    return

def setup_multiplexer(cg):
    X = [cg.add_node("input", f"X{i}") for i in range(4)]
    S = [cg.add_node("input", f"S{i}") for i in range(2)]
    mux = multiplexer(cg, [x.ports[0] for x in X], [s.ports[0] for s in S])
    out = cg.add_node("output", "MUX OUT", inputs=[mux])
    return

def setup_one_bit_comparator(cg):
    less, equals, greater = one_bit_comparator(cg, cg.add_node("input", "x").ports[0], cg.add_node("input", "y").ports[0])
    less_node = cg.add_node("output", "LESS", inputs=[less])
    equals_node = cg.add_node("output", "EQUALS", inputs=[equals])
    greater_node = cg.add_node("output", "GREATER", inputs=[greater])
    return

def setup_n_bit_comparator(cg):
    A = [cg.add_node("input", f"A{i}") for i in range(4)]
    B = [cg.add_node("input", f"B{i}") for i in range(4)]
    less, equals, greater = n_bit_comparator(cg, [a.ports[0] for a in A], [b.ports[0] for b in B])
    less_node = cg.add_node("output", "LESS", inputs=[less])
    equals_node = cg.add_node("output", "EQUALS", inputs=[equals])
    greater_node = cg.add_node("output", "GREATER", inputs=[greater])
    return

def setup_xnor_gate(cg):
    X = [cg.add_node("input", f"X{i}") for i in range(2)]
    out = xnor_gate(cg, X[0].ports[0], X[1].ports[0])
    xnor_output = cg.add_node("output", "XNOR OUTPUT", inputs=[out])
    return

def setup_constant_zero(cg):
    X = cg.add_node("input", "X")
    zero_port = constant_zero(cg, X.ports[0])
    zero_node = cg.add_node("output", "CONSTANT ZERO", inputs=[zero_port])
    return

def setup_constant_one(cg):
    X = cg.add_node("input", "X")
    one_port = constant_one(cg, X.ports[0])
    one_node = cg.add_node("output", "CONSTANT ONE", inputs=[one_port])
    return

def setup_and_tree_recursive(cg):
    X = [cg.add_node("input", f"X{i}") for i in range(4)]
    out = and_tree_recursive(cg, [x.ports[0] for x in X])
    cg.add_node("output", "OUTPUT", inputs=[out])
    return

def setup_or_tree_recursive(cg):
    X = [cg.add_node("input", f"X{i}") for i in range(4)]
    out = or_tree_recursive(cg, [x.ports[0] for x in X])
    cg.add_node("output", "OUTPUT", inputs=[out])
    return

def setup_half_adder(cg):
    A = cg.add_node("input", "A")
    B = cg.add_node("input", "B")
    sum_port, carry_port = half_adder(cg, A.ports[0], B.ports[0])
    sum_node = cg.add_node("output", "sum")
    carry_node = cg.add_node("output", "carry")
    cg.add_edge(sum_port, sum_node.ports[0])
    cg.add_edge(carry_port, carry_node.ports[0])
    
    return A, B, sum_node, carry_node

def setup_full_adder(cg):
    A = cg.add_node("input", "A")
    B = cg.add_node("input", "B")
    cin = cg.add_node("input", "Cin")
    sum_port, carry_port = full_adder(cg, A.ports[0], B.ports[0], cin.ports[0])
    sum_node = cg.add_node("output", "sum")
    carry_node = cg.add_node("output", "carry")
    cg.add_edge(sum_port, sum_node.ports[0])
    cg.add_edge(carry_port, carry_node.ports[0])
    return A, B, cin, sum_node, carry_node

def setup_ripple_carry_adder(cg):
    A = [cg.add_node("input", f"A{i}") for i in range(4)]
    B = [cg.add_node("input", f"B{i}") for i in range(4)]
    cin = cg.add_node("input", "Cin")
    sum_ports, carry_port = ripple_carry_adder(cg, [a.ports[0] for a in A], [b.ports[0] for b in B], cin.ports[0])
    for i, s_port in enumerate(sum_ports):
        sum_node = cg.add_node("output", f"sum_{i}")
        cg.add_edge(s_port, sum_node.ports[0])
    carry_node = cg.add_node("output", "carry")
    cg.add_edge(carry_port, carry_node.ports[0])
    return
    A = [cg.add_input(f"A{i}") for i in range(4)]
    B = [cg.add_input(f"B{i}") for i in range(4)]
    Cin = cg.add_input("Cin")
    sum_outputs, carry_out = ripple_carry_adder(cg, A, B, Cin)
    for sum in sum_outputs:
        cg.add_output(sum, "sum")
    cg.add_output(carry_out, "carry")

def setup_carry_look_ahead_adder(cg):
    A = [cg.add_node("input", f"A{i}") for i in range(4)]
    B = [cg.add_node("input", f"B{i}") for i in range(4)]
    cin = cg.add_node("input", "Cin")
    sum_outputs, carry_out = carry_look_ahead_adder(cg, [a.ports[0] for a in A], [b.ports[0] for b in B], cin.ports[0])
    for sum in sum_outputs:
        cg.add_node("output", "SUM", inputs=[sum])
    cg.add_node("output", "CARRY", inputs=[carry_out])
    return
    A = [cg.add_input(f"A{i}") for i in range(4)]
    B = [cg.add_input(f"B{i}") for i in range(4)]
    Cin = cg.add_input("Cin")
    sum_outputs, carry_out = carry_look_ahead_adder(cg, A, B, Cin)
    for sum in sum_outputs:
        cg.add_output(sum, "sum")
    cg.add_output(carry_out, "carry")

def setup_wallace_tree_multiplier(cg):
    A = [cg.add_node("input", f"A{i}") for i in range(2)]
    B = [cg.add_node("input", f"B{i}") for i in range(2)]
    outputs = wallace_tree_multiplier(cg, [a.ports[0] for a in A], [b.ports[0] for b in B])
    for out in outputs:
        cg.add_node("output", "PRODUCT", inputs=[out])
    return

# def setup_four_bit_wallace_tree_multiplier(cg):
#    A = [cg.add_input(f"A{i}") for i in range(4)]
#    B = [cg.add_input(f"B{i}") for i in range(4)]
#    Cin = cg.add_input("cin")
#    sum_outputs, carry_out = four_bit_wallace_tree_multiplier(cg, A, B, Cin)
#    for i, sum in enumerate(sum_outputs):
#        cg.add_output(sum, f"p_{i}")
#    cg.add_output(carry_out, f"p_{len(sum_outputs)}")
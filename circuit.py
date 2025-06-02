import utils
import graph

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

def half_adder(circuit, x, y, parent_group=None):
    ha_group = circuit.add_group("HA_GROUP")
    ha_group.set_parent(parent_group)
    xor_gate = circuit.add_node("xor", "HA_XOR", inputs=[x, y], group_id=ha_group.id)
    and_gate = circuit.add_node("and", "HA_AND", inputs=[x, y], group_id=ha_group.id)
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

def adder_tree_recursive(circuit, summand_lists, zero):
    # carry does not work, pseudo non deterministic behaviour
    if len(summand_lists) == 1:
        return summand_lists[0], zero
    
    if len(summand_lists) == 2:
        sums, carry = ripple_carry_adder(circuit, summand_lists[0], summand_lists[1], zero)
        return sums, carry
    
    mid = len(summand_lists) // 2
    left_sums, left_carry = adder_tree_recursive(circuit, summand_lists[:mid], zero)
    right_sums, right_carry = adder_tree_recursive(circuit, summand_lists[mid:], zero)
    sums, carry = ripple_carry_adder(circuit, left_sums, right_sums, zero)
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

# if condition is one then zero it
def conditional_zeroing(circuit, x_list, cond):
    ports = []
    not_cond_node = circuit.add_node("not", "NOT", inputs=[cond])
    not_cond_port = not_cond_node.ports[1]
    for x in x_list:
        and_node = circuit.add_node("and", "AND", inputs=[x, not_cond_port])
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

def precompute_a_i(const_zero, const_one, int_m, n):
    print("int_m ", int_m)
    print("n ", n)
    a_i_lists = []
    for i in range(n):
        print("index ", i)
        calc = (2**i) % int_m
        print("calc ", calc)
        a = []
        for j in range(n):
            if calc % 2 == 0:
                a.append(const_zero)
                print("0")
            else:
                a.append(const_one)
                print("1")
            calc >>= 1
        a_i_lists.append(a)
    return a_i_lists

def small_mod_lemma_4_1(circuit, x_list, m_list, int_m):

    n = len(x_list)

    print("m, n: ", int_m, n)

    input = circuit.add_node("input", "INPUT")
    const_zero = constant_zero(circuit, input.ports[0])
    const_one = constant_one(circuit, input.ports[0])

    # precompute constants: a_im = 2^i mod m values
    a_i_lists = precompute_a_i(const_zero, const_one, int_m, n)

    print("a_i_lists", a_i_lists)
    
    # compute summands of y
    summands = []
    for ind, x_i in enumerate(x_list):
        not_x_i_node = circuit.add_node("not", "NOT", inputs=[x_i])
        summand = conditional_zeroing(circuit, a_i_lists[ind], not_x_i_node.ports[1])
        summands.append(summand)

    y, carry = adder_tree_recursive(circuit, summands, const_zero)

    print("y: ", y)

    results = []
    for i in range(n):
        bin_i = utils.int2binlist(i, bit_len=len(x_list))
        coef = [const_zero if bit == 0 else const_one for bit in bin_i]
        print("coef len", len(coef))
        print("m_list len", len(m_list))
        mult_m = wallace_tree_multiplier(circuit, m_list, coef)
        mult_m = mult_m[:-(len(mult_m)//2)]

        # mult_m should not be greater than y since y - mult_m should be in [0, m[
        _, _, greater = n_bit_comparator(circuit, mult_m, y)

        # mult_m_plus_m should not be less than y since y - mult_m should be in [0, m[
        print("len mult_m ", len(mult_m))
        print("len m_list ", len(m_list))
        mult_m_plus_m, _ = ripple_carry_adder(circuit, mult_m, m_list, const_zero)
        print("len mult_m_plus_m ", len(mult_m_plus_m))
        print("len y ", len(y))
        less, _, _ = n_bit_comparator(circuit, mult_m_plus_m, y)

        negative_mult_m = two_complement(circuit, mult_m)
        diff, carry = ripple_carry_adder(circuit, y, negative_mult_m, const_zero)

        result = conditional_zeroing(circuit, diff, greater)
        result = conditional_zeroing(circuit, diff, less)

        # always a zero list gets appended if the conditions are not fullfilled, else the result (x mod m) gets appended as a list
        results.append(result)

        #negative_mult_m = two_complement(circuit, mult_m)
        
        #sum, carry = ripple_carry_adder(circuit, y, negative_mult_m, const_zero)

        #is_negative = sign_detector(circuit, sum) # if 1 then negative
        #result = conditional_zeroing(circuit, coef, is_negative)
        #less, equals, greater = n_bit_comparator(circuit, sum, m_list)
        #not_less_node = circuit.add_node("not", "NOT", inputs=[less])
        #result = conditional_zeroing(circuit, result, not_less_node.ports[1])
        #results.append(result)

    #final = []
    #for i in range(n):
    #    for j in range(len(results)):
    #        curr_list = []
    #        curr_list.append(results[j][i])
    #    bit = or_tree_recursive(circuit, curr_list)
    #    final.append(bit)
    sums, carry = adder_tree_recursive(circuit, results, const_zero)
    return sums

# 1-bit shift to the left without carry
def one_left_shift(circuit, x_list):
    result = []
    result.append(constant_zero(circuit, x_list[0]))
    for x in x_list[:-1]:
        result.append(x)
    return result

# 1-bit shift to the right
def one_right_shift(circuit, x_list):
    result = []
    for x in x_list[1:]:
        result.append(x)
    result.append(constant_zero(circuit, x_list[len(x_list)-1]))
    return result

# check how to handle too large amounts
def n_left_shift(circuit, x_list, amount):
    # assert len(x_list) == len(amount), "x_list and amount must have the same number of bits"
    n = len(x_list) if len(x_list) < len(amount) else len(amount)
    current = x_list
    # depending on the signal (bit of amount), do a shift or not
    for i in range(n):
        #shift by 1, 2, 4, 8, ...
        shift_amount = 2**i
        shifted = []
        if shift_amount <= n:
            for k in range(shift_amount):
                shifted.append(constant_zero(circuit, current[0])) # filling the lsb zeros
            for j in range(n - shift_amount):
                shifted.append(current[j])
        else:
            for m in range(n):
                shifted.append(constant_zero(circuit, current[0]))
        assert len(shifted) == len(current)
        next_current = []
        for l in range(n):
            next_current.append(mux2(circuit, amount[i], shifted[l], current[l]))
        current = next_current
    return current

def n_right_shift(circuit, x_list, amount):
    assert len(x_list) == len(amount), "x_list and amount must have the same number of bits"
    n = len(x_list)
    current = x_list
    for i in range(n):
        shift_amount = 2**i
        shifted = []
        if shift_amount <= n:
            for j in range(shift_amount, n):
                shifted.append(current[j])
            for k in range(shift_amount):
                shifted.append(constant_zero(circuit, current[0]))
        else:
            for m in range(n):
                shifted.append(constant_zero(circuit, current[0]))
        assert len(shifted) == len(current)
        next_current = []
        for l in range(n):
            next_current.append(mux2(circuit, amount[i], shifted[l], current[l]))
        current = next_current
    return current

# if signal then a else b
def mux2(circuit, signal, a, b):
    not_signal = circuit.add_node("not", "MUX_NOT", inputs=[signal])
    not_signal_port = not_signal.ports[1]
    first_and = circuit.add_node("and", "MUX_AND", inputs=[signal, a])
    second_and = circuit.add_node("and", "MUX_AND", inputs=[not_signal_port, b])
    first_and_port = first_and.ports[2]
    second_and_port = second_and.ports[2]
    out_node = circuit.add_node("or", "MUX_OR", inputs=[first_and_port, second_and_port])
    out_port = out_node.ports[2]
    return out_port

def log2_estimate(circuit, x_list):
    n = len(x_list)
    
    zero = constant_zero(circuit, x_list[0])


    result_bits = [zero] * n
    found_any_one = zero

    for i in range(n):
        bit_position = n - 1 - i
        current_bit = x_list[bit_position]
        is_first_one = circuit.add_node("and", f"IS_FIRST_ONE_{i}", inputs=[current_bit, circuit.add_node("not", f"NOT_FOUND_{i}", inputs=[found_any_one]).ports[1]]).ports[2]
        found_any_one = circuit.add_node("or", f"FOUND_UPDATE_{i}", inputs=[found_any_one, current_bit]).ports[2]

        for j in range(n):
            bit_j_of_position = 1 if (bit_position >> j) & 1 else 0
            if bit_j_of_position == 1:
                result_bits[j] = circuit.add_node("or", f"RESULT_BIT_{j}_{bit_position}", inputs=[result_bits[j], is_first_one]).ports[2]

    return result_bits

def reciprocal_newton_raphson(circuit, m_bits, n):
    # Computes 1 / m using Newton-Raphson method.
    # x_{i+1} = x_i * (2 - m * x_i)
    # Converges to 1/m if x_0 is a good initial approximation.

    x_0 = initial_approximation(circuit, m_bits, n)
    m_x0 = fixed_point_multiply(circuit, m_bits, x_0, n)
    two_minus_mx0 = fixed_point_subtract_from_two(circuit, m_x0, n)
    x_1 = fixed_point_multiply(circuit, x_0, two_minus_mx0, n)
    return x_1

def scaled_multiply(circuit, a_bits, b_bits, n):
    full_product = wallace_tree_multiplier(circuit, a_bits, b_bits)
    if len(full_product) >= n + n//2:
        result = full_product[n//2:n//2+n]
    else:
        zero = constant_zero(circuit, a_bits[0])
        result = [zero] * (n//2) + full_product[:n//2]
        result = result[:n]
    return result

def fixed_point_multiply(circuit, a_bits, b_bits, n):
    product = wallace_tree_multiplier(circuit, a_bits, b_bits)
    result = product[n:3*n]
    return result

def fixed_point_subtract_from_two(circuit, a_bits, bit_length):
    zero = constant_zero(circuit, a_bits[0])
    one = constant_one(circuit, a_bits[0])
    two_fixed_point = [zero] * bit_length
    two_fixed_point[bit_length//2+1] = one
    result = subtract(circuit, two_fixed_point, a_bits)
    return result

def initial_approximation(circuit, m_bits, n):
    log2_m = log2_estimate(circuit, m_bits)
    zero = constant_zero(circuit, m_bits[0])
    one = constant_one(circuit, m_bits[0])
    n_bits = [zero] * n
    n_binary = bin(n)[2:]
    for i, bit in enumerate(reversed(n_binary)):
        if bit == '1':
            n_bits[i] = one

    shift_amount = subtract(circuit, n_bits, log2_m)
    one_fixed_point = [zero] * n
    one_fixed_point[n//2] = one
    initial_approx = n_left_shift(circuit, one_fixed_point, shift_amount)
    return initial_approx

def subtract(circuit, a_bits, b_bits):
    # a - b
    zero = constant_zero(circuit, a_bits[0])
    n = len(a_bits)
    b_complement = two_complement(circuit, b_bits)
    result, carry = ripple_carry_adder(circuit, a_bits, b_complement, zero)
    return result

def conditional_subtract(circuit, x_bits, m_bits, select):
    n = len(x_bits)
    assert len(m_bits) == n, "Both inputs must have the same bit length"
    difference = subtract(circuit, x_bits, m_bits)
    result = [None] * n
    for i in range(n):
        not_select = circuit.add_node("not", f"NOT_SEL_{i}", inputs=[select]).ports[1]
        and1 = circuit.add_node("and", f"AND_DIFF_{i}", inputs=[select, difference[i]]).ports[2]
        and2 = circuit.add_node("and", f"AND_X_{i}", inputs=[not_select, x_bits[i]]).ports[2]
        result[i] = circuit.add_node("or", f"OR_RES_{i}", inputs=[and1, and2]).ports[2]
    
    return result

def modulo_circuit(circuit, x_bits, m_bits):
    n = len(x_bits)
    assert len(m_bits) == n, "Both inputs must have the same bit length"
    m_powers = [m_bits]
    for i in range(1, n):
        prev_power = m_powers[i-1]
        next_power = [constant_zero(circuit, prev_power[0])] + prev_power[:-1]
        m_powers.append(next_power)
    current_remainder = x_bits
    for i in range(len(m_powers)-1, -1, -1):
        power_m = m_powers[i]
        less, equals, greater = n_bit_comparator(circuit, current_remainder, power_m)
        can_subtract = circuit.add_node("not", "NOT", inputs=[less]).ports[1]
        current_remainder = conditional_subtract(circuit, current_remainder, power_m, can_subtract)
    return current_remainder

def modular_exponentiation(circuit, base, exponent, modulus):
    n = len(base)
    assert n == len(exponent) and n == len(modulus), "All input must have the same bit length"

    zero = constant_zero(circuit, base[0])
    one = constant_one(circuit, base[0])

    result = [zero] * n
    result[0] = one
    base_mod = modulo_circuit(circuit, base, modulus)
    for i in range(n):
        bit_pos = n - 1 - i
        current_bit = exponent[bit_pos]
        squared = wallace_tree_multiplier(circuit, result, result)
        squared = squared[:len(base)]
        squared_mod = modulo_circuit(circuit, squared, modulus)
        with_multiply = wallace_tree_multiplier(circuit, squared_mod, base_mod)
        with_multiply = with_multiply[:len(base)]
        multiply_mod = modulo_circuit(circuit, with_multiply, modulus)
        new_result = [None] * n
        for j in range(n):
            not_bit = circuit.add_node("not", f"NOT_BIT_{bit_pos}_{j}", inputs=[current_bit]).ports[1]
            and1 = circuit.add_node("and", f"AND_MULT_{bit_pos}_{j}", 
                                  inputs=[current_bit, multiply_mod[j]]).ports[2]
            and2 = circuit.add_node("and", f"AND_SQR_{bit_pos}_{j}", 
                                  inputs=[not_bit, squared_mod[j]]).ports[2]
            new_result[j] = circuit.add_node("or", f"OR_RESULT_{bit_pos}_{j}", 
                                           inputs=[and1, and2]).ports[2]
        result = new_result
    return result

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
    "precompute_a_i": lambda cg: setup_precompute_a_i(cg),
    "conditional_zeroing": lambda cg: setup_conditional_zeroing(cg),
    "one_left_shift": lambda cg: setup_one_left_shift(cg),
    "one_right_shift": lambda cg: setup_one_right_shift(cg),
    "n_left_shift": lambda cg: setup_n_left_shift(cg),
    "n_right_shift": lambda cg: setup_n_right_shift(cg),
    "log2_estimate": lambda cg: setup_log2_estimate(cg),
    "reciprocal_newton_raphson": lambda cg: setup_reciprocal_newton_raphson(cg),
    "modulo_circuit": lambda cg: setup_modulo_circuit(cg),
    "modular_exponentiation": lambda cg: setup_modular_exponentiation(cg),
}

def setup_modular_exponentiation(cg):
    B = [cg.add_node("input", f"B_{i}") for i in range(4)]
    E = [cg.add_node("input", f"E_{i}") for i in range(4)]
    M = [cg.add_node("input", f"M_{i}") for i in range(4)]
    OUT = modular_exponentiation(cg, [b.ports[0] for b in B], [e.ports[0] for e in E], [m.ports[0] for m in M])
    OUT_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)]
    return B, E, M, OUT_NODES

def setup_modulo_circuit(cg):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    A = [cg.add_node("input", f"A_{i}") for i in range(4)]
    OUT = modulo_circuit(cg, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)]
    return X, A, OUT_NODES

def setup_reciprocal_newton_raphson(cg):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    OUT = reciprocal_newton_raphson(cg, [x.ports[0] for x in X], 4)
    OUT_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)]
    return X, OUT_NODES

def setup_log2_estimate(cg):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    OUT = log2_estimate(cg, [x.ports[0] for x in X])
    OUT_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)]
    return X, OUT_NODES

def setup_n_left_shift(cg):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    A = [cg.add_node("input", f"A_{i}") for i in range(4)]
    OUT = n_left_shift(cg, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)]
    return X, A, OUT_NODES

def setup_n_right_shift(cg):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    A = [cg.add_node("input", f"A_{i}") for i in range(4)]
    OUT = n_right_shift(cg, [x.ports[0] for x in X], [a.ports[0] for a in A])
    OUT_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)]
    return X, A, OUT_NODES

def setup_one_left_shift(cg):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    OUT = one_left_shift(cg, [x.ports[0] for x in X])
    OUT_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)]
    return X, OUT_NODES

def setup_one_right_shift(cg):
    X = [cg.add_node("input", f"X_{i}") for i in range(4)]
    OUT = one_right_shift(cg, [x.ports[0] for x in X])
    OUT_NODES = [cg.add_node("output", f"OUT_{i}", inputs=[o]) for i, o in enumerate(OUT)]
    return X, OUT_NODES

def setup_small_mod_lemma_4_1(cg):
    X = [cg.add_node("input", f"X{i}") for i in range(4)]
    M = [cg.add_node("input", f"M{i}") for i in range(4)]
    # in_node = cg.add_node("input", f"IN")
    # build M from int
    # int_m = 3
    # bin_list_m = utils.int2binlist(int_m, 4)
    # print("bin_list_m shape: ", len(bin_list_m))
    # M = [constant_one(cg, in_node.ports[0]) if bit else constant_zero(cg, in_node.ports[0]) for bit in bin_list_m]
    outputs = small_mod_lemma_4_1(cg, [x.ports[0] for x in X], M, 2)
    out_nodes = []
    for out in outputs:
        out_node = cg.add_node("output", "REMAINDER", inputs=[out])
        out_nodes.append(out_node)
    return X, out_nodes

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
    L = cg.add_node("output", "LESS", inputs=[less])
    E = cg.add_node("output", "EQUALS", inputs=[equals])
    G = cg.add_node("output", "GREATER", inputs=[greater])
    return A, B, L, E, G

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
    root_group = cg.add_group("ROOT_GROUP")
    A = cg.add_node("input", "A")
    B = cg.add_node("input", "B")
    sum_port, carry_port = half_adder(cg, A.ports[0], B.ports[0], parent_group=root_group)
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
    sum_nodes = []
    for sum in sum_outputs:
        sum_nodes.append(cg.add_node("output", "SUM", inputs=[sum]))
    carry_node = cg.add_node("output", "CARRY", inputs=[carry_out])
    return A, B, cin, sum_nodes, carry_node

def setup_wallace_tree_multiplier(cg):
    A = [cg.add_node("input", f"A{i}") for i in range(4)]
    B = [cg.add_node("input", f"B{i}") for i in range(4)]
    outputs = wallace_tree_multiplier(cg, [a.ports[0] for a in A], [b.ports[0] for b in B])
    output_nodes = []
    for out in outputs:
        output_nodes.append(cg.add_node("output", "PRODUCT", inputs=[out]))
    return A, B, output_nodes

def setup_precompute_a_i(cg):
    input_node = cg.add_node("input", "INPUT")
    input_node_port = input_node.ports[0]
    zero_port = constant_zero(cg, input_node_port)
    one_port = constant_one(cg, input_node_port)
    out_port = precompute_a_i(zero_port, one_port, 2, 4)
    output_nodes = []
    for i, number in enumerate(out_port):
        for j, digit in enumerate(number):
            output_node = cg.add_node("output", f"OUT_{i}_{j}", inputs=[digit])
            output_nodes.append(output_node)
    return output_nodes

def setup_conditional_zeroing(cg):
    X = [cg.add_node("input", f"A{i}") for i in range(4)]
    C = cg.add_node("input", f"COND")
    zero_port = constant_zero(cg, C.ports[0])
    one_port = constant_one(cg, C.ports[0])
    output = conditional_zeroing(cg, [x.ports[0] for x in X], C.ports[0])
    output_nodes = []
    for i, out in enumerate(output):
        output_node = cg.add_node("output", f"OUT_{i}", inputs=[out])
        output_nodes.append(output_node)
    return X, C, output_nodes
# def setup_four_bit_wallace_tree_multiplier(cg):
#    A = [cg.add_input(f"A{i}") for i in range(4)]
#    B = [cg.add_input(f"B{i}") for i in range(4)]
#    Cin = cg.add_input("cin")
#    sum_outputs, carry_out = four_bit_wallace_tree_multiplier(cg, A, B, Cin)
#    for i, sum in enumerate(sum_outputs):
#        cg.add_output(sum, f"p_{i}")
#    cg.add_output(carry_out, f"p_{len(sum_outputs)}")
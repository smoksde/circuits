import unittest
from circuits import *
from graph import *
from formula import *
from utils import int2binlist, iter_random_bin_list
import sanity
from node import Node
from port import Port
from edge import Edge
import json
from io import StringIO
import random
import software_beame as sb


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCircuitSimulation)

    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    output = {
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "wasSuccessful": result.wasSuccessful(),
        "details": stream.getvalue(),
    }

    return json.dumps(output)


class TestCircuitSimulation(unittest.TestCase):

    def test_half_adder(self):
        circuit = CircuitGraph()
        a, b, sum_out, carry_out = setup_half_adder(circuit)

        expected = {
            (0, 0): (0, 0),
            (0, 1): (1, 0),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
        }
        for val_a, val_b in expected:
            circuit.node_values[str(a.node_id)] = val_a
            circuit.node_values[str(b.node_id)] = val_b
            circuit.simulate()
            self.assertEqual(
                circuit.get_port_value(sum_out.ports[0]), expected[(val_a, val_b)][0]
            )
            self.assertEqual(
                circuit.get_port_value(carry_out.ports[0]), expected[(val_a, val_b)][1]
            )

    def test_full_adder(self):
        circuit = CircuitGraph()
        a, b, cin, sum_out, carry_out = setup_full_adder(circuit)
        expected = {
            (0, 0, 0): (0, 0),
            (0, 0, 1): (1, 0),
            (0, 1, 0): (1, 0),
            (0, 1, 1): (0, 1),
            (1, 0, 0): (1, 0),
            (1, 0, 1): (0, 1),
            (1, 1, 0): (0, 1),
            (1, 1, 1): (1, 1),
        }
        for val_a, val_b, val_cin in expected:
            circuit.node_values[str(a.node_id)] = val_a
            circuit.node_values[str(b.node_id)] = val_b
            circuit.node_values[str(cin.node_id)] = val_cin
            circuit.simulate()
            self.assertEqual(
                circuit.get_port_value(sum_out.ports[0]),
                expected[(val_a, val_b, val_cin)][0],
            )
            self.assertEqual(
                circuit.get_port_value(carry_out.ports[0]),
                expected[(val_a, val_b, val_cin)][1],
            )

    def test_carry_look_ahead_adder(self):
        circuit = CircuitGraph()
        bit_len = 4
        A, B, cin, sums, carry = setup_carry_look_ahead_adder(circuit, bit_len=bit_len)
        for i in range(100):
            rand_a = random.randrange(2**bit_len - 1)
            rand_b = random.randrange(2**bit_len - 1)
            rand_cin = random.randrange(2)
            expected_num = rand_a + rand_b + rand_cin
            expected_bin_list = int2binlist(expected_num, bit_len=bit_len + 1)
            a_bin_list = int2binlist(rand_a, bit_len=bit_len)
            b_bin_list = int2binlist(rand_b, bit_len=bit_len)
            cin_bin_list = int2binlist(rand_cin, bit_len=1)
            circuit.fill_node_values_via_ports(A, a_bin_list)
            circuit.fill_node_values_via_ports(B, b_bin_list)
            circuit.node_values[str(cin.node_id)] = cin_bin_list[0]
            circuit.simulate()

            for idx, e in enumerate(expected_bin_list[:-1]):
                self.assertEqual(circuit.get_port_value(sums[idx].ports[0]), e)
            self.assertEqual(
                circuit.get_port_value(carry.ports[0]), expected_bin_list[-1]
            )

    def test_wallace_tree_multiplier(self):
        circuit = CircuitGraph()
        bit_len = 4
        A, B, outputs = setup_wallace_tree_multiplier(circuit, bit_len=bit_len)
        for i in range(10):
            rand_a = random.randrange(2**bit_len - 1)
            rand_b = random.randrange(2**bit_len - 1)
            expected_num = rand_a * rand_b
            expected_bin_list = int2binlist(expected_num, bit_len=2 * bit_len)
            a_bin_list = int2binlist(rand_a, bit_len=bit_len)
            b_bin_list = int2binlist(rand_b, bit_len=bit_len)
            for idx, a in enumerate(A):
                circuit.node_values[str(a.node_id)] = a_bin_list[idx]
            for idx, b in enumerate(B):
                circuit.node_values[str(b.node_id)] = b_bin_list[idx]
            circuit.simulate()

            for idx, e in enumerate(expected_bin_list):
                self.assertEqual(circuit.get_port_value(outputs[idx].ports[0]), e)

    def test_conditional_zeroing(self):
        circuit = CircuitGraph()
        bit_len = 4
        X, C, O = setup_conditional_zeroing(circuit, bit_len=bit_len)
        for i in range(40):
            rand_x = random.randrange((2**bit_len - 1) // 2)
            rand_c = random.randrange(2)
            x_bin_list = int2binlist(rand_x, bit_len=bit_len)
            c_bin_list = int2binlist(rand_c, bit_len=1)
            if rand_c == 1:
                expect_bin_list = int2binlist(0, bit_len=bit_len)
            else:
                expect_bin_list = int2binlist(rand_x, bit_len=bit_len)
            for idx, x in enumerate(X):
                circuit.node_values[str(x.node_id)] = x_bin_list[idx]
            circuit.node_values[str(C.node_id)] = c_bin_list[0]
            circuit.simulate()
            for idx, e in enumerate(expect_bin_list):
                self.assertEqual(
                    circuit.get_port_value(O[idx].ports[0]),
                    e,
                    msg=(
                        f"Mismatch\n"
                        f"rand_x = {rand_x}\n"
                        f"rand_c = {rand_c}\n"
                        f"INPUT = {x_bin_list}\n"
                        f"OUTPUT = {[circuit.get_port_value(O[idx].ports[0]) for idx in range(len(O))]}\n"
                    ),
                )

    def test_conditional_subtract(self):
        circuit = CircuitGraph()
        bit_len = 8
        A, B, C, O = setup_conditional_subtract(circuit, bit_len=bit_len)
        instances = 0
        while True:
            rand_a = random.randrange(2**bit_len - 1)
            rand_b = random.randrange(2**bit_len - 1)
            rand_c = random.randrange(2)
            if rand_a < rand_b:
                continue
            a_bin_list = int2binlist(rand_a, bit_len=bit_len)
            b_bin_list = int2binlist(rand_b, bit_len=bit_len)
            c_bin_list = int2binlist(rand_c, bit_len=bit_len)
            if rand_c == 1:
                expect_bin_list = int2binlist(rand_a - rand_b, bit_len=bit_len)
            else:
                expect_bin_list = int2binlist(rand_a, bit_len=bit_len)
            for idx, a in enumerate(A):
                circuit.node_values[str(a.node_id)] = a_bin_list[idx]
            for idx, b in enumerate(B):
                circuit.node_values[str(b.node_id)] = b_bin_list[idx]
            circuit.node_values[str(C.node_id)] = c_bin_list[0]
            circuit.simulate()
            for idx, e in enumerate(expect_bin_list):
                self.assertEqual(
                    circuit.get_port_value(O[idx].ports[0]),
                    e,
                    msg=(
                        f"Mismatch\n"
                        f"rand_a = {rand_a}\n"
                        f"rand_b = {rand_b}\n"
                        f"rand_c = {rand_c}\n"
                        f"OUTPUT = {[circuit.get_port_value(O[idx].ports[0]) for idx in range(len(O))]}\n"
                    ),
                )

            instances += 1
            if instances >= 10:
                break

    """def test_or_tree_recursive(self):
        circuit = CircuitGraph()
        bit_len = 4
        X, o = setup_or_tree_recursive(circuit, bit_len=bit_len)
        for i in range(100):
            rand_x = random.randrange((2**bit_len - 1))
            expect = 1 if rand_x > 0 else 0
            x_bin_list = int2binlist(rand_x, bit_len=bit_len)
            expect_bin_list = int2binlist(expect, bit_len=bit_len)
            for idx, x in enumerate(X):
                circuit.node_values[str(x.node_id)] = x_bin_list[idx]
            circuit.simulate()
            for idx, e in enumerate(expect_bin_list):
                self.assertEqual(circuit.get_port_value(o.ports[0]), e)
    """

    def test_or_tree_iterative(self):
        circuit = CircuitGraph()
        bit_len = 4
        X, o = setup_or_tree_iterative(circuit, bit_len=bit_len)
        for i in range(100):
            rand_x = random.randrange((2**bit_len - 1))
            expect = 1 if rand_x > 0 else 0
            x_bin_list = int2binlist(rand_x, bit_len=bit_len)
            e = int2binlist(expect, bit_len=1)
            for idx, x in enumerate(X):
                circuit.node_values[str(x.node_id)] = x_bin_list[idx]
            circuit.simulate()
            self.assertEqual(circuit.get_port_value(o.ports[0]), e[0])

    def test_one_left_shift(self):
        circuit = CircuitGraph()
        bit_len = 4
        X, OUT = setup_one_left_shift(circuit, bit_len=bit_len)
        for i in range(40):
            rand_x = random.randrange((2**bit_len - 1) // 2)
            expected_num = rand_x * 2
            x_bin_list = int2binlist(rand_x, bit_len=bit_len)
            expected_bin_list = int2binlist(expected_num, bit_len=bit_len)
            for idx, x in enumerate(X):
                circuit.node_values[str(x.node_id)] = x_bin_list[idx]
            circuit.simulate()
            for idx, e in enumerate(expected_bin_list):
                self.assertEqual(circuit.get_port_value(OUT[idx].ports[0]), e)

    def test_one_right_shift(self):
        circuit = CircuitGraph()
        bit_len = 4
        X, OUT = setup_one_right_shift(circuit, bit_len=bit_len)
        for i in range(40):
            rand_x = random.randrange((2**bit_len - 1))
            expected_num = rand_x // 2
            x_bin_list = int2binlist(rand_x, bit_len=bit_len)
            expected_bin_list = int2binlist(expected_num, bit_len=bit_len)
            for idx, x in enumerate(X):
                circuit.node_values[str(x.node_id)] = x_bin_list[idx]
            circuit.simulate()
            for idx, e in enumerate(expected_bin_list):
                self.assertEqual(circuit.get_port_value(OUT[idx].ports[0]), e)

    def test_n_left_shift(self):
        circuit = CircuitGraph()
        bit_len = 8
        X, A, OUT = setup_n_left_shift(circuit, bit_len=bit_len)
        instances = 0
        while True:
            rand_x = random.randrange((2**bit_len - 1))
            rand_a = random.randrange(bit_len)
            expected_num = rand_x << rand_a
            if expected_num > 2**bit_len - 1:
                continue
            else:
                x_bin_list = int2binlist(rand_x, bit_len=bit_len)
                a_bin_list = int2binlist(rand_a, bit_len=bit_len)
                expected_bin_list = int2binlist(expected_num, bit_len=bit_len)
                for idx, x in enumerate(X):
                    circuit.node_values[str(x.node_id)] = x_bin_list[idx]
                for idx, a in enumerate(A):
                    circuit.node_values[str(a.node_id)] = a_bin_list[idx]
                circuit.simulate()
                for idx, e in enumerate(expected_bin_list):
                    self.assertEqual(circuit.get_port_value(OUT[idx].ports[0]), e)
                instances += 1
                if instances >= 40:
                    break

    def test_n_right_shift(self):
        circuit = CircuitGraph()
        bit_len = 4
        X, A, OUT = setup_n_right_shift(circuit, bit_len=bit_len)
        instances = 0
        while True:
            rand_x = random.randrange((2**bit_len - 1))
            rand_a = random.randrange(bit_len)
            expected_num = rand_x >> rand_a
            x_bin_list = int2binlist(rand_x, bit_len=bit_len)
            a_bin_list = int2binlist(rand_a, bit_len=bit_len)
            expected_bin_list = int2binlist(expected_num, bit_len=bit_len)
            for idx, x in enumerate(X):
                circuit.node_values[str(x.node_id)] = x_bin_list[idx]
            for idx, a in enumerate(A):
                circuit.node_values[str(a.node_id)] = a_bin_list[idx]
            circuit.simulate()
            for idx, e in enumerate(expected_bin_list):
                self.assertEqual(
                    circuit.get_port_value(OUT[idx].ports[0]),
                    e,
                    msg=(f"Mismatch" f"rand_x = {rand_x}" f"rand_a = {rand_a}"),
                )
            instances += 1
            if instances >= 40:
                break

    def test_n_bit_comparator(self):
        circuit = CircuitGraph()
        bit_len = 8
        A, B, L, E, G = setup_n_bit_comparator(circuit, bit_len=bit_len)
        for i in range(40):
            rand_a = random.randrange(2**bit_len - 1)
            rand_b = random.randrange(2**bit_len - 1)
            a_bin_list = int2binlist(rand_a, bit_len=bit_len)
            b_bin_list = int2binlist(rand_b, bit_len=bit_len)
            if rand_a < rand_b:
                expect_L = int2binlist(1, bit_len=1)[0]
                expect_E = int2binlist(0, bit_len=1)[0]
                expect_G = int2binlist(0, bit_len=1)[0]
            elif rand_a == rand_b:
                expect_L = int2binlist(0, bit_len=1)[0]
                expect_E = int2binlist(1, bit_len=1)[0]
                expect_G = int2binlist(0, bit_len=1)[0]
            else:
                expect_L = int2binlist(0, bit_len=1)[0]
                expect_E = int2binlist(0, bit_len=1)[0]
                expect_G = int2binlist(1, bit_len=1)[0]
            for idx, a in enumerate(A):
                circuit.node_values[str(a.node_id)] = a_bin_list[idx]
            for idx, b in enumerate(B):
                circuit.node_values[str(b.node_id)] = b_bin_list[idx]
            circuit.simulate()
            self.assertEqual(circuit.get_port_value(L.ports[0]), expect_L)
            self.assertEqual(circuit.get_port_value(E.ports[0]), expect_E)
            self.assertEqual(circuit.get_port_value(G.ports[0]), expect_G)

    def test_n_bit_equality(self):
        circuit = CircuitGraph()
        bit_len = 8
        n = bit_len
        A, B, EQUALS = setup_n_bit_equality(circuit, bit_len=bit_len)
        for _ in range(40):
            rand_a = random.randrange(2**n - 1)
            rand_b = random.randrange(2**n - 1)
            rand_set_equals = random.randrange(2)
            if rand_set_equals:
                rand_b = rand_a
            circuit.fill_node_values(A, int2binlist(rand_a, bit_len=n))
            circuit.fill_node_values(B, int2binlist(rand_b, bit_len=n))

            circuit.simulate()

            EQUALS_PORT = circuit.get_output_node_port(EQUALS)
            got = circuit.compute_value_from_ports([EQUALS_PORT])
            if rand_a == rand_b:
                expect = 1
            else:
                expect = 0
            self.assertEqual(got, expect)

    def test_modulo_circuit(self):
        circuit = CircuitGraph()
        bit_len = 4
        X, A, OUT_NODES = setup_modulo_circuit(circuit, bit_len=bit_len)
        for i in range(100):
            rand_x = random.randrange(2**bit_len - 1)
            rand_a = random.randrange(2 ** (bit_len // 2) - 1)
            if rand_a == 0:
                rand_a = 2
            if rand_a < 8:
                rand_a = 8
            expected_num = rand_x % rand_a
            expected_bin_list = int2binlist(expected_num, bit_len=bit_len)
            x_bin_list = int2binlist(rand_x, bit_len=bit_len)
            a_bin_list = int2binlist(rand_a, bit_len=bit_len)
            for idx, x in enumerate(X):
                circuit.node_values[str(x.node_id)] = x_bin_list[idx]
            for idx, a in enumerate(A):
                circuit.node_values[str(a.node_id)] = a_bin_list[idx]
            circuit.simulate()
            for idx, e in enumerate(expected_bin_list):
                self.assertEqual(
                    circuit.get_port_value(OUT_NODES[idx].ports[0]),
                    e,
                    msg=(
                        f"rand_x: {rand_x}\n"
                        f"rand_a: {rand_a}\n"
                        f"got: {[circuit.get_port_value(o.ports[0]) for o in OUT_NODES]}"
                        f"expect: {expected_bin_list}"
                    ),
                )

    def test_montgomery_ladder(self):
        circuit = CircuitGraph()
        bit_len = 4
        B, E, M, OUT_NODES = setup_montgomery_ladder(circuit, bit_len=bit_len)
        for i in range(1):
            rand_b = random.randrange(2**bit_len - 1)
            rand_e = random.randrange(2**bit_len - 1)
            rand_m = random.randrange(2 ** (bit_len // 2) - 1)
            if rand_m == 0:
                rand_m = 2
            if rand_m < 8:
                rand_m = 8
            expected_num = (rand_b**rand_e) % rand_m
            expected_bin_list = int2binlist(expected_num, bit_len=bit_len)
            b_bin_list = int2binlist(rand_b, bit_len=bit_len)
            e_bin_list = int2binlist(rand_e, bit_len=bit_len)
            m_bin_list = int2binlist(rand_m, bit_len=bit_len)
            for idx, b in enumerate(B):
                circuit.node_values[str(b.node_id)] = b_bin_list[idx]
            for idx, e in enumerate(E):
                circuit.node_values[str(e.node_id)] = e_bin_list[idx]
            for idx, m in enumerate(M):
                circuit.node_values[str(m.node_id)] = m_bin_list[idx]
            circuit.simulate()
            for idx, ex in enumerate(expected_bin_list):
                self.assertEqual(circuit.get_port_value(OUT_NODES[idx].ports[0]), ex)

    def test_modular_exponentiation(self):
        circuit = CircuitGraph()
        bit_len = 4
        B, E, M, OUT_NODES = setup_modular_exponentiation(circuit, bit_len=bit_len)
        for i in range(1):
            rand_b = random.randrange(2**bit_len - 1)
            rand_e = random.randrange(2**bit_len - 1)
            rand_m = random.randrange(2 ** (bit_len // 2) - 1)
            if rand_m == 0:
                rand_m = 2
            if rand_m < 8:
                rand_m = 8
            expected_num = (rand_b**rand_e) % rand_m
            expected_bin_list = int2binlist(expected_num, bit_len=bit_len)
            b_bin_list = int2binlist(rand_b, bit_len=bit_len)
            e_bin_list = int2binlist(rand_e, bit_len=bit_len)
            m_bin_list = int2binlist(rand_m, bit_len=bit_len)
            for idx, b in enumerate(B):
                circuit.node_values[str(b.node_id)] = b_bin_list[idx]
            for idx, e in enumerate(E):
                circuit.node_values[str(e.node_id)] = e_bin_list[idx]
            for idx, m in enumerate(M):
                circuit.node_values[str(m.node_id)] = m_bin_list[idx]
            circuit.simulate()
            for idx, ex in enumerate(expected_bin_list):
                self.assertEqual(circuit.get_port_value(OUT_NODES[idx].ports[0]), ex)

    def test_next_power_of_two(self):
        circuit = CircuitGraph()
        bit_len = 4
        X, O = setup_next_power_of_two(circuit, bit_len=bit_len)
        for _ in range(20):
            rand_x = random.randrange(2 ** (bit_len - 1) - 1)
            x_bin_list = int2binlist(rand_x, bit_len=bit_len)

            """def next_power(n: int) -> int:
                if n < 1:
                    return 1
                return 1 << (n - 1).bit_length()
            """

            # Funtion returns next higher power of two in case n is already a power of two
            def next_power(n: int) -> int:
                if n < 1:
                    return 0
                # If n is already a power of two, bit_length won't help — so use n.bit_length()
                return 1 << n.bit_length()

            expect = next_power(rand_x)
            expect_bin_list = int2binlist(expect, bit_len=bit_len)
            for idx, x in enumerate(X):
                circuit.node_values[str(x.node_id)] = x_bin_list[idx]
            circuit.simulate()
            for idx, e in enumerate(expect_bin_list):
                self.assertEqual(
                    circuit.get_port_value(O[idx].ports[0]),
                    e,
                    msg=(
                        f"X: {rand_x}"
                        f"Expect: {expect_bin_list}"
                        f"OUTPUT: {[circuit.get_port_value(o.ports[0]) for o in O]}"
                    ),
                )

    def test_bus_multiplexer(self):
        circuit = CircuitGraph()
        bit_len = 4
        num_amount = 4
        sel_bit_width = int(math.log2(num_amount))
        BUS, S, O = setup_bus_multiplexer(
            circuit, num_amount=num_amount, bit_len=bit_len
        )
        for _ in range(20):
            NUMS = []
            rand_selector = random.randrange(2 ** (sel_bit_width - 1) - 1)
            rand_selector_bits = int2binlist(rand_selector, bit_len=sel_bit_width)
            expect_list = []
            for i in range(num_amount):
                rand_num = random.randrange(2 ** (bit_len - 1) - 1)
                rand_num_bits = int2binlist(rand_num, bit_len=bit_len)
                if i == rand_selector - 1:
                    expect_list = rand_num_bits
                NUMS.append(rand_num_bits)

            # FILL BUS VALUES
            for num_idx, num in enumerate(BUS):
                for idx, port in enumerate(num):
                    circuit.node_values[str(port.node_id)] = NUMS[num_idx][idx]

            # FILL SELECTOR
            for idx, s in enumerate(S):
                circuit.node_values[str(s.node_id)] = rand_selector_bits[idx]

            circuit.simulate()
            for idx, e in expect_list:
                self.assertEqual(circuit.get_port_value(O[idx].ports[0]), e)

    def test_tensor_multiplexer(self):
        circuit = CircuitGraph()
        bit_len = 4
        num_amount = 4
        sel_bit_width = int(math.log2(num_amount))
        TENSOR, S, O = setup_tensor_multiplexer(
            circuit, num_amount=num_amount, bit_len=bit_len
        )
        for _ in range(20):
            NUMS = []
            NUMS_BITS = []
            rand_selector = random.randrange(2 ** (sel_bit_width - 1) - 1)
            rand_selector_bits = int2binlist(rand_selector, bit_len=sel_bit_width)
            for i in range(num_amount):
                num_row = []
                num_row_bits = []
                for j in range(num_amount):
                    rand_num = random.randrange(2 ** (bit_len - 1) - 1)
                    rand_num_bits = int2binlist(rand_num, bit_len=bit_len)
                    num_row.append(rand_num)
                    num_row_bits.append(rand_num_bits)
                NUMS.append(num_row)
                NUMS_BITS.append(num_row_bits)

            # FILL TENSOR VALUES
            for i, row in enumerate(TENSOR):
                for num_idx, num in enumerate(row):
                    circuit.fill_node_values(num, NUMS_BITS[i][num_idx])

            # FILL SELECTOR VALUE
            circuit.fill_node_values(S, rand_selector_bits)

            circuit.simulate()

            # for idx, e in expect_list:
            #    self.assertEqual(circuit.get_port_value(O[idx].ports[0]), e)
            for i in range(num_amount):
                expect = utils.binlist2int(NUMS_BITS[rand_selector][i])
                value = circuit.compute_value_from_ports(
                    circuit.get_output_nodes_ports(O[i])
                )
                self.assertEqual(
                    value, expect, msg=(f"selector: {rand_selector}", f"NUMS: {NUMS}")
                )

    def test_precompute_aim(self):
        circuit = CircuitGraph()
        bit_len = 4
        n = bit_len
        output = setup_precompute_aim(circuit, bit_len=bit_len)
        sb_output = sb.precompute_aim(bit_len)
        circuit.simulate()

        # print("sb_output: ", sb_output)
        # for m in range(len(output)):
        #    for i in range(n):
        #        value = circuit.compute_value_from_ports(
        #            circuit.get_output_nodes_ports(output[m][i])
        #        )
        #        print(value, end="")
        #    print()

        for m in range(0, n):
            for i in range(n):
                e_bin_list = int2binlist(sb_output[m, i])
                for idx, e in enumerate(e_bin_list):
                    if m == 0:
                        continue
                    self.assertEqual(
                        circuit.get_port_value(output[m][i][idx].ports[0]), e
                    )

    def test_lemma_4_1_reduce_in_parallel(self):
        circuit = CircuitGraph()
        bit_len = 4
        n = bit_len
        Y, M, O = setup_lemma_4_1_reduce_in_parallel(circuit, bit_len=bit_len)

        y_cases = [3]
        m_cases = [3]

        for y_value, m_value in zip(y_cases, m_cases):
            y_bits = int2binlist(y_value, bit_len=n)
            m_bits = int2binlist(m_value, bit_len=n)
            circuit.fill_node_values(Y, y_bits)
            circuit.fill_node_values(M, m_bits)
            circuit.simulate()

            expect = y_value % m_value

            O_PORTS = circuit.get_output_nodes_ports(O)
            O_VALUE = circuit.compute_value_from_ports(O_PORTS)
            # print(f"y: {y_value}, m: {m_value}, expect: {expect}, got: {O_VALUE}")
            self.assertEqual(O_VALUE, expect)

        for i in range(40):
            rand_y = random.randrange(2 ** (n - 1) - 1)
            rand_y_bits = int2binlist(rand_y, bit_len=n)
            rand_m = random.randrange(1, n + 1)
            rand_m_bits = int2binlist(rand_m, bit_len=n)
            expect_num = rand_y % rand_m

            circuit.fill_node_values(Y, rand_y_bits)
            circuit.fill_node_values(M, rand_m_bits)

            circuit.simulate()

            sim_value = circuit.compute_value_from_ports(
                circuit.get_output_nodes_ports(O)
            )
            self.assertEqual(
                sim_value,
                expect_num,
                msg=(
                    f"y: {rand_y}",
                    f"m: {rand_m}",
                    f"sim_value: {sim_value}",
                    f"expect_num: {expect_num}",
                ),
            )

    def test_lemma_4_1_compute_diffs(self):
        circuit = CircuitGraph()
        bit_len = 4
        n = bit_len
        Y, M, O = setup_lemma_4_1_compute_diffs(circuit, bit_len=bit_len)

        y_cases = [3]
        m_cases = [3]

        for y_value, m_value in zip(y_cases, m_cases):
            y_bits = int2binlist(y_value, bit_len=n)
            m_bits = int2binlist(m_value, bit_len=n)
            circuit.fill_node_values(Y, y_bits)
            circuit.fill_node_values(M, m_bits)

            circuit.simulate()

            diffs = sb.lemma_4_1_compute_diffs(y_value, m_value, n)
            # print("diffs:")
            # print(diffs)

            O_VALUES = []

            for o in O:
                O_PORTS = circuit.get_output_nodes_ports(o)
                O_VALUE = circuit.compute_value_from_ports(O_PORTS)
                O_VALUES.append(O_VALUE)

            for idx, diff in enumerate(diffs):
                if diff < 0:
                    # print("diff is smaller than 0, break")
                    break
                self.assertEqual(
                    O_VALUES[idx],
                    diff,
                    msg=f"idx: {idx}, expect: {diff}, got: {O_VALUES[idx]}",
                )
                # print(f"idx: {idx}, expect: {diff}, got: {O_VALUES[idx]}")

    def test_lemma_4_1(self):
        circuit = CircuitGraph()
        bit_len = 4
        n = bit_len
        X, M, O = setup_lemma_4_1(circuit, bit_len=bit_len)
        X_CY, M_CY, Y_CY = setup_lemma_4_1_compute_y(circuit, bit_len=bit_len)

        x_cases = [6]
        m_cases = [3]
        for x_value, m_value in zip(x_cases, m_cases):
            x_bits = int2binlist(x_value, bit_len=n)
            m_bits = int2binlist(m_value, bit_len=n)

            expect_num = int(x_value % m_value)

            circuit.fill_node_values(X, x_bits)
            circuit.fill_node_values(M, m_bits)
            circuit.fill_node_values(X_CY, x_bits)
            circuit.fill_node_values(M_CY, m_bits)

            circuit.simulate()

            O_PORTS = circuit.get_output_nodes_ports(O)
            Y_CY_PORTS = circuit.get_output_nodes_ports(Y_CY)

            O_VALUE = circuit.compute_value_from_ports(O_PORTS)
            Y_CY_VALUE = circuit.compute_value_from_ports(Y_CY_PORTS)
            expect_value = int(sb.lemma_4_1_compute_y(x_value, m_value, n))
            # print(
            #    f"x: {x_value}, m: {m_value}, n: {n}, expect: {expect_value}, got: {Y_CY_VALUE}"
            # )

            self.assertEqual(
                O_VALUE,
                expect_num,
                msg=(
                    f"X: {x_value}",
                    f"M: {m_value}",
                    f"M_DECR: {m_value - 1}",
                    f"O_VALUE: {O_VALUE}",
                    f"EXPECT: {expect_num}",
                    f"Y_CY_VALUE: {Y_CY_VALUE}",
                ),
            )

        for i in range(20):
            rand_x = random.randrange(2 ** (n - 1) - 1)
            rand_x_bits = int2binlist(rand_x, bit_len=n)
            rand_m = random.randrange(1, n, 1)
            rand_m_bits = int2binlist(rand_m, bit_len=n)
            expect_num = int(rand_x % rand_m)
            expect_bits = int2binlist(expect_num, bit_len=n)

            # print("x: ", rand_x, " m: ", rand_m)

            circuit.fill_node_values(X, rand_x_bits)
            circuit.fill_node_values(M, rand_m_bits)
            circuit.fill_node_values(X_CY, rand_x_bits)
            circuit.fill_node_values(M_CY, rand_m_bits)

            circuit.simulate()

            O_PORTS = circuit.get_output_nodes_ports(O)
            O_VALUE = circuit.compute_value_from_ports(O_PORTS)

            Y_CY_PORTS = circuit.get_output_nodes_ports(Y_CY)
            Y_CY_VALUE = circuit.compute_value_from_ports(Y_CY_PORTS)

            self.assertEqual(
                O_VALUE,
                expect_num,
                msg=(
                    f"X: {rand_x}",
                    f"M: {rand_m}",
                    f"O_VALUE: {O_VALUE}",
                    f"EXPECT: {expect_num}",
                    f"Y_CY_VALUE: {Y_CY_VALUE}",
                ),
            )

    def test_lemma_4_1_compute_summands(self):
        circuit = CircuitGraph()
        bit_len = 4
        num_amount = bit_len
        n = bit_len
        # O -> List[List[Port]]
        X, NUMS, O = setup_lemma_4_1_compute_summands(
            circuit, num_amount=num_amount, bit_len=bit_len
        )
        for i in range(20):
            rand_x = random.randrange(2 ** (n - 1) - 1)
            rand_x_bits = int2binlist(rand_x, bit_len=n)
            rand_nums = []
            rand_nums_bits = []
            for j in range(num_amount):
                rand_num = random.randrange(2 * (n - 1) - 1)
                rand_nums.append(rand_num)
                rand_num_bits = int2binlist(rand_num, bit_len=n)
                rand_nums_bits.append(rand_num_bits)
            circuit.fill_node_values(X, rand_x_bits)
            for idx, num in enumerate(NUMS):
                circuit.fill_node_values(num, rand_nums_bits[idx])
            circuit.simulate()
            for idx in range(num_amount):
                if rand_x_bits[idx] == 0:
                    e = 0
                else:
                    e = rand_nums[idx]
                value = circuit.compute_value_from_ports(
                    circuit.get_output_nodes_ports(O[idx])
                )
                self.assertEqual(
                    value,
                    e,
                    msg=(
                        f"idx: {idx}",
                        f"expect: {e}",
                        f"value: {value}",
                        f"rand_nums: {rand_nums}",
                        f"rand_x_bits: {rand_x_bits}",
                    ),
                )

    def test_lemma_4_1_provide_aims_given_m(self):
        circuit = CircuitGraph()
        bit_len = 4
        max_m = 4
        M, O = setup_lemma_4_1_provide_aims_given_m(circuit, bit_len=bit_len)
        for m in range(1, max_m + 1):
            m_bits = int2binlist(m, bit_len=bit_len)
            circuit.fill_node_values(M, m_bits)
            circuit.simulate()
            sim_values = []
            for o in O:
                value = circuit.compute_value_from_ports(
                    circuit.get_output_nodes_ports(o)
                )
                sim_values.append(value)
            sb_aims = sb.precompute_aim(max_m)
            sb_ais = sb_aims[m]

            for i in range(len(sim_values)):
                self.assertEqual(
                    sb_ais[i],
                    sim_values[i],
                    msg=(f"m: {m}", f"sim_values: {sim_values}", f"sb_ais: {sb_ais}"),
                )

    def test_adder_tree_iterative(self):
        circuit = CircuitGraph()
        bit_len = 4
        num_amount = 4
        n = bit_len
        SUMMANDS, O = setup_adder_tree_iterative(
            circuit, num_amount=num_amount, bit_len=bit_len
        )
        for i in range(20):
            nums = []
            nums_bits = []
            for j in range(num_amount):
                rand_x = random.randrange(int((2 ** (n - 1) - 1) / 4))
                rand_x_bits = int2binlist(rand_x, bit_len=bit_len)
                nums.append(rand_x)
                nums_bits.append(nums_bits)
                circuit.fill_node_values(SUMMANDS[j], rand_x_bits)
            circuit.simulate()
            expect_num = sum(nums)
            value = circuit.compute_value_from_ports(circuit.get_output_nodes_ports(O))
            self.assertEqual(expect_num, value)

    def test_max_tree_iterative(self):

        circuit = CircuitGraph()
        bit_len = 4
        num_amount = 4
        n = bit_len
        VALUES, O = setup_max_tree_iterative(
            circuit, num_amount=num_amount, bit_len=bit_len
        )
        for i in range(20):
            nums = []
            nums_bits = []
            for j in range(num_amount):
                rand_x = random.randrange(int((2 ** (n - 1) - 1)))
                rand_x_bits = int2binlist(rand_x, bit_len=bit_len)
                nums.append(rand_x)
                nums_bits.append(nums_bits)
                circuit.fill_node_values(VALUES[j], rand_x_bits)
            circuit.simulate()
            expect_num = max(nums)
            value = circuit.compute_value_from_ports(circuit.get_output_nodes_ports(O))
            self.assertEqual(expect_num, value)

    def test_lemma_4_1_compute_y(self):
        circuit = CircuitGraph()
        bit_len = 4
        num_amount = bit_len
        n = bit_len
        X, M, O = setup_lemma_4_1_compute_y(circuit, bit_len=bit_len)

        # print("compute_y_extra_test_cases: ")

        x_cases = [5, 5, 6]
        m_cases = [2, 3, 3]

        for x_value, m_value in zip(x_cases, m_cases):
            x_bits = int2binlist(x_value, bit_len=n)
            m_bits = int2binlist(m_value - 1, bit_len=n)
            circuit.fill_node_values(X, x_bits)
            circuit.fill_node_values(M, m_bits)
            circuit.simulate()
            y_value = sb.lemma_4_1_compute_y(x_value, m_value, n)
            value = circuit.compute_value_from_ports(circuit.get_output_nodes_ports(O))
            self.assertEqual(
                y_value,
                value,
                msg=(
                    f"expect: {y_value}",
                    f"value: {value}",
                    f"rand_x: {x_value}",
                    f"rand_m: {m_value}",
                ),
            )
            # print(f"expect: {y_value}, got: {value}, x: {x_value}, m: {m_value}")

        for i in range(20):
            rand_x = random.randrange(2 ** (n - 1) - 1)
            # rand_x = 3
            rand_x_bits = int2binlist(rand_x, bit_len=n)
            rand_m = random.randrange(1, n)
            # rand_m = 3
            rand_m_bits = int2binlist(rand_m - 1, bit_len=n)
            circuit.fill_node_values(X, rand_x_bits)
            circuit.fill_node_values(M, rand_m_bits)
            circuit.simulate()

            """y_value = 0
            for j in range(n):
                y_value += rand_x_bits[j] * aims[rand_m][j]
            """

            y_value = sb.lemma_4_1_compute_y(rand_x, rand_m, n)

            value = circuit.compute_value_from_ports(circuit.get_output_nodes_ports(O))
            self.assertEqual(
                y_value,
                value,
                msg=(
                    f"expect: {y_value}",
                    f"value: {value}",
                    f"rand_x: {rand_x}",
                    f"rand_m: {rand_m}",
                ),
            )

    def test_theorem_4_2_precompute_lookup_is_prime_power(self):
        circuit = CircuitGraph()
        for n in [4, 8]:
            sb_o = sb.theorem_4_2_precompute_lookup_is_prime_power(n)
            O = setup_theorem_4_2_precompute_lookup_is_prime_power(circuit, bit_len=n)
            circuit.simulate()
            o_ports = circuit.get_output_nodes_ports(O)
            for idx, port in enumerate(o_ports):
                self.assertEqual(circuit.get_port_value(port), sb_o[idx])

    def test_theorem_4_2_precompute_lookup_p_l(self):
        circuit = CircuitGraph()
        for n in [4, 8]:
            sb_o = sb.theorem_4_2_precompute_lookup_p_l(n)
            P_TABLE_NODES, L_TABLE_NODES = setup_theorem_4_2_precompute_lookup_p_l(
                circuit, bit_len=n
            )
            circuit.simulate()
            O_PORTS = []
            for p_nodes, l_nodes in zip(P_TABLE_NODES, L_TABLE_NODES):
                p_ports = circuit.get_output_nodes_ports(p_nodes)
                l_ports = circuit.get_output_nodes_ports(l_nodes)
                O_PORTS.append((p_ports, l_ports))
            for idx, (p_ports, l_ports) in enumerate(O_PORTS):
                p_value = circuit.compute_value_from_ports(p_ports)
                l_value = circuit.compute_value_from_ports(l_ports)
                p_expect = sb_o[idx][0]
                l_expect = sb_o[idx][1]
                self.assertEqual(p_value, p_expect)
                self.assertEqual(l_value, l_expect)

    def test_theorem_4_2_precompute_lookup_powers(self):
        circuit = CircuitGraph()
        for n in [4, 8]:
            sb_o = sb.theorem_4_2_precompute_lookup_powers(n)
            O = setup_theorem_4_2_precompute_lookup_powers(circuit, bit_len=n)
            circuit.simulate()
            O_PORTS = []
            for powers_of_p_nodes in O:
                powers_of_p_ports = []
                for p_nodes in powers_of_p_nodes:
                    p_ports = circuit.get_output_nodes_ports(p_nodes)
                    powers_of_p_ports.append(p_ports)
                O_PORTS.append(powers_of_p_ports)
            for i, powers_for_p in enumerate(O_PORTS):
                for j, power in enumerate(powers_for_p):
                    got = circuit.compute_value_from_ports(power)
                    expect = sb_o[i][j]
                    self.assertEqual(got, expect)

    def test_theorem_4_2_precompute_lookup_division(self):
        circuit = CircuitGraph()
        n = 8
        TABLE = setup_theorem_4_2_precompute_lookup_division(circuit, bit_len=n)
        circuit.simulate()
        TABLE_PORTS = []
        for row in TABLE:
            ports = [circuit.get_output_nodes_ports(entry) for entry in row]
            TABLE_PORTS.append(ports)
        for i, row in enumerate(TABLE_PORTS):
            for j, entry in enumerate(row):
                if i == 0 or j == 0:
                    continue
                got = circuit.compute_value_from_ports(entry)
                expect = (i) // (j)
                self.assertEqual(got, expect)

    def test_theorem_4_2_precompute_lookup_generator_powers(self):
        circuit = CircuitGraph()
        n = 8
        TABLE = setup_theorem_4_2_precompute_lookup_generator_powers(circuit, bit_len=n)
        circuit.simulate()
        TABLE_PORTS = []
        for row in TABLE:
            ports = [circuit.get_output_nodes_ports(entry) for entry in row]
            TABLE_PORTS.append(ports)

        software_generator_powers = sb.theorem_4_2_precompute_lookup_generator_powers(n)

        for i, row in enumerate(TABLE_PORTS):
            for j, entry in enumerate(row):
                if i == 0:
                    continue
                got = circuit.compute_value_from_ports(entry)
                expect = software_generator_powers[i][j]
                self.assertEqual(got, expect)

    def test_theorem_4_2_precompute_lookup_tables_B(self):
        circuit = CircuitGraph()
        n = 8
        TABLE_ZERO, TABLE_ONE = setup_theorem_4_2_precompute_lookup_tables_B(
            circuit, bit_len=n
        )

        circuit.simulate()

        TABLE_ZERO_PORTS = []
        for row in TABLE_ZERO:
            ports = [circuit.get_output_nodes_ports(entry) for entry in row]
            TABLE_ZERO_PORTS.append(ports)
        TABLE_ONE_PORTS = []
        for row in TABLE_ONE:
            ports = [circuit.get_output_nodes_ports(entry) for entry in row]
            TABLE_ONE_PORTS.append(ports)

        for i, row in enumerate(TABLE_ZERO_PORTS):
            for j, entry in enumerate(row):
                try:
                    expect = sanity.compute_a_b_l_formula(0, j, i)
                except:
                    expect = 0
                if expect > (2**n) - 1:
                    expect = 0
                got = circuit.compute_value_from_ports(entry)
                self.assertEqual(
                    got, expect, msg=(f"got: {got}, expect: {expect}, i: {i}, j: {j}")
                )

        for i, row in enumerate(TABLE_ONE_PORTS):
            for j, entry in enumerate(row):
                try:
                    expect = sanity.compute_a_b_l_formula(1, j, i)
                except:
                    expect = 0
                if expect > (2**n) - 1:
                    expect = 0
                got = circuit.compute_value_from_ports(entry)
                self.assertEqual(
                    got, expect, msg=(f"got: {got}, expect: {expect}, i: {i}, j: {j}")
                )

    def test_theorem_4_2_precompute_lookup_pexpl_minus_pexpl_minus_one(self):
        circuit = CircuitGraph()
        n = 8
        TABLE = setup_theorem_4_2_precompute_lookup_pexpl_minus_pexpl_minus_one(
            circuit, bit_len=n
        )
        circuit.simulate()
        TABLE_PORTS = []
        for nodes in TABLE:
            ports = circuit.get_output_nodes_ports(nodes)
            TABLE_PORTS.append(ports)

        p_l_lookup = sb.theorem_4_2_precompute_lookup_p_l(n)
        for idx, ports in enumerate(TABLE_PORTS):
            p, l = p_l_lookup[idx]
            got = circuit.compute_value_from_ports(ports)
            if l - 1 < 0:
                expect = 0
            else:
                expect = idx - (p ** (l - 1))
            self.assertEqual(got, expect)

    def test_theorem_4_2_step_1(self):
        circuit = CircuitGraph()
        n = 8
        X_LIST, P, PEXPL, EXPONENTS = setup_theorem_4_2_step_1(circuit, bit_len=n)

        for _ in range(1):
            # FILL
            x_list, pexpl, p, _, _ = utils.generate_test_values_for_theorem_4_2(n)

            # largest_power_lookup = sanity.theorem_4_2_precompute_lookup_step_1(n)
            # expected_exponents_list = []
            # for x in x_list:
            #    expected_exponents_list.append(largest_power_lookup[x][p])

            expected_exponents_list = (
                sanity.theorem_4_2_step_1_compute_largest_power_of_p(x_list, p)
            )

            for x, nodes in zip(x_list, X_LIST):
                x_bits = int2binlist(x, bit_len=n)
                circuit.fill_node_values(nodes, x_bits)

            pexpl_bits = int2binlist(pexpl, bit_len=n)
            p_bits = int2binlist(p, bit_len=n)
            # l_bits = int2binlist(l, bit_len=n)

            circuit.fill_node_values(P, p_bits)
            circuit.fill_node_values(PEXPL, pexpl_bits)

            circuit.simulate()

            EXPONENTS_PORTS = []
            for nodes in EXPONENTS:
                EXPONENTS_PORTS.append(circuit.get_output_nodes_ports(nodes))

            # for i in range(len(expected_exponents_list)):
            #    got_ports = EXPONENTS_PORTS[i]
            #    got_value = circuit.compute_value_from_ports(got_ports)
            #    print(f"got: {got_value}")
            #    print(f"expect: {expected_exponents_list[i]}")

            # print("STEP_1 TEST WITH")
            # print(f"x_list: {x_list}, pexpl: {pexpl}, p: {p}")

            for i, expect_expo in enumerate(expected_exponents_list):
                got_ports = EXPONENTS_PORTS[i]
                got_value = circuit.compute_value_from_ports(got_ports)
                self.assertEqual(
                    got_value,
                    expect_expo,
                    msg=(f"got: {got_value}", f"expect: {expect_expo}"),
                )

    def a_test_theorem_4_2_step_2(self):
        circuit = CircuitGraph()
        n = 8
        X_LIST_NODES, P_NODES, J_LIST_NODES, Y_LIST_NODES = setup_theorem_4_2_step_2(
            circuit, bit_len=n
        )
        for loop_idx in range(10):
            # print(f"LOOP INDEX: {loop_idx + 1}")
            # GENERATE TEST CASE VALUES
            x_list, pexpl, p, l, expectation = (
                utils.generate_test_values_for_theorem_4_2(n)
            )
            j_list = sanity.theorem_4_2_step_1_compute_largest_power_of_p(x_list, p)
            y_list = sanity.theorem_4_2_step_2_compute_x_dividend_by_p(
                x_list, j_list, p
            )
            # FILL VALUES
            for nodes, x in zip(X_LIST_NODES, x_list):
                circuit.fill_node_values(nodes, int2binlist(x, bit_len=n))
            circuit.fill_node_values(P_NODES, int2binlist(p, bit_len=n))
            for nodes, j in zip(J_LIST_NODES, j_list):
                circuit.fill_node_values(nodes, int2binlist(j, bit_len=n))

            circuit.simulate()

            Y_LIST_PORTS = [
                circuit.get_output_nodes_ports(nodes) for nodes in Y_LIST_NODES
            ]

            for idx, expect_y in enumerate(y_list):
                got_y = circuit.compute_value_from_ports(Y_LIST_PORTS[idx])
                self.assertEqual(got_y, expect_y)

    def test_theorem_4_2_compute_sum(self):
        circuit = CircuitGraph()
        n = 8
        J_LIST_NODES, J_NODES = setup_theorem_4_2_compute_sum(circuit, bit_len=n)

        for loop_idx in range(10):
            rand_nums = [random.randrange(0, 2 * n // n)]
            # FILL
            for nodes, num in zip(J_LIST_NODES, rand_nums):
                circuit.fill_node_values(nodes, int2binlist(num, bit_len=n))

            circuit.simulate()

            J_PORTS = circuit.get_output_nodes_ports(J_NODES)

            got_j = circuit.compute_value_from_ports(J_PORTS)
            expect_j = sum(rand_nums)
            self.assertEqual(got_j, expect_j)

    def test_theorem_4_2_step_4(self):
        circuit = CircuitGraph()
        n = 8
        P_NODES, PEXPL_NODES, FLAG_NODE = setup_theorem_4_2_step_4(circuit, bit_len=n)

        for loop_idx in range(100):
            p = random.randrange(0, 2 ** (n - 1) - 1)
            pexpl = random.randrange(0, 2 ** (n - 1) - 1)

            r1 = random.randrange(0, 6)
            r2 = random.randrange(0, 6)

            if r1 == 0:
                p = 2
            if r2 == 0:
                pexpl = 2
            elif r2 == 1:
                pexpl = 4

            # FILL

            circuit.fill_node_values(P_NODES, int2binlist(p, bit_len=n))
            circuit.fill_node_values(PEXPL_NODES, int2binlist(pexpl, bit_len=n))

            circuit.simulate()

            FLAG_PORT = circuit.get_output_node_port(FLAG_NODE)

            got_flag = circuit.compute_value_from_ports([FLAG_PORT])
            if p != 2 or pexpl in [2, 4]:
                except_flag = 1
            else:
                except_flag = 0

            self.assertEqual(got_flag, except_flag)

    def test_theorem_4_2_A_step_5(self):
        circuit = CircuitGraph()
        n = 8
        Y_LIST_NODES, PEXPL_NODES, A_LIST_NODES = setup_theorem_4_2_A_step_5(
            circuit, bit_len=n
        )

        sw_disc_log_lookup = sb.theorem_4_2_precompute_lookup_generator_powers(n)

        for loop_idx in range(3):

            while True:
                x_list, pexpl, p, l, expectation = (
                    utils.generate_test_values_for_theorem_4_2(n)
                )
                if p != 2 or pexpl in [2, 4]:
                    break
            # print(f"loop_idx: {loop_idx}")
            # print(f"x_list: {x_list}")
            # print(f"pexpl: {pexpl}")
            # print(f"p: {p}")
            # print(f"l: {l}")
            j_list = sanity.theorem_4_2_step_1_compute_largest_power_of_p(x_list, p)
            # print("x_list: ")
            # print(x_list)
            y_list = sanity.theorem_4_2_step_2_compute_x_dividend_by_p(
                x_list, j_list, p
            )
            # print("y_list: ")
            # print(y_list)
            a_list = sanity.theorem_4_2_A_step_5_find_discrete_logarithms(
                sw_disc_log_lookup, pexpl, y_list
            )

            # FILL
            for idx, nodes in enumerate(Y_LIST_NODES):
                circuit.fill_node_values(nodes, int2binlist(y_list[idx], bit_len=n))

            circuit.fill_node_values(PEXPL_NODES, int2binlist(pexpl, bit_len=n))

            circuit.simulate()

            A_LIST_PORTS = [
                circuit.get_output_nodes_ports(nodes) for nodes in A_LIST_NODES
            ]

            for idx, a_ports in enumerate(A_LIST_PORTS):
                got = circuit.compute_value_from_ports(a_ports)
                expect = a_list[idx]
                # print(f"got: {got}, expect: {expect}")
                self.assertEqual(got, expect)

    def test_theorem_4_2_A_step_7(self):
        circuit = CircuitGraph()
        n = 8
        A_NODES, PEXPL_NODES, A_HAT_NODES = setup_theorem_4_2_A_step_7(
            circuit, bit_len=n
        )

        sw_disc_log_lookup = sanity.theorem_4_2_precompute_lookup_generator_powers(n)

        for loop_idx in range(3):
            while True:
                x_list, pexpl, p, l, expectation = (
                    utils.generate_test_values_for_theorem_4_2(n)
                )
                if p != 2 or pexpl in [2, 4]:
                    break
            j_list = sanity.theorem_4_2_step_1_compute_largest_power_of_p(x_list, p)
            y_list = sanity.theorem_4_2_step_2_compute_x_dividend_by_p(
                x_list, j_list, p
            )
            a_list = sanity.theorem_4_2_A_step_5_find_discrete_logarithms(
                sw_disc_log_lookup, pexpl, y_list
            )
            a = sanity.theorem_4_2_A_step_6_compute_a_sum(a_list)
            a_hat = sanity.theorem_4_2_A_step_7_compute_a_mod_pexpl_minus_pexpldecr(
                a, p, l
            )

            # FILL
            circuit.fill_node_values(A_NODES, int2binlist(a, bit_len=n))
            circuit.fill_node_values(PEXPL_NODES, int2binlist(pexpl, bit_len=n))

            circuit.simulate()

            A_HAT_PORTS = circuit.get_output_nodes_ports(A_HAT_NODES)
            got = circuit.compute_value_from_ports(A_HAT_PORTS)
            expect = a_hat
            self.assertEqual(got, expect)

    def test_theorem_4_2_A_step_8(self):
        circuit = CircuitGraph()
        n = 8
        A_HAT_NODES, PEXPL_NODES, Y_PRODUCT_NODES = setup_theorem_4_2_A_step_8(
            circuit, bit_len=n
        )

        sw_disc_log_lookup = sanity.theorem_4_2_precompute_lookup_generator_powers(n)

        for loop_idx in range(5):
            while True:
                x_list, pexpl, p, l, expectation = (
                    utils.generate_test_values_for_theorem_4_2(n)
                )
                if p != 2 or pexpl in [2, 4]:
                    break
            j_list = sanity.theorem_4_2_step_1_compute_largest_power_of_p(x_list, p)
            y_list = sanity.theorem_4_2_step_2_compute_x_dividend_by_p(
                x_list, j_list, p
            )
            a_list = sanity.theorem_4_2_A_step_5_find_discrete_logarithms(
                sw_disc_log_lookup, pexpl, y_list
            )
            a = sanity.theorem_4_2_A_step_6_compute_a_sum(a_list)
            a_hat = sanity.theorem_4_2_A_step_7_compute_a_mod_pexpl_minus_pexpldecr(
                a, p, l
            )

            y_product = sanity.theorem_4_2_A_step_8_read_reverse_log(
                sw_disc_log_lookup, pexpl, a_hat
            )

            # FILL
            circuit.fill_node_values(A_HAT_NODES, int2binlist(a_hat, bit_len=n))
            circuit.fill_node_values(PEXPL_NODES, int2binlist(pexpl, bit_len=n))

            circuit.simulate()

            Y_PRODUCT_PORTS = circuit.get_output_nodes_ports(Y_PRODUCT_NODES)
            got = circuit.compute_value_from_ports(Y_PRODUCT_PORTS)
            expect = y_product
            self.assertEqual(got, expect)

    def test_theorem_4_2_B_step_5(self):
        circuit = CircuitGraph()
        n = 8

        Y_LIST_NODES, L_NODES, A_LIST_NODES, B_LIST_NODES = setup_theorem_4_2_B_step_5(
            circuit, bit_len=n
        )

        # sw_a_zero, sw_a_one = sanity.theorem_4_2_precompute_lookup_tables_B(n)

        for loop_idx in range(3):
            while True:
                x_list, pexpl, p, l, expectation = (
                    utils.generate_test_values_for_theorem_4_2(n)
                )
                if not (p != 2 or pexpl in [2, 4]):
                    break
            j_list = sanity.theorem_4_2_step_1_compute_largest_power_of_p(x_list, p)
            y_list = sanity.theorem_4_2_step_2_compute_x_dividend_by_p(
                x_list, j_list, p
            )

            a_b_list = sanity.theorem_4_2_B_step_5_find_values(l, y_list)

            # FILL

            for idx, nodes in enumerate(Y_LIST_NODES):
                circuit.fill_node_values(nodes, int2binlist(y_list[idx], bit_len=n))

            circuit.fill_node_values(L_NODES, int2binlist(l, bit_len=n))

            circuit.simulate()

            A_LIST_PORTS = [
                circuit.get_output_nodes_ports(nodes) for nodes in A_LIST_NODES
            ]

            B_LIST_PORTS = [
                circuit.get_output_nodes_ports(nodes) for nodes in B_LIST_NODES
            ]

            for idx, (a_ports, b_ports) in enumerate(zip(A_LIST_PORTS, B_LIST_PORTS)):
                got_a = circuit.compute_value_from_ports(a_ports)
                got_b = circuit.compute_value_from_ports(b_ports)
                expect_a = a_b_list[idx][0]
                expect_b = a_b_list[idx][1]
                self.assertEqual(
                    got_a, expect_a, msg=(f"got_a: {got_a}, expect_a: {expect_a}")
                )
                self.assertEqual(
                    got_b,
                    expect_b,
                    msg=(
                        f"got_b: {got_b}",
                        f"expect_b: {expect_b}",
                        f"l: {l}",
                        f"y_list: {y_list}",
                    ),
                )

    def test_theorem_4_2_B_step_7(self):
        circuit = CircuitGraph()
        n = 8

        A_NODES, B_NODES, L_NODES, A_HAT_NODES, B_HAT_NODES = (
            setup_theorem_4_2_B_step_7(circuit, bit_len=n)
        )

        for loop_idx in range(10):
            a = random.randrange(1, 2**n - 1)
            b = random.randrange(1, 2**n - 1)
            l = random.randrange(1, int(math.log2(n)))

            expect_a = a % 2
            expect_b = b % (2 ** (l - 2))

            # FILL

            circuit.fill_node_values(A_NODES, int2binlist(a, bit_len=n))
            circuit.fill_node_values(B_NODES, int2binlist(b, bit_len=n))
            circuit.fill_node_values(L_NODES, int2binlist(l, bit_len=n))

            circuit.simulate()

            A_HAT_PORTS = circuit.get_output_nodes_ports(A_HAT_NODES)
            B_HAT_PORTS = circuit.get_output_nodes_ports(B_HAT_NODES)

            got_a_hat = circuit.compute_value_from_ports(A_HAT_PORTS)
            got_b_hat = circuit.compute_value_from_ports(B_HAT_PORTS)

            self.assertEqual(got_a_hat, expect_a)
            self.assertEqual(got_b_hat, expect_b, msg=(f"b: {b}, l: {l}"))

    def test_theorem_4_2_step_9(self):
        circuit = CircuitGraph()
        n = 8

        P_NODES, J_NODES, PEXPL_NODES, Y_PRODUCT_NODES, RESULT_NODES = (
            setup_theorem_4_2_step_9(circuit, bit_len=n)
        )

        sw_disc_log_lookup = sanity.theorem_4_2_precompute_lookup_generator_powers(n)

        for loop_idx in range(10):
            while True:
                x_list, pexpl, p, l, expectation = (
                    utils.generate_test_values_for_theorem_4_2(n)
                )
                if p != 2 or pexpl in [2, 4]:
                    break
            j_list = sanity.theorem_4_2_step_1_compute_largest_power_of_p(x_list, p)
            y_list = sanity.theorem_4_2_step_2_compute_x_dividend_by_p(
                x_list, j_list, p
            )
            j = sanity.theorem_4_2_step_3_compute_j(j_list)
            a_list = sanity.theorem_4_2_A_step_5_find_discrete_logarithms(
                sw_disc_log_lookup, pexpl, y_list
            )
            a = sanity.theorem_4_2_A_step_6_compute_a_sum(a_list)
            a_hat = sanity.theorem_4_2_A_step_7_compute_a_mod_pexpl_minus_pexpldecr(
                a, p, l
            )

            y_product = sanity.theorem_4_2_A_step_8_read_reverse_log(
                sw_disc_log_lookup, pexpl, a_hat
            )

            result = sanity.theorem_4_2_step_9_compute_final_product(p, j, y_product, l)

            # FILL
            circuit.fill_node_values(P_NODES, int2binlist(p, bit_len=n))
            circuit.fill_node_values(J_NODES, int2binlist(j, bit_len=n))
            circuit.fill_node_values(PEXPL_NODES, int2binlist(pexpl, bit_len=n))
            circuit.fill_node_values(Y_PRODUCT_NODES, int2binlist(y_product, bit_len=n))

            circuit.simulate()

            RESULT_PORTS = circuit.get_output_nodes_ports(RESULT_NODES)
            got = circuit.compute_value_from_ports(RESULT_PORTS)
            expect = result
            self.assertEqual(got, expect)

    def test_theorem_4_2(self):
        circuit = CircuitGraph()
        n = 4
        X_LIST_NODES, PEXPL_NODES, RESULT_NODES = setup_theorem_4_2(circuit, bit_len=n)

        sw_disc_log_lookup = sanity.theorem_4_2_precompute_lookup_generator_powers(n)
        sw_part_b_lookup = sanity.theorem_4_2_precompute_lookup_tables_B(n)

        for loop_idx in range(2):
            while True:
                x_list, pexpl, p, l, expectation = (
                    utils.generate_test_values_for_theorem_4_2(n)
                )
                if p != 2 or pexpl in [2, 4]:
                    break
            j_list = sanity.theorem_4_2_step_1_compute_largest_power_of_p(x_list, p)
            y_list = sanity.theorem_4_2_step_2_compute_x_dividend_by_p(
                x_list, j_list, p
            )
            j = sanity.theorem_4_2_step_3_compute_j(j_list)
            do_a = sanity.theorem_4_2_step_4_test_condition(p, l)
            a_list = sanity.theorem_4_2_A_step_5_find_discrete_logarithms(
                sw_disc_log_lookup, pexpl, y_list
            )
            a = sanity.theorem_4_2_A_step_6_compute_a_sum(a_list)
            a_hat = sanity.theorem_4_2_A_step_7_compute_a_mod_pexpl_minus_pexpldecr(
                a, p, l
            )

            y_product = sanity.theorem_4_2_A_step_8_read_reverse_log(
                sw_disc_log_lookup, pexpl, a_hat
            )

            expect = sanity.theorem_4_2_step_9_compute_final_product(p, j, y_product, l)

            # FILL
            for idx, nodes in enumerate(X_LIST_NODES):
                circuit.fill_node_values(nodes, int2binlist(x_list[idx], bit_len=n))
            circuit.fill_node_values(PEXPL_NODES, int2binlist(pexpl, bit_len=n))

            circuit.simulate()

            RESULT_PORTS = circuit.get_output_nodes_ports(RESULT_NODES)
            got = circuit.compute_value_from_ports(RESULT_PORTS)
            self.assertEqual(got, expect)


class TestUtilsFunctions(unittest.TestCase):
    def test_int2binlist(self):
        expected = {
            (7, 5): [1, 1, 1, 0, 0],
            (64, 8): [0, 0, 0, 0, 0, 0, 1, 0],
            (129, 9): [1, 0, 0, 0, 0, 0, 0, 1, 0],
        }

        for num, bit_len in expected:
            self.assertEqual(
                int2binlist(num, bit_len=bit_len), expected[(num, bit_len)]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

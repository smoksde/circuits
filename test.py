import unittest
from circuits import *
from graph import *
from utils import int2binlist, iter_random_bin_list
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

        print("sb_output: ", sb_output)
        for m in range(len(output)):
            for i in range(n):
                value = circuit.compute_value_from_ports(
                    circuit.get_output_nodes_ports(output[m][i])
                )
                print(value, end="")
            print()

        for m in range(0, n):
            for i in range(n):
                e_bin_list = int2binlist(sb_output[m, i])
                for idx, e in enumerate(e_bin_list):
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
            print(f"y: {y_value}, m: {m_value}, expect: {expect}, got: {O_VALUE}")
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
            print("diffs:")
            print(diffs)

            O_VALUES = []

            for o in O:
                O_PORTS = circuit.get_output_nodes_ports(o)
                O_VALUE = circuit.compute_value_from_ports(O_PORTS)
                O_VALUES.append(O_VALUE)

            for idx, diff in enumerate(diffs):
                if diff < 0:
                    print("diff is smaller than 0, break")
                    break
                self.assertEqual(
                    O_VALUES[idx],
                    diff,
                    msg=f"idx: {idx}, expect: {diff}, got: {O_VALUES[idx]}",
                )
                print(f"idx: {idx}, expect: {diff}, got: {O_VALUES[idx]}")

    def test_lemma_4_1(self):
        circuit = CircuitGraph()
        bit_len = 4
        n = bit_len
        X, M, M_DECR, O = setup_lemma_4_1(circuit, bit_len=bit_len)
        X_CY, M_CY, Y_CY = setup_lemma_4_1_compute_y(circuit, bit_len=bit_len)

        x_cases = [6]
        m_cases = [3]
        for x_value, m_value in zip(x_cases, m_cases):
            x_bits = int2binlist(x_value, bit_len=n)
            m_bits = int2binlist(m_value, bit_len=n)
            m_decr_bits = int2binlist(m_value - 1, bit_len=n)

            expect_num = int(x_value % m_value)

            circuit.fill_node_values(X, x_bits)
            circuit.fill_node_values(M, m_bits)
            circuit.fill_node_values(M_DECR, m_decr_bits)
            circuit.fill_node_values(X_CY, x_bits)
            circuit.fill_node_values(M_CY, m_decr_bits)

            circuit.simulate()

            O_PORTS = circuit.get_output_nodes_ports(O)
            Y_CY_PORTS = circuit.get_output_nodes_ports(Y_CY)

            O_VALUE = circuit.compute_value_from_ports(O_PORTS)
            Y_CY_VALUE = circuit.compute_value_from_ports(Y_CY_PORTS)
            expect_value = int(sb.lemma_4_1_compute_y(x_value, m_value, n))
            print(
                f"x: {x_value}, m: {m_value}, n: {n}, expect: {expect_value}, got: {Y_CY_VALUE}"
            )

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
            rand_m_decr_bits = int2binlist(rand_m - 1, bit_len=n)
            expect_num = int(rand_x % rand_m)
            expect_bits = int2binlist(expect_num, bit_len=n)

            print("x: ", rand_x, " m: ", rand_m)

            circuit.fill_node_values(X, rand_x_bits)
            circuit.fill_node_values(M, rand_m_bits)
            circuit.fill_node_values(M_DECR, rand_m_decr_bits)
            circuit.fill_node_values(X_CY, rand_x_bits)
            circuit.fill_node_values(M_CY, rand_m_decr_bits)

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
                    f"M_DECR: {rand_m - 1}",
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
            m_bits = int2binlist(m - 1, bit_len=bit_len)
            circuit.fill_node_values(M, m_bits)
            circuit.simulate()
            sim_values = []
            for o in O:
                value = circuit.compute_value_from_ports(
                    circuit.get_output_nodes_ports(o)
                )
                sim_values.append(value)
            sb_aims = sb.precompute_aim(max_m)
            sb_ais = sb_aims[m - 1]

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

    def test_lemma_4_1_compute_y(self):
        circuit = CircuitGraph()
        bit_len = 4
        num_amount = bit_len
        n = bit_len
        X, M, O = setup_lemma_4_1_compute_y(circuit, bit_len=bit_len)

        print("compute_y_extra_test_cases: ")

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
            print(f"expect: {y_value}, got: {value}, x: {x_value}, m: {m_value}")

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

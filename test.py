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
            for idx, a in enumerate(A):
                circuit.node_values[str(a.node_id)] = a_bin_list[idx]
            for idx, b in enumerate(B):
                circuit.node_values[str(b.node_id)] = b_bin_list[idx]
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
        bit_len = 8
        X, A, OUT_NODES = setup_modulo_circuit(circuit, bit_len=bit_len)
        for i in range(10):
            rand_x = random.randrange(2**bit_len - 1)
            rand_a = random.randrange(2 ** (bit_len // 2) - 1)
            if rand_a == 0:
                rand_a = 2
            if rand_a < 16:
                rand_a = 16
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

    def test_modular_exponentiation(self):
        circuit = CircuitGraph()
        bit_len = 8
        B, E, M, OUT_NODES = setup_modular_exponentiation(circuit, bit_len=bit_len)
        for i in range(1):
            rand_b = random.randrange(2**bit_len - 1)
            rand_e = random.randrange(2**bit_len - 1)
            rand_m = random.randrange(2 ** (bit_len // 2) - 1)
            if rand_m == 0:
                rand_m = 2
            if rand_m < 16:
                rand_m = 16
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

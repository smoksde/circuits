import unittest
from circuit import *
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
        "details": stream.getvalue()
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
            self.assertEqual(circuit.get_port_value(sum_out.ports[0]), expected[(val_a, val_b)][0])
            self.assertEqual(circuit.get_port_value(carry_out.ports[0]), expected[(val_a, val_b)][1])

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
            self.assertEqual(circuit.get_port_value(sum_out.ports[0]), expected[(val_a, val_b, val_cin)][0])
            self.assertEqual(circuit.get_port_value(carry_out.ports[0]), expected[(val_a, val_b, val_cin)][1])

    def test_carry_look_ahead_adder(self):
        circuit = CircuitGraph()
        A, B, cin, sums, carry = setup_carry_look_ahead_adder(circuit)

        bit_len = 4

        for i in range(100):
            rand_a = random.randrange(2**bit_len - 1)
            rand_b = random.randrange(2**bit_len - 1)
            rand_cin = random.randrange(2)

            expected_num = rand_a + rand_b + rand_cin

            expected_bin_list = int2binlist(expected_num, bit_len=bit_len+1)

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
            self.assertEqual(circuit.get_port_value(carry.ports[0]), expected_bin_list[-1])


    def test_wallace_tree_multiplier(self):
        circuit = CircuitGraph()
        A, B, outputs = setup_wallace_tree_multiplier(circuit)

        bit_len = 4

        for i in range(10):
            rand_a = random.randrange(2**bit_len - 1)
            rand_b = random.randrange(2**bit_len - 1)

            expected_num = rand_a * rand_b

            expected_bin_list = int2binlist(expected_num, bit_len=2*bit_len)

            a_bin_list = int2binlist(rand_a, bit_len=bit_len)
            b_bin_list = int2binlist(rand_b, bit_len=bit_len)

            for idx, a in enumerate(A):
                circuit.node_values[str(a.node_id)] = a_bin_list[idx]
            for idx, b in enumerate(B):
                circuit.node_values[str(b.node_id)] = b_bin_list[idx]

            circuit.simulate()
            
            for idx, e in enumerate(expected_bin_list):
                self.assertEqual(circuit.get_port_value(outputs[idx].ports[0]), e)

    # precompute constants: a_im = 2^i mod m values
    #def test_precompute_a_i(self):
    #    circuit = CircuitGraph()
    #    input_node = circuit.add_node("input", "INPUT")
    #    port = input_node.ports[0]
    #    zero_node = constant_zero(circuit, port)
    #    one_node = constant_one(circuit, port)
    #    a_i_lists = precompute_a_i(zero_node, one_node, 2, 3) # m, n
    #    expected = [1, 0]
    #    self.assertEqual(a_i_lists, expected)

class TestUtilsFunctions(unittest.TestCase):
    def test_int2binlist(self):
        expected = {
            (7, 5): [1, 1, 1, 0, 0],
            (64, 8): [0, 0, 0, 0, 0, 0, 1, 0],
            (129, 9): [1, 0, 0, 0, 0, 0, 0, 1, 0]
        }

        for num, bit_len in expected:
            self.assertEqual(int2binlist(num, bit_len=bit_len), expected[(num, bit_len)])

if __name__ == "__main__":
    unittest.main()

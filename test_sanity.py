import unittest
import json
from io import StringIO
import random

import lemma_5_1_sanity
import theorem_5_2_sanity
import theorem_5_3_sanity


# The circuit construction related tests are NOT in this file.

# This file contains all tests for the sanity implementations.
# These implementations are just the transcribed formulas from the
# Paper of Beame et al. to check if the formulas were understood
# correctly.


def run_tests():
    loader = unittest.TestLoader()
    classes = [TestTheorem_5_3_Sanity, TestLemma_5_1_Sanity, TestTheorem_5_2_Sanity]
    suites = []
    for cl in classes:
        suite = loader.loadTestsFromTestCase(cl)
        suites.append(suite)

    final_suite = unittest.TestSuite(suites)

    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(final_suite)

    output = {
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "wasSuccessful": result.wasSuccessful(),
        "details": stream.getvalue(),
    }

    return json.dumps(output)


class TestLemma_5_1_Sanity(unittest.TestCase):
    def test_step_2(self):
        tests = [
            (1, ([2], 2), [1]),
            (2, ([2, 3], 6), [3, 2]),
            (3, ([2, 3, 5], 30), [15, 10, 6]),
            (4, ([2, 3, 5, 7], 210), [105, 70, 42, 30]),
        ]
        for n, (primes, primes_product), v_list in tests:
            got_v_list = lemma_5_1_sanity.step_2(primes, primes_product)
            self.assertEqual(got_v_list, v_list)

    def test_step_3(self):
        tests = [([5, 1, 5, 2], [3, 5, 6, 7], [2, 1, 5, 4])]
        for v_list, c_list, w_list in tests:
            got_w_list = lemma_5_1_sanity.step_3(v_list, c_list)
            self.assertEqual(got_w_list, w_list)

    def test_step_4(self):
        tests = [([4, 7, 3, 6], [8, 4, 3, 1], [32, 28, 9, 6])]
        for v_list, w_list, u_list in tests:
            got_u_list = lemma_5_1_sanity.step_4(v_list, w_list)
            self.assertEqual(got_u_list, u_list)

    def test_step_5(self):
        tests = [(5, [2, 3, 7, 6], [1, 4, 2, 3], 34)]

        for x, c_list, u_list, y in tests:
            x_mod_c_i_list = [x % c_i for c_i in c_list]
            got_y = lemma_5_1_sanity.step_5(u_list, x_mod_c_i_list)
            self.assertEqual(got_y, y)

    # A few steps of Lemma 5.1 were skipped here since the central test function below already worked

    # This is the central test function for Lemma 5.1
    def test_lemma_5_1(self):
        n_list = [4, 8]
        for n in n_list:
            for _ in range(10):
                c_list, c = theorem_5_3_sanity.compute_good_modulus_sequence(n)
                x = random.randrange(1, 2 * n)
                x_mod_c_i_list = []
                for c_i in c_list:
                    x_mod_c_i_list.append(x % c_i)
                expect = x % c
                got = lemma_5_1_sanity.lemma_5_1(c_list, c, x_mod_c_i_list)
                self.assertEqual(got, expect)


class TestTheorem_5_2_Sanity(unittest.TestCase):
    def test_theorem_5_2(self):
        for _ in range(10):
            n = 4
            x_list = []
            for _ in range(4):
                x_list.append(random.randrange(2**n - 1))
            expect = 1
            for x in x_list:
                expect *= x
            _, c = theorem_5_3_sanity.compute_good_modulus_sequence(n * n)
            expect = expect % c
            got = theorem_5_2_sanity.theorem_5_2(x_list)
            self.assertEqual(got, expect)


class TestTheorem_5_3_Sanity(unittest.TestCase):
    def test_compute_good_modulus_sequence(self):
        tests = [(1, ([2], 2)), (2, ([2, 3], 6)), (3, ([2, 3, 5], 30))]
        for n, (primes, primes_product) in tests:
            got_primes, got_primes_product = (
                theorem_5_3_sanity.compute_good_modulus_sequence(n)
            )
            self.assertEqual(got_primes, primes)
            self.assertEqual(got_primes_product, primes_product)


if __name__ == "__main__":
    unittest.main(verbosity=2)

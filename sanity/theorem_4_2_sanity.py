from sanity.software_beame import theorem_4_2_precompute_lookup_generator_powers
from utils import is_prime_power, wheel_factorize

import math
import random


def compute_a_b_l_formula(a, b, l):
    if a == 0:
        return int((5**b) % (2**l))
    elif a == 1:
        return int(((-1) * (5**b)) % (2**l))
    raise ValueError(
        f"In a,b,l Formula of Theorem 4.2 a has to be in [0,1] but got a: {a}"
    )


def precompute_lookup_tables_B(n: int):
    table_zero = []
    for l in range(0, int(math.log2(n)) + 1):
        row = []
        for b in range(0, n + 1):
            try:
                value = compute_a_b_l_formula(0, b, l)
            except:
                value = 0
            if value > 2**n - 1:
                value = 0
            row.append(value)
        table_zero.append(row)

    table_one = []
    for l in range(0, int(math.log2(n)) + 1):
        row = []
        for b in range(0, n + 1):
            try:
                value = compute_a_b_l_formula(1, b, l)
            except:
                value = 0
            if value > (2**n) - 1:
                value = 0
            row.append(value)
        table_one.append(row)

    return table_zero, table_one


def precompute_lookup_step_1(n: int):
    table = []
    for x in range(0, n + 1):
        row = []
        for p in range(0, n + 1):
            if x == 0 or p == 0:
                exp_list = [0]
            else:
                exp_list = step_1_compute_largest_power_of_p([x], p)
            row.append(exp_list[0])
        table.append(row)
    return table


def step_1_compute_largest_power_of_p(x_list, p):
    n = len(x_list)
    largest_exp_list = []
    for i in range(n):
        largest_exp = 0
        j = 0
        while True:
            power = p**j
            if power > x_list[i]:
                break
            elif x_list[i] % power == 0:
                largest_exp = j
            j += 1
        largest_exp_list.append(largest_exp)
    return largest_exp_list


def step_2_compute_x_dividend_by_p(x_list, j_list, p):
    y_list = []
    n = len(x_list)
    for i in range(n):
        y = x_list[i] // (p ** j_list[i])
        y_list.append(y)
    return y_list


def step_3_compute_j(j_list):
    return sum(j_list)


# Do A if return is 0, do B if return is 1
def step_4_test_condition(p, l):
    if p != 2 or p**l == 2 or p**l == 4:
        return 1
    else:
        return 0


def A_step_5_find_discrete_logarithms(disc_log_lookup, pexpl, y_list):
    powers_list = disc_log_lookup[pexpl]
    a_list = []
    for y in y_list:
        if y not in powers_list:
            print(f"pexpl: {pexpl}")
            print(f"y_list: {y_list}")
            print(f"powers_list: {powers_list}")
            print(f"y: {y}")
        a = powers_list.index(
            y
        )  # usual way, but not compatible with how the index in the circuit part is calculated
        # a = len(powers_list) - 1 - powers_list[::-1].index(y)
        a_list.append(a)
    return a_list


def A_step_6_compute_a_sum(a_list):
    return sum(a_list)


def A_step_7_compute_a_mod_pexpl_minus_pexpldecr(a, p, l):
    return a % (p**l - p ** (l - 1))


def A_step_8_read_reverse_log(disc_log_lookup, pexpl_idx, a_idx):
    return disc_log_lookup[pexpl_idx][a_idx]


def B_step_5_find_values(l, y_list):
    print("------------B step 5")
    print("int(2 ** (l - 2))")
    print(int(2 ** (l - 2)))
    limit_b = max(int(2 ** (l - 2)),1)
    zero_a_values = [compute_a_b_l_formula(0, b, l) for b in range(limit_b)]
    one_a_values = [compute_a_b_l_formula(1, b, l) for b in range(limit_b)]

    # print("B step 5")
    print(f"l: {l}")
    print("zero_a_values")
    print(zero_a_values)
    print("one_a_values")
    print(one_a_values)

    a_b_list = []
    for y in y_list:
        if y in zero_a_values:
            a = 0
            b = zero_a_values.index(y)
        elif y in one_a_values:
            a = 1
            b = one_a_values.index(y)
        else:
            raise ValueError(f"y: {y} not found in lookup")
        a_b_list.append((a, b))
    return a_b_list


def B_step_6_compute_sums(a_b_list):
    return sum([t[0] for t in a_b_list]), sum([t[1] for t in a_b_list])


def B_step_7_compute_mods(a, b, l):
    a_hat = a % 2
    b_hat = b % 2 ** (l - 2)
    return a_hat, b_hat


def B_step_8_read_off_product(a, b, l):
    y_product = compute_a_b_l_formula(a, b, l)
    return y_product


def step_9_compute_final_product(p, j, y, l):
    return (p**j * y) % p**l


def compute(x_list, p, l, debug=False):
    if debug:
        print("--- Start of Theorem 4.2 Computation ---")
        print("x list:", end=" ")
        print(x_list)
        print(f"p: {p}, l: {l}")
    largest_exp_list = step_1_compute_largest_power_of_p(x_list, p)
    if debug:
        print("Largest Exponent List")
        print(largest_exp_list)
    y_list = step_2_compute_x_dividend_by_p(x_list, largest_exp_list, p)
    if debug:
        print("Compute x divided by p")
        print("y_list: ")
        print(y_list)
    j = step_3_compute_j(largest_exp_list)
    do_b = step_4_test_condition(p, l)
    disc_log_lookup = theorem_4_2_precompute_lookup_generator_powers(len(x_list))
    pexpl = p**l
    if not do_b:
        print("Start of Part A")
        a_list = A_step_5_find_discrete_logarithms(disc_log_lookup, pexpl, y_list)
        print("a_list: ")
        print(a_list)
        a = A_step_6_compute_a_sum(a_list)
        print("a")
        print(a)
        a_hat = A_step_7_compute_a_mod_pexpl_minus_pexpldecr(a, p, l)
        print("a_hat")
        print(a_hat)
        a_hat_idx = a_hat  # - 1????
        y_product = A_step_8_read_reverse_log(disc_log_lookup, pexpl, a_hat_idx)
    else:
        print("Start of Part B")
        a_b_list = B_step_5_find_values(l, y_list)
        a, b = B_step_6_compute_sums(a_b_list)
        a_hat, b_hat = B_step_7_compute_mods(a, b, l)
        y_product = B_step_8_read_off_product(a_hat, b_hat, l)

    x_product = step_9_compute_final_product(p, j, y_product, l)
    if debug:
        print("--- End of Theorem 4.2 Computation ---")
    return x_product


if __name__ == "__main__":

    """x_list = [1, 3, 4, 5, 8, 10, 2, 11, 6, 4, 2, 10, 1, 4, 8, 7]
    pexpl = 13
    p = 13
    l = 1
    expectation = 10
    result = theorem_4_2_compute(x_list, p, l, debug=True)
    print(result)

    exit()"""

    n = 64
    tests = 10

    while tests > 0:
        # print(f"TESTS REMAINING: {tests}")
        values = [random.randrange(1, n) for _ in range(n + 1)]
        pexpl = max(values) + 1
        values.remove(max(values))
        x_list = values

        product = 1
        for x in x_list:
            product *= x
        # print(f"product: {product}")

        factors = wheel_factorize(pexpl)
        l = len(factors)
        p = factors[0]

        expectation = product % p**l

        if expectation == 0:
            continue

        # Reject test case if any x_i is divisible by p
        if any(x % pexpl == 0 for x in x_list):
            continue

        if not is_prime_power(pexpl):
            continue

        tests -= 1

        print("x_list")
        for i in range(len(x_list)):
            print(f"x_{i}: ", x_list[i])
        print(f"p: {p}")
        print(f"l: {l}")
        result = compute(x_list, p, int(l), debug=True)
        print(f"result: {result}")

        print(f"expectation: {expectation}")

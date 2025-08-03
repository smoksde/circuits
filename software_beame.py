import math
import numpy as np
from typing import Iterable
from functools import reduce
import operator
from utils import int2binlist, is_prime_power, wheel_factorize

# Software level implementations of sections out of Log Depth Circuits for Division and Related Problems


def compute_first_n_primes(n):
    if n <= 0:
        return []

    # Estimate upper bound using n-th prime approximation: p_n = n log n
    if n < 6:
        upper_bound = 15
    else:
        upper_bound = int(n * (math.log(n) + math.log(math.log(n)))) + 10

    sieve = [True] * (upper_bound + 1)
    sieve[0:2] = [False, False]
    primes = []

    for num in range(2, upper_bound + 1):
        if sieve[num]:
            primes.append(num)
            if len(primes) == n:
                break
            for multiple in range(num * num, upper_bound + 1, num):
                sieve[multiple] = False
    return primes


def compute_product(l: Iterable[int]) -> int:
    return reduce(operator.mul, l, 1)


# Theorem 5.2 (3)
def compute_b_i_j(X: Iterable[int], C: Iterable[int], n: int, s: int):
    B = np.zeros((n, s))
    for i in range(n):
        for j in range(s):
            B[i, j] = X[i] % C[j]
    return


# Theorem 5.2 (4)
def compute_b_j(B: Iterable[int], C: Iterable[int], n: int, s: int):
    B_J = np.zeros((s))
    for j in range(s):
        b_j = 1
        for i in range(n):
            b_j *= B[i, j] % C[j]
        B_J[j] = b_j
    return B_J


# Theorem 5.2 (5)
def compute_iterated(X: Iterable[int], M_n2: int):
    X = [x % M_n2 for x in X]
    return compute_product(X)


def test_theorem_5_2():
    X = [5, 6, 7]
    n = 4
    s = n**2
    primes = compute_first_n_primes(n**2)
    M_n2 = compute_product(primes)
    b_i_j = compute_b_i_j(X, primes, n, s)
    b_j = compute_b_j(b_i_j, primes, n, s)
    # last step missing


# Lemma 4.1


# Works fine
def precompute_aim(n):
    aims = np.zeros((n, n), dtype=int)
    for m in range(1, n + 1):
        for i in range(n):
            aims[m - 1, i] = int((2**i) % m)
    return aims


def lemma_4_1_compute_diffs(y: int, m: int, n: int):
    diffs = []
    for i in range(n):
        diffs.append(y - (m * i))
    return diffs


def compute_y_lemma_4_1(X, A, m, n):
    sum = 0
    for i in range(n):
        part = X[i] * A[i, m]
        sum += part
    return sum


def lemma_4_1_compute_y(x: int, m: int, n: int):
    x_bits = int2binlist(x, bit_len=n)
    aims = precompute_aim(n)
    y = 0
    print("aims:")
    print(aims)
    print("x_bits")
    print(x_bits)
    for j in range(n):
        y += x_bits[j] * aims[m - 1][j]
        print(f"product y: {y}")
    return y


def get_binary_list_lsb_first(x):
    return [int(bit) for bit in bin(x)[2:]][::-1]


def compute_mod_lemma_4_1(x, m, n):
    x_list = get_binary_list_lsb_first(x)


# returns list for [1, n]
def theorem_4_2_precompute_lookup_is_prime_power(n: int):
    return [is_prime_power(i + 1) for i in range(n)]


def theorem_4_2_precompute_lookup_p_l(n: int):
    result = []
    for i in range(1, n + 1):
        if is_prime_power(i):
            factorization = wheel_factorize(i)
            p = factorization[0]
            l = len(factorization)
            result.append((p, l))
        else:
            result.append((0, 0))
    return result


def theorem_4_2_precompute_lookup_powers(n: int):
    result = []
    for p in range(1, n + 1):
        powers_of_p = []
        for e in range(n):
            if e > math.log2(n) or p**e > n:
                # power is larger than n, just fill with 0
                power = 0
            else:
                power = p**e
            powers_of_p.append(power)
        result.append(powers_of_p)
    return result


def theorem_4_2_precompute_lookup_generator_powers(n: int):
    result = []
    primitive_roots = find_primitive_roots(n)
    p_l_lookup = theorem_4_2_precompute_lookup_p_l(n)
    for pexpl_idx, pexpl in enumerate(range(1, n + 1)):
        p, l = p_l_lookup[pexpl_idx]
        if p == 0 or l == 0:
            tresh = 0
        else:
            tresh = int(math.pow(p, l)) - int(math.pow(p, l - 1))
        g = primitive_roots[pexpl_idx]
        pows_of_g = compute_powers_mod_up_to(g, pexpl, tresh)
        while len(pows_of_g) < n:
            pows_of_g.append(0)
        # here all pows_of_g lists are n entries long
        result.append(pows_of_g)
    return result


def is_primitive_root(g, n):
    if math.gcd(g, n) != 1:
        return False
    totient = len([i for i in range(1, n) if math.gcd(i, n) == 1])
    powers = set([pow(g, i, n) for i in range(1, totient + 1)])
    group = set([i for i in range(1, n) if math.gcd(i, n) == 1])
    return powers == group


# Finds the smallest primitve root for each p in [1, n] and returns the list
def find_primitive_roots(n):
    roots = []
    for p in range(1, n + 1):
        found = False
        value = 0
        for g in range(1, p):
            if is_primitive_root(g, p):
                found = True
                value = g
            if found:
                break
        roots.append(value)
    return roots


def compute_powers_up_to(g, thresh):
    powers = []
    if g == 1 or g == 0:
        return [0]
    i = 0
    while True:
        power = int(math.pow(g, i))
        i += 1
        # print(f"g: {g}, i: {i}, power: {power}, thresh: {thresh}")
        if power <= thresh:
            powers.append(power)
        else:
            break
    return powers


def compute_powers_mod_up_to(g, m, thresh):
    powers = []
    if g == 1 or g == 0:
        return [0]
    i = 0
    while True:
        if i > m:  # check this statement
            break
        power = int(pow(g, i, m))
        i += 1
        if power <= thresh:
            powers.append(power)
        else:
            break
    return powers


# def theorem_4_2_precompute_lookup_powers_of_g()

if __name__ == "__main__":
    # test_theorem_5_2()
    n = 16
    roots = find_primitive_roots(n)
    print(roots)
    powers = compute_powers_up_to(5, 500)
    print(powers)
    theorem_4_2_precompute_lookup_generator_powers(n)

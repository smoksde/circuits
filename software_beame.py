import math
import numpy as np
from typing import Iterable
from functools import reduce
import operator
from utils import int2binlist

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


if __name__ == "__main__":
    # test_theorem_5_2()
    print("hello")

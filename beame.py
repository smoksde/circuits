import math
import numpy as np
from typing import Iterable
from functools import reduce
import operator


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


if __name__ == "__main__":
    primes = compute_first_n_primes(16)
    print(primes)
    product = compute_product(primes)
    print(product)
    print(product.bit_length())

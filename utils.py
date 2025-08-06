import random
import math


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


# Write function that converts integer to binary list
def int2binlist(x, bit_len=None):
    if x < 0:
        raise ValueError("Only non-negative integers are supported.")
    res = [int(b) for b in bin(x)[2:]]
    res.reverse()
    if bit_len is not None:
        if bit_len < len(res):
            raise ValueError("bit_len is too small to represent the number.")
        res = res + [0] * (bit_len - len(res))
    return res


def binlist2int(bin_list):
    value = 0
    for i, bit in enumerate(bin_list):
        value += (2**i) * bit
    return int(value)


# Write function that converts binary list to integer


def iter_random_bin_list(list_len=4, amount=10):
    for i in range(amount):
        bin_list = [random.randint(0, 1) for _ in range(list_len)]
        yield bin_list


def wheel_factorize(n: int):
    factorization = []
    if n == 0:
        return [0]
    while n % 2 == 0:
        factorization.append(2)
        n //= 2
    d = 3
    while d * d <= n:
        while n % d == 0:
            factorization.append(d)
            n //= d
        d += 2
    if n > 1:
        factorization.append(n)
    return factorization


def is_prime_power(n: int):
    factors = wheel_factorize(n)
    # print("is_prime_power, factors", factors)
    if len(set(factors)) == 1 and (factors[0] in compute_first_n_primes(n)):
        return True
    return False


def generate_test_values_for_theorem_4_2(n: int):

    while True:

        values = [random.randrange(1, n) for _ in range(n + 1)]
        pexpl = max(values) + 1
        values.remove(pexpl - 1)
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

        break
    return x_list, pexpl, p, l, expectation


if __name__ == "__main__":
    print(compute_first_n_primes(8))

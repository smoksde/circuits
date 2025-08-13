from utils import compute_first_n_primes


def compute_good_modulus_sequence(n):
    primes = compute_first_n_primes(n)
    primes_product = 1
    for prime in primes:
        primes_product *= prime
    return primes, primes_product

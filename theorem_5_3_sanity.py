import utils


def compute_good_modulus_sequence(n):
    primes = utils.compute_first_n_primes(n)
    primes_product = 1
    for prime in primes:
        primes_product *= prime
    return primes, primes_product

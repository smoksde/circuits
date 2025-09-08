from typing import Optional, List, Tuple
from core.graph import *
import sanity.theorem_5_3_sanity as theorem_5_3_sanity
from .. import circuit_utils


# keeps bit width for primes but
# introduces n*n bit width for primes product!
def precompute_good_modulus_sequence(
    circuit: CircuitGraph,
    zero: Port,
    one: Port,
    n: int,
    parent_group: Optional[Group] = None,
) -> Tuple[List[List[Port]], List[Port]]:

    this_group = circuit.add_group("PRECOMPUTE_GOOD_MODULUS_SEQUENCE")
    if circuit.enable_groups and this_group is not None:
        this_group.set_parent(parent_group)

    value_primes, value_primes_product = (
        theorem_5_3_sanity.compute_good_modulus_sequence(n)
    )
    primes = []
    for p in value_primes:
        prime = circuit_utils.generate_number(p, n, zero, one, parent_group=this_group)
        primes.append(prime)
    primes_product = circuit_utils.generate_number(
        value_primes_product, n * n, zero, one, parent_group=this_group
    )
    return primes, primes_product

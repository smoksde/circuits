from circuits import circuit
import numpy as np
import math

import measurement

# returns a function that represents the theoretical approximation of the circuit node amount (context precompute circuits)
def function_approximation(circuit_name: str):

    circuit_name = circuit_name.removeprefix("setup_")

    name = circuit_name

    if name == "lemma_4_1_precompute_aim":
        return lambda n: math.pow(n, 3)
    if name == "theorem_4_2_precompute_lookup_is_prime_power":
        return lambda n: n
    if name == "theorem_4_2_precompute_lookup_p_l":
        return lambda n: math.pow(n, 2) * 2
    if name == "theorem_4_2_precompute_lookup_powers":
        return lambda n: math.pow(n, 3)
    if name == "theorem_4_2_precompute_lookup_generator_powers":
        return lambda n: math.pow(n, 3)
    
"""   
def components_approximation(circuit_name: str):

    circuit_name = circuit_name.removeprefix("setup_")

    if circuit_name == "lemma_4_1_precompute_aim":
        components = """ 
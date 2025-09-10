import tseitin

from core.graph import *
from circuits.circuit import *


def write_dimacs(clauses, num_vars, filename="circuit.cnf"):
    with open(filename, "w") as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def to_3sat(clauses, start_var):
    new_clauses = []
    next_var = start_var

    for clause in clauses:
        if len(clause) == 1:
            y, z = next_var, next_var + 1
            next_var += 2
            a = clause[0]
            new_clauses.extend(
                [
                    [a, y, z],
                    [a, y, -z],
                    [a, -y, z],
                    [a, -y, -z],
                ]
            )
        elif len(clause) == 2:
            y = next_var
            next_var += 1
            a, b = clause
            new_clauses.extend(
                [
                    [a, b, y],
                    [a, b, -y],
                ]
            )
        elif len(clause) == 3:
            new_clauses.append(clause)
        else:
            raise ValueError

    return new_clauses, next_var


# This script can be used to generate the DIMACS files for the constructed circuits.
# The general CNF form as well as a subsequent translation into 3-SAT CNF form is possible
# The 3-SAT variant is larger due to further added variables.

if __name__ == "__main__":
    circuit = CircuitGraph()
    interface = GraphInterface(circuit)
    bit_len = 4
    # setup_full_adder(interface, bit_len=8)
    setup_wallace_tree_multiplier(interface, bit_len=8)
    # setup_montgomery_ladder(interface, bit_len=bit_len)
    # setup_lemma_4_1(interface, bit_len=bit_len)
    # setup_theorem_4_2(interface, bit_len=bit_len)
    # setup_square_and_multiply(interface, bit_len=bit_len)

    clauses, output_vars, var_map = tseitin.tseitin_transform(interface)
    # clauses, next_var = to_3sat(clauses, start_var=max(var_map.values()) + 1)

    write_dimacs(
        clauses, num_vars=len(var_map), filename="circuit_dimacs_representation.cnf"
    )

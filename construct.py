from graph import *
from circuits.circuit import *

from tqdm import tqdm

if __name__ == "__main__":

    bit_lengths = [4, 8, 16, 32, 64, 128, 256]
    for n in tqdm(bit_lengths):
        tqdm.write(f"Constructing circuit for bit_length of {n} ...")
        circuit = CircuitGraph()
        setup_modular_exponentiation(circuit, bit_len=n)

def not_gate(x):
    return not x

def and_gate(x, y):
    return x and y

def xor_gate(x, y):
    return (x and not y) or (not x and y)

def half_adder(x, y):
    sum_out = xor_gate(x, y)
    carry_out = and_gate(x, y)
    return sum_out, carry_out

def full_adder(x, y, carry_in):
    sum_1, carry_out_1 = half_adder(x, y)
    sum_out, carry_out_2 = half_adder(sum_1, carry_in)
    carry_out = xor_gate(carry_out_1, carry_out_2)
    return sum_out, carry_out
    
def n_bit_adder(x, y, n, cin=False):
    """
    Implements an n-bit adder.
    
    Args:
        x: First binary number (list of booleans).
        y: Second binary number (list of booleans).
        n: Number of bits.
        cin: Carry-in for the least significant bit (default: False).

    Returns:
        A tuple containing the sum (as a list of booleans) and the carry-out. 
    """
    if n <= 0:
        raise ValueError("Number of bits (n) must be greater than 0.")
    if len(x) != n or len(y) != n:
        raise ValueError("Input boolean lists must have the same length.")

    sum_out = []
    carry_in = cin

    for i in range(n):
        bit_sum, carry_in = full_adder(x[i], y[i], carry_in)
        sum_out.append(bit_sum)
    return sum_out, carry_in

def int_to_bool_list(n: int, bits: int) -> list:
    """
    Converts an integer to a list of booleans that its binary representation.
    
    Args:
        n (int): The input integer.
        
    Returns:
        list: A list of booleans depicting the binary representation of the integer.
    """
    if n < 0:
        raise ValueError("Only non-negative integers are supported.")
    if n >= (1 << bits):
        raise ValueError(f"The integer {n} cannot be represented with {bits} bits.")
    bin_str = bin(n)[2:].zfill(bits)
    return [bit == '1' for bit in bin_str]

if __name__ == "__main__":
    a = 10
    b = 3


    n = 3
    sum_out, carry_out = n_bit_adder(a, b, n)

    print("Input A:", a)
    print("Input B:", b)
    print("Sum:", sum_out)
    print("Carry-out:", carry_out) 
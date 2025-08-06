def exp_by_squaring(base, exponent, modulus):
    """
    Exponentiation by squaring (also known as binary exponentiation)
    https://en.wikipedia.org/wiki/Modular_exponentiation
    """
    if modulus == 1:
        return 0
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result


def montgomery_ladder(base, exponent, modulus):
    r_0 = 1
    r_1 = base
    bits = exponent.bit_length()
    for i in reversed(range(bits)):
        bit = (exponent >> i) & 1
        if bit == 0:
            r_1 = r_0 * r_1 % modulus
            r_0 = r_0 * r_0 % modulus
        else:
            r_0 = r_0 * r_1 % modulus
            r_1 = r_1 * r_1 % modulus
    return r_0


if __name__ == "__main__":
    print(exp_by_squaring(4, 1, 10))
    print(montgomery_ladder(4, 1, 10))
    print(exp_by_squaring(6, 8, 7))
    print(montgomery_ladder(6, 8, 7))
    print(exp_by_squaring(16, 87, 23))
    print(montgomery_ladder(16, 87, 23))

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


if __name__ == "__main__":
    print(exp_by_squaring(4, 1, 10))

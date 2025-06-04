import random


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


# Write function that converts binary list to integer


def iter_random_bin_list(list_len=4, amount=10):
    for i in range(amount):
        bin_list = [random.randint(0, 1) for _ in range(list_len)]
        yield bin_list

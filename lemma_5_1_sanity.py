import random

import theorem_5_3_sanity


# def step_1(): This just gets the good modulus sequence


def step_2(c_list, c_product):
    v_list = []
    for c in c_list:
        v_i = c_product // c
        v_list.append(v_i)
    return v_list


def step_3(v_list, c_list):
    n = len(v_list)
    w_list = []
    for v_i, c_i in zip(v_list, c_list):
        for w_i in range(1, n * n):
            if (v_i * w_i) % c_i == 1:
                w_list.append(w_i)
                break
    return w_list


def step_4(v_list, w_list):
    u_list = []
    for v_i, w_i in zip(v_list, w_list):
        u_i = v_i * w_i
        u_list.append(u_i)
    return u_list


def step_5(u_list, x_mod_c_i_list):
    y = 0
    for u_i, x_mod_ci in zip(u_list, x_mod_c_i_list):
        summand = x_mod_ci * u_i
        y += summand
    return y


def step_6(y, n, c_n, c):
    y_t_list = []
    for t in range(0, n * c_n + 1):
        y_t = y - (t * c)
        y_t_list.append(y_t)
    return y_t_list


def step_7(y_t_list, c):
    for y_t in y_t_list:
        if y_t >= 0 and y_t < c:
            return y_t


def lemma_5_1(c_list, c, x_mod_c_i_list):
    n = len(c_list)
    v_list = step_2(c_list, c)
    w_list = step_3(v_list, c_list)
    u_list = step_4(v_list, w_list)
    y = step_5(u_list, x_mod_c_i_list)
    y_t_list = step_6(y, n, c_list[-1], c)
    result = step_7(y_t_list, c)
    return result


if __name__ == "__main__":
    n = 4
    x_list = []
    x_product = 1
    for _ in range(n):
        x_i = random.randrange(1, 2**n - 1)
        x_list.append(x_i)
        x_product *= x_i
    c_list, c = theorem_5_3_sanity.compute_good_modulus_sequence(n)
    x_mod_c_i_list = []
    for c_i in c_list:
        x_mod_c_i_list.append(x_product % c_i)
    v_list = step_2(c_list, c)
    w_list = step_3(v_list, c_list)
    u_list = step_4(v_list, w_list)
    print(f"u_list: {u_list}")
    y = step_5(u_list, x_mod_c_i_list)
    print(f"y: {y}")

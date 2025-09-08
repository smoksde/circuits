import sanity.lemma_5_1_sanity as lemma_5_1_sanity
import sanity.theorem_5_3_sanity as theorem_5_3_sanity

# Steps 1 - 2 dont need a sanity check since values are already given by other Lemma / Theorems
# Step 5 is essentially Lemma 5.1


# Builds matrix of b_i_j entries but with switched i, j such that its rather b_j_i
def step_3(x_list, c_list):
    n = len(x_list)
    s = len(c_list)

    matrix = []
    for j in range(1, s + 1):
        row = []
        for i in range(1, n + 1):
            value = x_list[i - 1] % c_list[j - 1]
            row.append(value)
        matrix.append(row)
    return matrix


def step_4(b_j_i_matrix, c_list):
    b_list = []
    for idx, row in enumerate(b_j_i_matrix):
        product = 1
        for b in row:
            product *= b
        b_j = product % c_list[idx]
        b_list.append(b_j)
    return b_list


def theorem_5_2(x_list):
    n = len(x_list)
    c_list, c = theorem_5_3_sanity.compute_good_modulus_sequence(n * n)
    b_j_i_matrix = step_3(x_list, c_list)
    b_list = step_4(b_j_i_matrix, c_list)
    result = lemma_5_1_sanity.lemma_5_1(c_list, c, b_list)
    return result

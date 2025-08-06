from graph import *
from typing import List, Any


def assert_list_of_ports(obj: Any):
    if not isinstance(obj, list) or not all(isinstance(p, Port) for p in obj):
        raise TypeError(
            f"Expected List[Port], but got: {type(obj)} with elements {[type(p) for p in obj]}"
        )


def assert_matrix_of_ports(obj: Any):
    if not isinstance(obj, list) or not all(
        isinstance(sublist, list) for sublist in obj
    ):
        raise TypeError(
            f"Expected List[List[Port]], but outer list has types: {[type(sub) for sub in obj]}"
        )
    for sublist in obj:
        if not all(isinstance(p, Port) for p in sublist):
            raise TypeError(
                f"Expected List[List[Port]], but inner list has types: {[type(p) for p in sublist]}"
            )


def assert_tensor_of_ports(obj: Any):
    if not isinstance(obj, list) or not all(isinstance(matrix, list) for matrix in obj):
        raise TypeError(
            f"Expected List[List[List[Port]]], but outer list has types: {[type(m) for m in obj]}"
        )
    for matrix in obj:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError(
                f"Expected List[List[List[Port]]], but 2nd-level list has types: {[type(r) for r in matrix]}"
            )
        for row in matrix:
            if not all(isinstance(p, Port) for p in row):
                raise TypeError(
                    f"Expected List[List[List[Port]]], but innermost list has types: {[type(p) for p in row]}"
                )

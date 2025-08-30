from circuits.circuit import *
from core.graph import *
from tqdm import tqdm

type_to_symbol_dict = {"and": "&", "or": "||", "xor": "^", "not": "!"}


def map_type_to_symbol(type: str):
    if type not in type_to_symbol_dict:
        print(f"No symbol for type {type}")
        return "ERROR"
    return type_to_symbol_dict[type]


def compute_formula_iterative_for_node(circuit: CircuitGraph, output_node: Node):
    topo_nodes = circuit.topological_sort()
    port_map = circuit.compute_target_to_source_port_map()

    port_by_id = {}
    for node in circuit.nodes.values():
        for port in node.ports:
            port_by_id[port.id] = port

    formulas = {}

    # debug
    for node in topo_nodes:
        if node.type in ["and", "or", "xor", "not"]:
            in_ports = circuit.get_input_ports_of_node(node)
            for in_port in in_ports:
                pre_port_id = port_map[in_port.id]
                pre_port = port_by_id[pre_port_id]
                pre_node = circuit.get_node_of_port(pre_port)
                assert (
                    pre_node in topo_nodes
                ), f"Predecessor {pre_node.node_id} of {node.node_id} is missing from topo_nodes!"
                assert topo_nodes.index(pre_node) < topo_nodes.index(
                    node
                ), f"Node {pre_node.node_id} appears after {node.node_id} in topological sort!"

    for node in tqdm(topo_nodes, desc="Compute nodes formulas: "):
        tqdm.write(f"Processing node {node.node_id} ({node.type})")

        if node.type == "input":
            formulas[node.node_id] = str(node.label)

        elif node.type == "not":
            in_port = circuit.get_input_ports_of_node(node)[0]
            pre_port = port_by_id[port_map[in_port.id]]
            pre_node = circuit.get_node_of_port(pre_port)
            op = map_type_to_symbol(node.type)
            formulas[node.node_id] = (op, formulas[pre_node.node_id])

        elif node.type in ["and", "or", "xor"]:
            in_ports = circuit.get_input_ports_of_node(node)
            pre_ports = [port_by_id[port_map[p.id]] for p in in_ports]
            pre_nodes = [circuit.get_node_of_port(p) for p in pre_ports]
            op = map_type_to_symbol(node.type)
            formulas[node.node_id] = (
                op,
                formulas[pre_nodes[0].node_id],
                formulas[pre_nodes[1].node_id],
            )

        elif node.type == "output":
            in_port = circuit.get_input_ports_of_node(node)[0]
            pre_port = port_by_id[port_map[in_port.id]]
            pre_node = circuit.get_node_of_port(pre_port)
            formulas[node.node_id] = formulas[pre_node.node_id]

        else:
            print(f"Unhandled node type: {node.type}")
            formulas[node.node_id] = "?"

    return formulas[output_node.node_id]


def formula_tree_to_string(tree):
    if isinstance(tree, str):
        return tree
    if isinstance(tree, tuple):
        tuple_len = len(tree)
        if tuple_len == 2:
            op, remain = tree
            return f"{op}({formula_tree_to_string(remain)})"
        elif tuple_len == 3:
            op, left, right = tree
            return (
                f"({formula_tree_to_string(left)}{op}{formula_tree_to_string(right)})"
            )
    return "?"


def formula_tree_to_string_iterative(tree):
    parts = []
    stack = [(tree, False)]

    while stack:
        node, visited = stack.pop()
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, tuple):
            tuple_len = len(node)
            if tuple_len == 2:
                op, remain = node
                if visited:
                    continue
                stack.append((remain, False))
                parts.append(op)
            elif tuple_len == 3:
                op, left, right = node
                if visited:
                    parts.append(")")
                else:
                    # Post order traversal
                    stack.append(((")", None), True))
                    stack.append((right, False))
                    parts.append(op)
                    stack.append((left, False))
                    parts.append("(")
        elif node == (")", None):
            parts.append(")")
    return "".join(parts)


def formula_tree_to_string_generator(tree):
    def gen(n):
        if isinstance(n, str):
            yield n
        elif isinstance(n, tuple):
            tuple_len = len(n)
            if tuple_len == 2:
                op, remain = n
                yield op
                yield "("
                yield from gen(remain)
                yield ")"
            elif tuple_len == 3:
                op, left, right = n
                yield "("
                yield from gen(left)
                yield op
                yield from gen(right)
                yield ")"

    print("inside generator joining strings now...")
    lst = list(gen(tree))
    print("Parts generated: ", len(lst))
    return "".join(lst)


def iter_formula_tree(tree):
    if isinstance(tree, str):
        yield tree
    elif isinstance(tree, tuple):
        if len(tree) == 2:
            op, sub = tree
            yield op
            yield "("
            yield from iter_formula_tree(sub)
            yield ")"
        elif len(tree) == 3:
            op, left, right = tree
            yield "("
            yield from iter_formula_tree(left)
            yield op
            yield from iter_formula_tree(right)
            yield ")"
    else:
        yield "?"


def recursive_compute_formula_for_node(
    circuit: CircuitGraph, node: Node, port_map=None, memo=None
):

    if memo is None:
        memo = {}

    if node.node_id in memo:
        return memo[node.node_id]

    if port_map is None:
        port_map = circuit.compute_target_to_source_port_map()

    if node.type == "output":
        input_ports = circuit.get_input_ports_of_node(node)
        pre_ports = [
            circuit.find_port_by_port_id(port_map[port.id]) for port in input_ports
        ]
        pre_node = circuit.get_node_of_port(pre_ports[0])
        result = recursive_compute_formula_for_node(
            circuit, pre_node, port_map=port_map, memo=memo
        )

    elif node.type == "input":
        result = str(node.label)

    elif node.type == "not":
        input_ports = circuit.get_input_ports_of_node(node)
        pre_ports = [
            circuit.find_port_by_port_id(port_map[port.id]) for port in input_ports
        ]
        pre_nodes = [circuit.get_node_of_port(pre_port) for pre_port in pre_ports]
        result = f"!{recursive_compute_formula_for_node(circuit, pre_nodes[0], port_map=port_map, memo=memo)}"

    elif node.type in ["xor", "and", "or"]:
        input_ports = circuit.get_input_ports_of_node(node)
        pre_ports = [
            circuit.find_port_by_port_id(port_map[port.id]) for port in input_ports
        ]
        pre_nodes = [circuit.get_node_of_port(pre_port) for pre_port in pre_ports]
        result = (
            recursive_compute_formula_for_node(
                circuit, pre_nodes[0], port_map=port_map, memo=memo
            )
            + f"{map_type_to_symbol(node.type)}"
            + recursive_compute_formula_for_node(
                circuit, pre_nodes[1], port_map=port_map, memo=memo
            )
        )

    else:
        print(f"Unhandled node type: {node.type}")
        result = "?"

    memo[node.node_id] = result
    return result


def write_formula_to_file(tree, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for token in iter_formula_tree(tree):
            f.write(token)


if __name__ == "__main__":
    bit_lengths = [4, 8]

    """
    print("carry look ahead")
    for n in bit_lengths:
        circuit = CircuitGraph()
        A, B, cin, sum_nodes, carry_node = setup_carry_look_ahead_adder(
            circuit, bit_len=n
        )
        formula = recursive_compute_formula_for_node(circuit, sum_nodes[0])
        print(formula)

    print("conditional subtract")
    for n in bit_lengths:
        circuit = CircuitGraph()
        print("setup circuit...")
        A, B, C, O = setup_conditional_subtract(circuit, bit_len=n)
        print("compute formula...")
        formula = compute_formula_iterative_for_node(circuit, O[0])
        print(formula)
    """

    """
    print("square and multiply")
    for n in bit_lengths:
        circuit = CircuitGraph()
        print("setup circuit...")
        B, E, M, OUT_NODES = setup_modular_exponentiation(circuit, bit_len=n)
        print("compute formula...")
        formula = compute_formula_iterative_for_node(circuit, OUT_NODES[0])
        print("print formula...")
        print(formula)"""

    for n in bit_lengths:
        circuit = CircuitGraph()
        # A, B, sum_node, carry_node = setup_half_adder(circuit, bit_len=n)
        B, E, M, OUT_NODES = setup_modular_exponentiation(circuit, bit_len=n)
        # A, B, cin, sum_nodes, carry_node = setup_carry_look_ahead_adder(
        #    circuit, bit_len=n
        # )
        print("compute formula tree")
        string_tree = compute_formula_iterative_for_node(circuit, OUT_NODES[0])

        # print("write formula to file...")
        # write_formula_to_file(string_tree, f"formula_{n}_bits.txt")

        print("file created")
        # exit()

        print("Incremental in-order printing: ")
        for i, part in enumerate(iter_formula_tree(string_tree)):
            print(part, end="")
        if i >= 500:
            print("\n... [truncated]")
            break

        # print("len string_tree")
        # print(len(string_tree))
        # print("concat formula string")

        # formula_str = formula_tree_to_string_generator(string_tree)
        # print("print formula string")
        # print(formula_str)

    """
    for n in bit_lengths:
        print("setup circuit...")
        X, M, M_DECR, O_NODES = setup_lemma_4_1(circuit, bit_len=n)
        print("compute formula...")
        formula = compute_formula_for_node(circuit, O_NODES[0])
        print(formula)
    """

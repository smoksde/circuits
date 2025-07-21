from circuits.circuit import *
from graph import *
from tqdm import tqdm

type_to_symbol_dict = {"and": "&", "or": "||", "xor": "^"}


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

    # Debug: Check that for all gate nodes, their inputs are already defined
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

            assert (
                pre_node.node_id in formulas
            ), f"Node {node.node_id} depends on {pre_node.node_id}, which has no formula yet!"

            formulas[node.node_id] = f"!{formulas[pre_node.node_id]}"

        elif node.type in ["and", "or", "xor"]:
            in_ports = circuit.get_input_ports_of_node(node)
            pre_ports = [port_by_id[port_map[p.id]] for p in in_ports]
            pre_nodes = [circuit.get_node_of_port(p) for p in pre_ports]

            for pre_node in pre_nodes:
                assert (
                    pre_node.node_id in formulas
                ), f"Node {node.node_id} depends on {pre_node.node_id}, which has no formula yet!"

            op = map_type_to_symbol(node.type)
            formulas[node.node_id] = (
                f"({formulas[pre_nodes[0].node_id]}{op}{formulas[pre_nodes[1].node_id]})"
            )

        elif node.type == "output":
            in_port = circuit.get_input_ports_of_node(node)[0]
            pre_port = port_by_id[port_map[in_port.id]]
            pre_node = circuit.get_node_of_port(pre_port)

            assert (
                pre_node.node_id in formulas
            ), f"Node {node.node_id} depends on {pre_node.node_id}, which has no formula yet!"

            formulas[node.node_id] = formulas[pre_node.node_id]

        else:
            print(f"Unhandled node type: {node.type}")
            formulas[node.node_id] = "?"

    return formulas[output_node.node_id]


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


if __name__ == "__main__":
    bit_lengths = [4, 8, 16, 32]

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
    print("square and multiply")
    for n in bit_lengths:
        circuit = CircuitGraph()
        print("setup circuit...")
        B, E, M, OUT_NODES = setup_modular_exponentiation(circuit, bit_len=n)
        print("compute formula...")
        formula = compute_formula_iterative_for_node(circuit, OUT_NODES[0])
        print("print formula...")
        print(formula)

    """
    for n in bit_lengths:
        print("setup circuit...")
        X, M, M_DECR, O_NODES = setup_lemma_4_1(circuit, bit_len=n)
        print("compute formula...")
        formula = compute_formula_for_node(circuit, O_NODES[0])
        print(formula)
    """

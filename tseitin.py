from core.graph import *
from circuits.circuit import *

from tqdm import tqdm


def tseitin_transform(circuit: CircuitGraph, speed_up=True):
    clauses = []
    var_map = {}

    if speed_up:
        port_map = circuit.compute_target_to_source_port_map()
        port_to_node_map = circuit.compute_port_to_node_mapping()

    # Map node ids to variables
    current_var = 1
    for node_id, node in circuit.nodes.items():
        if node.type == "input":
            var_map[node_id] = current_var
            current_var += 1

    for node_id, node in circuit.nodes.items():
        if node.type != "input":
            var_map[node_id] = current_var
            current_var += 1

    def v(nid):
        return var_map[nid]

    for node_id, node in tqdm(circuit.nodes.items()):
        if node.type == "input":
            continue

        cur_var = v(node_id)
        if node.type in ["and", "or", "xor"]:
            input_ports = circuit.get_input_ports_of_node(node)
            input_vars = []
            for port in input_ports:
                # this section can be replaced by helper function from graph
                if speed_up:
                    source_port_id = port_map[port.id]
                else:
                    source_port_id = None
                    for edge in circuit.edges:
                        if edge.target_port_id == port.id:
                            source_port_id = edge.source_port_id
                            break
                if source_port_id is None:
                    raise RuntimeError(f"Input port {port.id} not connected")
                # this should also be replaced
                source_node_id = None
                for n in circuit.nodes.values():
                    for p in n.ports:
                        if p.id == source_port_id:
                            source_node_id = str(n.node_id)
                            break
                    if source_node_id:
                        break
                if source_node_id is None:
                    raise RuntimeError(
                        f"Source port node not found for port {source_port_id}"
                    )
                input_vars.append(v(source_node_id))

            x, y = input_vars

            if node.type == "and":
                # z <-> x & y
                clauses.append([-cur_var, x])
                clauses.append([-cur_var, y])
                clauses.append([cur_var, -x, -y])
            elif node.type == "or":
                # z <-> x | y
                clauses.append([cur_var, -x])
                clauses.append([cur_var, -y])
                clauses.append([-cur_var, x, y])
            elif node.type == "xor":
                # z <-> x xor y
                clauses.append([-cur_var, -x, -y])
                clauses.append([-cur_var, x, y])
                clauses.append([cur_var, -x, y])
                clauses.append([cur_var, x, -y])

        elif node.type == "not":
            input_port = circuit.get_input_ports_of_node(node)[0]

            # should be replaced with function
            if speed_up:
                source_port_id = port_map[input_port.id]
            else:
                source_port_id = None
                for edge in circuit.edges:
                    if edge.target_port_id == input_port.id:
                        source_port_id = edge.source_port_id
                        break
            if source_port_id is None:
                raise RuntimeError(f"Input port {input_port.id} not connected")

            source_node_id = None
            for n in circuit.nodes.values():
                for p in n.ports:
                    if p.id == source_port_id:
                        source_node_id = str(n.node_id)
                        break
                if source_node_id:
                    break
            if source_node_id is None:
                raise RuntimeError(
                    f"Source port node not found for port {source_port_id}"
                )

            x = v(source_node_id)
            z = cur_var
            # z <-> ¬x
            clauses.append([-z, -x])
            clauses.append([z, x])

        elif node.type == "output":
            input_port = circuit.get_input_ports_of_node(node)[0]

            # should be replaced with function
            source_port_id = None
            for edge in circuit.edges:
                if edge.target_port_id == input_port.id:
                    source_port_id = edge.source_port_id
                    break
            if source_port_id is None:
                raise RuntimeError(f"Input port {input_port.id} not connected")

            source_node_id = None
            for n in circuit.nodes.values():
                for p in n.ports:
                    if p.id == source_port_id:
                        source_node_id = str(n.node_id)
                        break
                if source_node_id:
                    break
            if source_node_id is None:
                raise RuntimeError(
                    f"Source port node not found for port {source_port_id}"
                )

            z = v(node_id)
            x = v(source_node_id)
            clauses.append([-z, x])
            clauses.append([z, -x])

        else:
            raise RuntimeError(f"Unknown node type {node.type}")

    output_vars = [
        v(str(node.node_id)) for node in circuit.nodes.values() if node.type == "output"
    ]

    return clauses, output_vars, var_map


if __name__ == "__main__":
    circuit = CircuitGraph()
    bit_len = 8
    setup_full_adder(circuit, bit_len=8)
    # setup_modular_exponentiation(circuit, bit_len=bit_len)
    clauses, output_vars, var_map = tseitin_transform(circuit)

    print("CNF Klauseln:", clauses)
    print("Output Variablen:", output_vars)
    print("Variablen Zuordnung:", var_map)

from collections import Counter, defaultdict, deque
from graph import *
from circuits import *
import matplotlib.pyplot as plt


def compute_node_edges(cg):
    port_to_node_dict = {}
    for node in cg.nodes.values():
        node_id = node.node_id
        for port in node.ports:
            port_id = port.id
            port_to_node_dict[port_id] = node_id
    node_edges = []
    for edge in cg.edges:
        source_port = edge.source_port_id
        target_port = edge.target_port_id
        node_edges.append(
            (port_to_node_dict[source_port], port_to_node_dict[target_port])
        )
    node_edges = list(set(node_edges))
    return node_edges


def longest_path_length(cg):
    input_nodes = [node.node_id for node in cg.nodes.values() if node.type == "input"]
    output_nodes = [node.node_id for node in cg.nodes.values() if node.type == "output"]

    node_edges = compute_node_edges(cg)

    in_degree = defaultdict(int)
    for node in cg.nodes.values():
        in_degree[node] = len([port for port in node.ports if port.type == "input"])

    queue = deque(
        [node.node_id for node in cg.nodes.values() if in_degree[node.node_id] == 0]
    )
    topo_order = []

    while queue:
        node_id = queue.popleft()
        topo_order.append(node_id)
        for successor in [target for source, target in node_edges if source == node_id]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    dist = defaultdict(lambda: -float("inf"))
    for node_id in output_nodes:
        dist[node_id] = 0

    for node_id in reversed(topo_order):
        for successor in [target for source, target in node_edges if source == node_id]:
            if dist[successor] != -float("inf"):
                dist[node_id] = max(dist[node_id], dist[successor] + 1)

    return max(dist[node] for node in input_nodes if dist[node] != -float("inf"))


def analyze_circuit_function(name, setup_fn, bit_len=4):
    cg = CircuitGraph()
    setup_fn(cg, bit_len)
    num_nodes = len(cg.nodes)
    num_edges = len(cg.edges)
    depth = longest_path_length(cg)
    return {
        "name": name,
        "bit_len": bit_len,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "depth": depth,
    }


def analyze_all_functions():
    results = []
    for name, fn in CIRCUIT_FUNCTIONS.items():
        try:
            result = analyze_circuit_function(name, fn)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing '{name}': {e}")

    for r in results:
        print(f"\nCircuit: {r['name']}")
        print(f" Bit Length: {r['bit_len']}")
        print(f"  Nodes: {r['num_nodes']}")
        print(f"  Edges: {r['num_edges']}")
        print(f"Max depth: {r['depth']}")


def plot_metrics_for_adders():
    results = []
    functions = {"carry_look_ahead_adder": setup_carry_look_ahead_adder}
    # functions = {"ripple_carry_adder": setup_ripple_carry_adder}
    bit_lengths = [4, 8, 16, 32, 64]

    plt.figure(figsize=(8, 5))
    for key, value in functions.items():
        depths = []
        node_nums = []
        edge_nums = []
        for i in bit_lengths:
            # result = analyze_circuit_function("n_bit_comparator", setup_n_bit_comparator, i)
            result = analyze_circuit_function(key, value, i)
            depths.append(result["depth"])
            node_nums.append(result["num_nodes"])
            edge_nums.append(result["num_edges"])
            results.append(result)

        plt.plot(
            bit_lengths,
            depths,
            marker="o",
            label="Circuit Depth",
            linestyle="--",
            color="blue",
        )
        # plt.plot(bit_lengths, node_nums, marker='x', label='Node Count', linestyle='-.', color='purple')
        # plt.plot(bit_lengths, edge_nums, marker='x', label='Edge Count', linestyle='-.', color="green")

    plt.title("Circuit Characteristics")
    plt.xlabel("Bit Length (Number representation size)")
    plt.ylabel("Values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
    for r in results:
        print(f"\nCircuit: {r['name']}")
        print(f" Bit Length: {r['bit_len']}")
        print(f"  Nodes: {r['num_nodes']}")
        print(f"  Edges: {r['num_edges']}")
        print(f"Max depth: {r['depth']}")"""


if __name__ == "__main__":

    plot_metrics_for_adders()

    # analyze_all_functions()

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


def circuit_depth(circuit: CircuitGraph) -> int:
    port_map = circuit.compute_target_to_source_port_map()
    port_to_node = circuit.compute_port_to_node_mapping()
    depth = {}
    topo_order = circuit.topological_sort()
    for node in topo_order:
        in_ports = circuit.get_input_ports_of_node(node)
        source_ports_ids = [port_map[in_port.id] for in_port in in_ports]
        preds = [port_to_node[source_port] for source_port in source_ports_ids]
        if not preds or len(preds) < 1:
            depth[node.node_id] = 0
        else:
            depth[node.node_id] = 1 + max(depth[p] for p in preds)
    return max(depth.values(), default=0)


def longest_path_length(cg):
    input_nodes = [node.node_id for node in cg.nodes.values() if node.type == "input"]
    output_nodes = set(
        node.node_id for node in cg.nodes.values() if node.type == "output"
    )

    node_edges = compute_node_edges(cg)

    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for node in cg.nodes.values():
        in_degree[node.node_id] = 0

    for source, target in node_edges:
        graph[source].append(target)
        in_degree[target] += 1

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])

    dist = defaultdict(lambda: -float("inf"))

    for nid in input_nodes:
        dist[nid] = 0

    while queue:
        node = queue.popleft()
        for succ in graph[node]:
            dist[succ] = max(dist[succ], dist[node] + 1)
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    max_length = max(
        (dist[nid] for nid in output_nodes if dist[nid] != -float("inf")), default=0
    )
    return max_length


"""
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
"""


def analyze_circuit_function(name, setup_fn, bit_len=4):
    cg = CircuitGraph(enable_groups=False)
    setup_fn(cg, bit_len)
    num_nodes = len(cg.nodes)
    num_edges = len(cg.edges)
    # depth = 0
    # depth = circuit_depth(cg)
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


def plot_metrics_for_modulo_functions():
    results = []
    functions = {
        # "slow_modulo_circuit": setup_slow_modulo_circuit,
        "optimized_modulo_circuit": setup_optimized_modulo_circuit,
    }
    bit_lengths = [4, 8, 16, 32, 64]
    plt.figure(figsize=(8, 5))

    colors = ["blue", "red", "green"]

    for idx, (key, value) in enumerate(functions.items()):
        depths = []
        node_nums = []
        edge_nums = []
        for i in bit_lengths:
            result = analyze_circuit_function(key, value, i)
            depths.append(result["depth"])
            node_nums.append(result["num_nodes"])
            edge_nums.append(result["num_edges"])
            results.append(result)

        plt.plot(
            bit_lengths,
            depths,
            marker="o",
            label=f"Circuit Depth of {key}",
            linestyle="--",
            color=colors[idx],
        )

    plt.title("Circuit Characteristics")
    plt.xlabel("Bit Length (Number representation size)")
    plt.ylabel("Values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


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


def plot_metrics_for_lemma_4_1():
    results = []
    functions = {"lemma_4_1": setup_lemma_4_1}
    # functions = {"ripple_carry_adder": setup_ripple_carry_adder}
    bit_lengths = [4, 8, 16, 32]

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


def compare_sam_ml_depth():
    results = []
    functions = {
        "modular_exponentiation": setup_modular_exponentiation,
        "montgomery_ladder": setup_montgomery_ladder,
    }

    bit_lengths = [4, 8, 16]
    colors = ["blue", "red", "green"]

    plt.figure(figsize=(8, 5))
    for idx, (key, value) in enumerate(functions.items()):
        depths = []
        node_nums = []
        edge_nums = []
        for i in bit_lengths:
            print(key, i)
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
            label=f"Circuit Depth of {key}",
            linestyle="--",
            color=colors[idx],
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


def plot_circuit_metrics(experiments, metric="depth", title="Circuit Characteristics"):
    plt.figure(figsize=(8, 5))
    results = []

    for idx, exp in enumerate(experiments):
        name = exp["name"]
        setup_fn = exp["setup_fn"]
        bit_lengths = exp["bit_lengths"]
        color = exp.get("color", f"C{idx}")
        style = exp.get("style", "--")
        label = exp.get("label", f"{metric.capitalize()} of {name}")

        metric_values = []

        for bit_len in bit_lengths:
            print(f"analysing circuit: {name} with bit width: {bit_len} ...")
            result = analyze_circuit_function(name, setup_fn, bit_len)
            results.append(result)
            metric_values.append(result[metric])

        print(bit_lengths)
        print(metric_values)

        plt.plot(
            bit_lengths,
            metric_values,
            marker="o",
            label=label,
            linestyle=style,
            color=color,
        )

    plt.title(title)
    plt.xlabel("N - Bit Width of Input Numbers")
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_selected_plot():
    """experiments_to_run = [
        {
            "name": "carry_look_ahead_adder",
            "setup_fn": setup_carry_look_ahead_adder,
            "bit_lengths": [4, 8, 16, 32, 64, 128, 256],
            "color": "blue",
            "style": "--",
            "label": "Carry Lookahead Adder",
        },
    ]"""

    experiments = [
        {
            "name": "n_bit_comparator",
            "setup_fn": setup_n_bit_comparator,
            "bit_lengths": [4, 8, 16],
            "color": "blue",
            "style": "--",
            "label": "N-bit Comparator",
        }
    ]

    """{
            "name": "lemma_4_1_compute_y",
            "setup_fn": setup_lemma_4_1_compute_y,
            "bit_lengths": [4, 8, 16, 32],
            "color": "blue",
            "style": "--",
            "label": "lemma_4_1_compute_y",
        },"""

    experiments = [
        {
            "name": "lemma_4_1_reduce_in_parallel",
            "setup_fn": setup_lemma_4_1_reduce_in_parallel,
            "bit_lengths": [4, 8, 16, 32],
            "color": "blue",
            "style": "--",
            "label": "lemma_4_1_reduce_in_parallel",
        },
        {
            "name": "lemma_4_1_compute_diffs",
            "setup_fn": setup_lemma_4_1_compute_diffs,
            "bit_lengths": [4, 8, 16, 32],
            "color": "red",
            "style": "--",
            "label": "lemma_4_1_compute_diffs",
        },
    ]
    """
    experiments = [
        {
            "name": "wallace_tree_multiplier",
            "setup_fn": setup_wallace_tree_multiplier,
            "bit_lengths": [4, 8, 16, 32, 64, 128, 256, 512],
            "color": "blue",
            "style": "--",
            "label": "wallace_tree_multiplier",
        },
    ]"""

    experiments = [
        {
            "name": "lemma_4_1",
            "setup_fn": setup_lemma_4_1,
            "bit_lengths": [4, 8, 16, 32],
            "color": "blue",
            "style": "--",
            "label": "lemma_4_1",
        },
        {
            "name": "square_and_multiply",
            "setup_fn": setup_modular_exponentiation,
            "bit_lengths": [4, 8, 16, 32],
            "color": "red",
            "style": "--",
            "label": "square_and_multiply",
        },
        {
            "name": "montgomery_ladder",
            "setup_fn": setup_montgomery_ladder,
            "bit_lengths": [4, 8, 16, 32],
            "color": "green",
            "style": "--",
            "label": "montgomery_ladder",
        },
    ]

    experiments = [
        {
            "name": "lemma_4_1",
            "setup_fn": setup_lemma_4_1,
            "bit_lengths": [4, 8, 16, 32],
            "color": "blue",
            "style": "--",
            "label": "lemma_4_1",
        },
    ]

    experiments = [
        {
            "name": "Carry Look-Ahead Adder",
            "setup_fn": setup_carry_look_ahead_adder,
            "bit_lengths": [4, 8, 16, 32, 64, 128, 256, 512],
            "color": "blue",
            "style": "--",
            "label": "Carry Look-Ahead Adder",
        },
    ]

    experiments = [
        {
            "name": "Wallace Tree Multiplier",
            "setup_fn": setup_wallace_tree_multiplier,
            "bit_lengths": [4, 8, 16, 32, 64, 128, 256],
            "color": "blue",
            "style": "--",
            "label": "Wallace Tree Multiplier",
        },
        {
            "name": "Faulty Wallace Tree Multiplier",
            "setup_fn": setup_faulty_wallace_tree_multiplier,
            "bit_lengths": [4, 8, 16, 32, 64, 128, 256],
            "color": "red",
            "style": "--",
            "label": "Faulty Wallace Tree Multiplier",
        },
    ]

    """
    experiments = [
        {
            "name": "lemma_4_1",
            "setup_fn": setup_lemma_4_1,
            "bit_lengths": [2, 4, 8, 16],
            "color": "blue",
            "style": "--",
            "label": "lemma_4_1",
        },
        {
            "name": "theorem_4_2_step_1",
            "setup_fn": setup_theorem_4_2_step_1,
            "bit_lengths": [2, 4, 8],
            "color": "red",
            "style": "--",
            "label": "theorem_4_2_step_1",
        },
    ]

    experiments = [
        {
            "name": "theorem_4_2",
            "setup_fn": setup_theorem_4_2,
            "bit_lengths": [4, 8, 16],
            "color": "red",
            "style": "--",
            "label": "theorem_4_2",
        },
    ]"""

    metric = "depth"  # depth, num_nodes or num_edges

    plot_circuit_metrics(
        experiments,
        metric=metric,
        title=f"Circuits {metric}",
    )


if __name__ == "__main__":

    # plot_metrics_for_adders()
    # plot_metrics_for_modulo_functions()d"  # dep
    # analyze_all_functions()
    # compare_sam_ml_depth()
    # plot_metrics_for_lemma_4_1()
    run_selected_plot()

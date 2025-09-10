from collections import Counter, defaultdict, deque

from core.graph import *
from circuits import *
import matplotlib.pyplot as plt

import cache

from pathlib import Path

# This file contains functions and measurement tools that operate on the graph representation of the constructed circuits.

plot_types = ["general", "function_approximation", "component_approximation"]

metrics = [
    "depth",
    "num_gates",
    "num_nodes",
    "num_input_nodes",
    "num_output_nodes",
    "num_edges",
]

interfaces = [
    "graph",
    "depth",
]

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "measurements_cache.json"

"""
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
    return max(depth.values(), default=0)"""


def count_components(setup_fn, component_label, bit_len=4):
    cg = CircuitGraph(enable_groups=True)
    setup_fn(cg, bit_len)
    count = sum(1 for group in cg.groups.values() if group.label == component_label)
    return count


def analyze_circuit_function(
    name,
    setup_fn,
    bit_len=4,
    interface_name="graph",
    metric="depth",
    use_cache=True,
    fill_cache=True,
):
    if use_cache:
        metrics = cache.get(CACHE_FILE, name, bit_len)
        if metrics:
            if metric in metrics:
                return metrics

    if interface_name == "depth":
        interface = DepthInterface()
        measured_depth = setup_fn(interface, bit_len)
        if fill_cache:
            cache.update_field(CACHE_FILE, name, bit_len, "depth", measured_depth)
        return cache.get(CACHE_FILE, name, bit_len)
    else:

        cg = CircuitGraph(enable_groups=False)
        graph_interface = GraphInterface(cg)
        setup_fn(graph_interface, bit_len)
        num_nodes = len(cg.nodes)  # cg.nodes.values()
        num_gate_nodes = len(
            [node for node in cg.nodes.values() if node.type not in ["input", "output"]]
        )
        num_input_nodes = len(
            [node for node in cg.nodes.values() if node.type == "input"]
        )
        num_output_nodes = len(
            [node for node in cg.nodes.values() if node.type == "output"]
        )
        num_edges = len(cg.edges)  # cg.edges.values()
        depth = cg.longest_path_length()

        dic = {
            "name": name,
            "bit_len": bit_len,
            "num_nodes": num_nodes,
            "num_gates": num_gate_nodes,
            "num_input_nodes": num_input_nodes,
            "num_output_nodes": num_output_nodes,
            "num_edges": num_edges,
            "depth": depth,
        }

        if fill_cache:
            cache.update(CACHE_FILE, name, bit_len, dic)

        return dic


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
            result = analyze_circuit_function(
                name, setup_fn, bit_len, use_cache=True, fill_cache=True
            )
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

    metric = metrics[0]  # 0 is depth metric

    plot_circuit_metrics(
        experiments,
        metric=metric,
        title=f"Circuits {metric}",
    )


if __name__ == "__main__":

    name = "setup_montgomery_ladder"
    setup_fn = setup_montgomery_ladder
    component_label = "WALLACE_TREE_MULTIPLIER"
    bit_len = 8

    count = count_components(setup_fn, component_label, bit_len)
    print(f"Count of {component_label} components: {count}")

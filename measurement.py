from collections import Counter, defaultdict, deque

from graph import *
from circuits import *
import matplotlib.pyplot as plt

import json
import os
import time
import tempfile
from pathlib import Path

metrics = [
    "depth",
    "num_nodes",
    "num_gate_nodes",
    "num_input_nodes",
    "num_output_nodes",
    "num_edges",
]

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "measurements_cache.json"


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        bak = CACHE_FILE.with_suffix(".bad")
        os.replace(CACHE_FILE, bak)
        print(f"Corrupt cache backed up to {bak.name}")
        return {}


def save_cache(cache: dict):
    fd, tmp_path = tempfile.mkstemp(dir=BASE_DIR, prefix=".cache_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            json.dump(cache, tmpf, indent=2, sort_keys=True)
            tmpf.flush()
            os.fsync(tmpf.fileno())
        os.replace(tmp_path, CACHE_FILE)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def update_cached_metrics(circuit: str, bit_width: int, metrics: dict):
    cache = load_cache()
    cache.setdefault(circuit, {})[str(bit_width)] = metrics
    save_cache(cache)


def get_cached_metrics(circuit: str, bit_width: int):
    cache = load_cache()
    return cache.get(circuit, {}).get(str(bit_width))


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


def analyze_circuit_function(
    name, setup_fn, bit_len=4, use_cache=True, fill_cache=True
):
    if use_cache:
        metrics = get_cached_metrics(name, bit_len)
        if metrics:
            return metrics

    cg = CircuitGraph(enable_groups=False)
    setup_fn(cg, bit_len)
    num_nodes = len(cg.nodes)  # cg.nodes.values()
    num_gate_nodes = len(
        [node for node in cg.nodes.values() if node.type not in ["input", "output"]]
    )
    num_input_nodes = len([node for node in cg.nodes.values() if node.type == "input"])
    num_output_nodes = len(
        [node for node in cg.nodes.values() if node.type == "output"]
    )
    num_edges = len(cg.edges)  # cg.edges.values()
    # depth = circuit_depth(cg)
    depth = longest_path_length(cg)

    dic = {
        "name": name,
        "bit_len": bit_len,
        "num_nodes": num_nodes,
        "num_gate_nodes": num_gate_nodes,
        "num_input_nodes": num_input_nodes,
        "num_output_nodes": num_output_nodes,
        "num_edges": num_edges,
        "depth": depth,
    }

    if fill_cache:
        update_cached_metrics(name, bit_len, dic)

    return dic


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
    # analyze_all_functions()
    run_selected_plot()

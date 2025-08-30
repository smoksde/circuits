from circuits import *
from graph import *
import plotly.express as px
import pandas as pd
from collections import defaultdict


def build_area_treemap(groups, nodes):
    children_map = defaultdict(list)
    for group in groups.values():
        if group.parent:
            children_map[group.parent.id].append(group.id)

    def collect_descendants(group_id):
        descendants = [group_id]
        for child_id in children_map[group_id]:
            descendants.extend(collect_descendants(child_id))
        return descendants

    node_counts = defaultdict(int)
    for node in nodes.values():
        node_counts[node.group_id] += 1

    group_values = {}
    for group in groups.values():
        all_descendants = collect_descendants(group.id)
        group_values[group.id] = sum(node_counts[gid] for gid in all_descendants)

    ids = []
    labels = []
    parents = []
    values = []
    custom_data = []

    for group in groups.values():
        ids.append(str(group.id))
        parent_id = str(group.parent.id) if group.parent else ""
        parents.append(parent_id)
        labels.append(group.label)
        values.append(group_values[group.id])
        custom_data.append(f"Group ID: {group.id} — Nodes: {group_values[group.id]}")

    df = pd.DataFrame(
        {
            "id": ids,
            "label": labels,
            "parent": parents,
            "value": values,
            "custom": custom_data,
        }
    )

    fig = px.treemap(
        df,
        ids="id",
        parents="parent",
        names="label",
        values="value",
        custom_data=["custom"],
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>%{customdata[0]}")
    fig.show()


def build_treemap(groups, nodes):
    ids = []
    labels = []
    parents = []
    values = []
    custom_data = []

    for group in groups.values():
        ids.append(str(group.id))
        parent_id = str(group.parent.id) if group.parent else ""
        parents.append(parent_id)
        labels.append(group.label)
        values.append(1)
        custom_data.append(f"Group ID: {group.id}")

    """for node in nodes.values():
        group = next((g for g in groups if g.id == node.group_id), None)
        group_label = group.label if group else ""
        labels.append(node.label)
        parents.append(group_label)
        values.append(node.value if node.value else 1)
        custom_data.append(f"Node Type: {node.type}")"""

    df = pd.DataFrame(
        {
            "id": ids,
            "label": labels,
            "parent": parents,
            "value": values,
            "custom": custom_data,
        }
    )

    fig = px.treemap(
        df,
        ids="id",
        parents="parent",
        names="label",
        values="value",
        custom_data=["custom"],
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>%{customdata[0]}")
    fig.show()


if __name__ == "__main__":

    circuit = CircuitGraph()
    bit_len = 4  # 16
    # _ = setup_full_adder(circuit, bit_len=bit_len)
    # _ = setup_n_left_shift(circuit, bit_len=bit_len)
    # _ = setup_n_bit_comparator(circuit, bit_len=bit_len)
    # _ = setup_wallace_tree_multiplier(circuit, bit_len=bit_len)
    # _ = setup_modulo_circuit(circuit, bit_len=bit_len)
    # _ = setup_modular_exponentiation(circuit, bit_len=bit_len)
    _ = setup_lemma_4_1(circuit, bit_len=bit_len)
    # _ = setup_theorem_4_2_step_1(circuit, bit_len=bit_len)
    # _ = setup_theorem_4_2(circuit, bit_len=bit_len)
    # _ = setup_theorem_5_2(circuit, bit_len=bit_len)
    # _ = setup_lemma_4_1_reduce_in_parallel(circuit, bit_len=bit_len)
    #_ = setup_lemma_5_1_precompute_u_list(circuit, bit_len=bit_len)
    # circuit.simulate()
    # build_treemap(circuit.groups, circuit.nodes)
    build_area_treemap(circuit.groups, circuit.nodes)

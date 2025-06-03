from circuit import *
from graph import *
import plotly.express as px
import pandas as pd

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

    df = pd.DataFrame({
        "id": ids,
        "label": labels,
        "parent": parents,
        "value": values,
        "custom": custom_data,
    })

    fig = px.treemap(
        df,
        ids="id",
        parents="parent",
        names="label",
        values="value",
        custom_data=["custom"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>%{customdata[0]}"
    )
    fig.show()

if __name__ == "__main__":
    
    circuit = CircuitGraph()
    bit_len = 4
    _ = setup_modular_exponentiation(circuit, bit_len=bit_len)
    circuit.simulate()
    build_treemap(circuit.groups, circuit.nodes)
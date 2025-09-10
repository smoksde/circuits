from core.graph import CircuitGraph


class Interface:
    pass


class GraphInterface(Interface):

    enable_groups = True

    def __init__(self, circuit: CircuitGraph):
        self.circuit = circuit
        self.enable_groups = circuit.enable_groups

    def __getattr__(self, name):
        # Forward any functions to the underlying Circuit Graph
        return getattr(self.circuit, name)


class DepthInterface(Interface):

    enable_groups = False

    def add_node(self, node_type, label, inputs=None, group_id=None):
        if node_type in ["xor", "and", "or", "not"]:
            return max(inputs) + 1
        elif node_type == "input":
            return 0
        elif node_type == "output":
            return max(inputs)
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def add_group(self, label="DEFAULT_GROUP"):
        return None

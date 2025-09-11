from typing import List, Optional
from collections import defaultdict, deque

from core.node import Node
from core.port import Port
from core.edge import Edge
from core.group import Group

from utils import binlist2int

# Make x, y node coords and groups optional


class CircuitGraph:
    def __init__(self, enable_groups: bool = True):
        self.nodes = {}
        self.edges = []

        self.node_count = 0
        self.node_values = {}
        self.port_count = 0
        self.enable_groups = enable_groups
        if self.enable_groups:
            self.groups = {}
            self.group_count = 0

    def add_node(
        self,
        node_type: str,
        label: str,
        inputs: Optional[List[Port]] = [],
        group_id: Optional[int] = -1,
    ) -> Node:
        node_id = self.node_count
        self.node_count += 1
        node = Node(node_id, node_type, label, group_id=group_id)

        if node_type in ["xor", "and", "or"]:
            input_port_1 = node.add_port(Port(self.port_count, "input", node_id, 0, 10))
            self.port_count += 1
            input_port_2 = node.add_port(Port(self.port_count, "input", node_id, 0, 30))
            self.port_count += 1
            _ = node.add_port(Port(self.port_count, "output", node_id, 40, 20))
            self.port_count += 1
            if len(inputs) == 2:
                self.add_edge(inputs[0], input_port_1)
                self.add_edge(inputs[1], input_port_2)
        elif node_type == "not":
            input_port = node.add_port(Port(self.port_count, "input", node_id, 0, 20))
            self.port_count += 1
            _ = node.add_port(Port(self.port_count, "output", node_id, 40, 20))
            self.port_count += 1
            if len(inputs) == 1:
                self.add_edge(inputs[0], input_port)
        elif node_type == "input":
            _ = node.add_port(Port(self.port_count, "output", node_id, 40, 20))
            self.port_count += 1
        elif node_type == "output":
            input_port_1 = node.add_port(Port(self.port_count, "input", node_id, 0, 20))
            self.port_count += 1
            if len(inputs) == 1:
                self.add_edge(inputs[0], input_port_1)
        else:
            print("Unknown node type!")
            exit()

        self.nodes[str(node_id)] = node
        self.node_values[str(node_id)] = 0
        return node

    def add_edge(self, source_port, target_port):
        edge = Edge(source_port.id, target_port.id)
        self.edges.append(edge)
        return edge

    def add_group(self, label="DEFAULT_GROUP"):
        if not self.enable_groups:
            return None
        group_id = self.group_count
        self.group_count += 1
        group = Group(group_id, label)
        self.groups[group.id] = group
        return group

    def add_input_nodes(self, amount, label: Optional[str] = "INPUT"):
        return [self.add_node("input", f"{label}_{i}") for i in range(amount)]

    # Returns the output port for gates and input nodes but raises error for i.e. output nodes
    def get_output_port_of_gate(self, node: Node):
        if node.type in ["xor", "and", "or"]:
            return node.ports[2]
        else:
            raise ValueError(f"Node type: {node.type} is unsupported here.")

    # returns a list
    def get_input_ports_of_node(self, node: Node):
        if node.type in ["xor", "and", "or"]:
            return node.ports[:2]
        elif node.type == "output":
            return [node.ports[0]]
        elif node.type == "not":
            return [node.ports[0]]
        else:
            return []

    # returns the found port or None
    def get_output_port_of_node(self, node: Node):
        if node.type in ["xor", "and", "or"]:
            return node.ports[2]
        elif node.type == "input":
            return node.ports[0]
        elif node.type == "not":
            return node.ports[1]
        else:
            return None

    # Returns the output port of an input node and by that the only port this node has
    def get_input_node_port(self, node: Node):
        if node.type == "input":
            return node.ports[0]
        else:
            raise ValueError(f"Node type: {node.type} is unsupported here.")

    # Returns the output ports for all nodes of a input node list
    def get_input_nodes_ports(self, nodes: List[Node]):
        return [self.get_input_node_port(node) for node in nodes]

    # Returns the input port of an output node and by that the only port this node has
    def get_output_node_port(self, node: Node):
        if node.type == "output":
            return node.ports[0]
        else:
            raise ValueError(f"Node type: {node.type} is unsupported here.")

    # Returns the input ports for all nodes of a output node list
    def get_output_nodes_ports(self, nodes: List[Node]):
        return [self.get_output_node_port(node) for node in nodes]

    def generate_output_node_from_port(self, port: Port, label="OUTPUT"):
        node = self.add_node("output", label, inputs=[port])
        return node

    def generate_output_nodes_from_ports(self, ports: List[Port], label="OUTPUT"):
        nodes = []
        for i, port in enumerate(ports):
            nodes.append(
                self.generate_output_node_from_port(port, label=f"{label}_{i}")
            )
        return nodes

    def fill_node_values(self, nodes: List[Node], bin_list: List[int]):
        for idx, node in enumerate(nodes):
            self.node_values[str(node.node_id)] = bin_list[idx]

    def fill_node_values_via_ports(self, ports: List[Port], bin_list: List[int]):
        for idx, port in enumerate(ports):
            self.node_values[str(port.node_id)] = bin_list[idx]

    def compute_value_from_ports(self, ports: List[Port]):
        bin_list = []
        for port in ports:
            port_value = self.get_port_value(port)
            bin_list.append(port_value)
        return binlist2int(bin_list)

    def get_node_of_port(self, port: Port):
        return self.nodes[str(port.node_id)]

    def compute_target_to_source_port_map(self):
        ports_map = {}
        for edge in self.edges:
            ports_map[edge.target_port_id] = edge.source_port_id
        return ports_map

    def find_port_by_port_id(self, port_id: int) -> Port:
        for node in self.nodes.values():
            for port in node.ports:
                if port.id == port_id:
                    return port
        raise ValueError(f"Port {port_id} not found")

    def compute_port_to_node_mapping(self):
        port_to_node = {}
        for node in self.nodes.values():
            for port in node.ports:
                port_to_node[port.id] = node.node_id
        return port_to_node

    def compute_node_adj_and_in_degrees(self):
        port_to_node_dict = self.compute_port_to_node_mapping()
        in_degrees = defaultdict(int)
        for node_id in self.nodes:
            in_degrees[node_id] = 0
        node_adj = defaultdict(list)
        for edge in self.edges:
            source_node = port_to_node_dict[edge.source_port_id]
            target_node = port_to_node_dict[edge.target_port_id]
            node_adj[source_node].append(target_node)
            in_degrees[target_node] += 1
        return node_adj, in_degrees

    def get_port_value(self, port):
        return self.port_values.get(port.id, None)

    def fill_values(self, nodes, values):
        assert len(nodes) == len(
            values
        ), "In function fill_values length of nodes and values must be the same"
        for idx, node in enumerate(nodes):
            self.node_values[str(node.node_id)] = values[idx]

    def topological_sort(self):
        adj, in_degrees = self.compute_node_adj_and_in_degrees()
        queue = deque([nid for nid, deg in in_degrees.items() if deg == 0])

        topological_order = []
        while queue:
            nid = queue.popleft()
            topological_order.append(self.nodes[nid])
            for neighbor in adj[nid]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(neighbor)

        if len(topological_order) != len(self.nodes):
            raise ValueError("Graph has a cycle or is malformed.")

        return topological_order

    def topological_sort_generator(self):
        adj, in_degrees = self.compute_node_adj_and_in_degrees()
        queue = deque([nid for nid, deg in in_degrees.items() if deg == 0])
        while queue:
            nid = queue.popleft()
            yield self.nodes[nid]
            for neighbor in adj[nid]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(neighbor)

    def to_json(self):
        nodes = [node.to_dict() for node in self.nodes.values()]
        edges = [edge.to_dict() for edge in self.edges]
        data = {
            "nodes": nodes,
            "edges": edges,
            "values": self.node_values,
        }
        if self.enable_groups:
            data["groups"] = [group.to_dict() for group in self.groups.values()]
        return data

    def simulate(self):
        # compute ingoing edges for all nodes
        port_sources = self.compute_target_to_source_port_map()

        # initialize port values of input nodes
        port_values = {}
        for node_id, node in self.nodes.items():
            if node.type == "input":
                val = self.node_values[node_id]
                output_port = node.ports[0]
                port_values[output_port.id] = val

        # iterate through topological order and resolve port values
        for node in self.topological_sort():
            input_ports = [p for p in node.ports if p.type == "input"]
            try:
                inputs = [port_values[port_sources[p.id]] for p in input_ports]
            except KeyError:
                continue
            output_val = None
            if node.type == "input":
                continue
            elif node.type == "output":
                port_values[node.ports[0].id] = inputs[0]
                continue
            else:
                output_val = self.eval_gate(inputs, node.type)
                output_port = [p for p in node.ports if p.type == "output"][0]
                port_values[output_port.id] = output_val

        self.port_values = port_values

    def eval_gate(self, inputs, gate_type):
        if gate_type == "and":
            val = inputs[0] & inputs[1]
        elif gate_type == "or":
            val = inputs[0] | inputs[1]
        elif gate_type == "xor":
            val = inputs[0] ^ inputs[1]
        elif gate_type == "not":
            val = 0 if inputs[0] else 1
        return val

    def longest_path_length(self):
        input_nodes = [
            node.node_id for node in self.nodes.values() if node.type == "input"
        ]
        output_nodes = set(
            node.node_id for node in self.nodes.values() if node.type == "output"
        )
        adj, in_degrees = self.compute_node_adj_and_in_degrees()

        dist = defaultdict(lambda: -float("inf"))
        for nid in input_nodes:
            print(nid)
            dist[nid] = 0

        queue = deque(input_nodes)
        while queue:
            node = queue.popleft()
            for succ in adj[node]:
                dist[succ] = max(dist[succ], dist[node] + 1)
                in_degrees[succ] -= 1
                if in_degrees[succ] == 0:
                    queue.append(succ)

        max_length = max(
            (dist[nid] for nid in output_nodes if dist[nid] != -float("inf")), default=0
        )
        return max_length

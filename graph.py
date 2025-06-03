from node import Node
from port import Port
from edge import Edge
from group import Group

class CircuitGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.groups = {}
        self.node_count = 0
        self.node_values = {}
        self.port_count = 0
        self.group_count = 0

    def add_node(self, node_type, label, inputs=[], group_id=-1):
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
        group_id = self.group_count
        self.group_count += 1
        group = Group(group_id, label)
        self.groups[group.id] = group
        return group

    def to_json(self):
        nodes = [node.to_dict() for node in self.nodes.values()]
        edges = [edge.to_dict() for edge in self.edges]
        groups = [group.to_dict() for group in self.groups.values()] 
        return {"nodes": nodes, "edges": edges, "groups": groups, "values": self.node_values}
    
    def simulate(self):
        port_values = {}
        port_sources = {}
        
        for edge in self.edges:
            port_sources[edge.target_port_id] = edge.source_port_id
        
        for node_id, node in self.nodes.items():
            if node.type == "input":
                val = self.node_values[node_id]
                output_port = node.ports[0]
                port_values[output_port.id] = val

        remaining = set(self.nodes.keys()) - {nid for nid, n in self.nodes.items() if n.type == "input"}
        resolved = set()

        while remaining:
            progress = False
            for node_id in list(remaining):
                node = self.nodes[node_id]

                input_ports = [p for p in node.ports if p.type == "input"]
                try:
                    inputs = [port_values[port_sources[p.id]] for p in input_ports]
                except KeyError:
                    continue

                output_val = None
                if node.type == "and":
                    output_val = inputs[0] & inputs[1]
                elif node.type == "or":
                    output_val = inputs[0] | inputs[1]
                elif node.type == "xor":
                    output_val = inputs[0] ^ inputs[1]
                elif node.type == "not":
                    output_val = 0 if inputs[0] else 1
                elif node.type == "output":
                    port_values[node.ports[0].id] = inputs[0]
                    resolved.add(node_id)
                    remaining.remove(node_id)
                    progress = True
                    continue
                else:
                    continue

                output_port = [p for p in node.ports if p.type == "output"][0]
                port_values[output_port.id] = output_val
                resolved.add(node_id)
                remaining.remove(node_id)
                progress = True

            if not progress:
                raise RuntimeError("Simulation stalled; possible cycle or unconnected inputs.")

        self.port_values = port_values

    def get_port_value(self, port):
        return self.port_values.get(port.id, None)
    
    def fill_values(self, nodes, values):
        assert len(nodes) == len(values), "In function fill_values length of nodes and values must be the same"
        for idx, node in enumerate(nodes):
            self.node_values[str(node.node_id)] = values[idx]
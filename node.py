class Node:
    def __init__(self, node_id, node_type, node_label, group_id=-1, x=0, y=0, value=0):
        self.node_id = node_id
        self.type = node_type
        self.label = node_label
        self.group_id = group_id
        self.x = x
        self.y = y
        self.value = value
        self.ports = []

    def add_port(self, port):
        self.ports.append(port)
        return port
    
    def to_dict(self):
        return {
            "id": self.node_id,
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "value": self.value,
            "label": self.label,
            "group_id": self.group_id,
            "ports": [port.to_dict() for port in self.ports]
        }
    
    def __repr__(self):
        return f"label: {self.label}"
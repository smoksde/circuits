class Port:
    def __init__(self, port_id, port_type, node_id, x, y):
        self.id = port_id
        self.type = port_type
        self.node_id = node_id
        self.x = x
        self.y = y

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "nodeId": self.node_id,
        }

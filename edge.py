class Edge:
    def __init__(self, source_port_id, target_port_id):
        self.source_port_id = source_port_id
        self.target_port_id = target_port_id

    def to_dict(self):
        return {
            "sourcePortId": self.source_port_id,
            "targetPortId": self.target_port_id,
        }

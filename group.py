class Group:
    def __init__(self, group_id, group_label, collapsed=False):
        self.id = group_id
        self.label = group_label
        self.parent = None # holds group
        self.collapsed = collapsed
    
    def set_parent(self, parent):
        self.parent = parent

    def collapse(self):
        self.collapsed = True

    def expand(self):
        self.collapsed = False

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "parent": self.parent.id if self.parent else -1,
        }

    def __repr__(self):
        return f""


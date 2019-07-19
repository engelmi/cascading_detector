from cascading_detector.serializable import Serializable


class Detectable(Serializable):

    def __init__(self, class_id, width, height, parent=None):
        self.class_id = class_id
        self.size = (width, height)
        if parent is not None and not isinstance(parent, Detectable):
            raise Exception("Parameter 'parent' must be of type '" + type(Detectable) + "'!")
        self.parent = parent
        self.children = []

    def add_child(self, child):
        if not isinstance(child, Detectable):
            raise Exception("Can not add object of type '" + type(child) + "' to list of children!")
        child.set_parent(self)
        self.children.append(child)

    def set_parent(self, parent):
        if not isinstance(parent, Detectable):
            raise Exception("Can not set object of type '" + type(parent) + "' as parent!")
        self.parent = parent

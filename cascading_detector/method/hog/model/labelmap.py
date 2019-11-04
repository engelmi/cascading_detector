from cascading_detector.serializable import Serializable


class LabelMap(Serializable):

    def __init__(self):
        self.cls_label_dict = dict()

    def get_cls_label_name(self, cls_id):
        if cls_id not in self.cls_label_dict.keys():
            return None
        return self.cls_label_dict[cls_id]

    def get_cls_label_ids(self, cls_name_to_find):
        cls_label_ids = []
        for cls_id, cls_name in self.cls_label_dict.items():
            if cls_name == cls_name_to_find:
                cls_label_ids.append(cls_id)
        return cls_label_ids

    def add_label(self, cls_name, cls_id=None):
        if cls_id is None:
            cls_id = self.next_cls_id()
        if cls_id not in self.cls_label_dict.keys():
            self.cls_label_dict[cls_id] = cls_name
            return True
        return False

    def next_cls_id(self):
        cls_id = len(self.cls_label_dict)
        while cls_id in self.cls_label_dict.keys():
            cls_id += 1
        return cls_id

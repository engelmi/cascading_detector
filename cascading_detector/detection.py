from cascading_detector.serializable import Serializable


class Detection(Serializable):

    def __init__(self, cls_id, xmin, ymin, xmax, ymax, score, sub_detection_list=[]):
        self.cls_id = cls_id
        self.bounding_box = {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax
        }
        self.score = score
        self.sub_detection_list = []

    def add_sub_detection(self, sub_detection):
        if not isinstance(sub_detection, Detection):
            raise Exception("Can not add object of type '" + str(sub_detection) + "' to list of sub detections. "
                                                                                  "Type '" + str(Detection) + "' required!")
        self.sub_detection_list.append(sub_detection)

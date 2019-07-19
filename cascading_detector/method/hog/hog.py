from cascading_detector.cdetector import CascadingDetector


class HOG(CascadingDetector):

    def __init__(self, path_to_trained_model):
        super(HOG, self).__init__()

        self.path_to_trained_model = path_to_trained_model

    def __new__(cls, *args, **kwargs):
        pass

    def detect_objects_in_np_image(self, np_image):
        pass

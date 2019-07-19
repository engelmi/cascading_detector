
import numpy as np
from PIL import Image
from os.path import isfile, exists


class CascadingDetector(object):

    def __init__(self):
        pass

    def detect_objects(self, image):
        if isinstance(image, str):
            return self.detect_objects_in_image(image)
        elif Image.isImageType(image):
            return self.detect_objects_in_pil_image(image)
        elif isinstance(image, np.ndarray):
            return self.detect_objects_in_np_image(image)
        return None

    def detect_objects_in_pil_image(self, pil_image):
        return self.detect_objects_in_np_image(np.array(pil_image))

    def detect_objects_in_np_image(self, np_image):
        raise NotImplementedError("Must be implemented by specialized detector!")

    def detect_objects_in_image(self, path_to_image):
        if not isfile(path_to_image):
            raise FileNotFoundError("Could not find file '" + str(path_to_image) + "'")
        return self.detect_objects_in_pil_image(Image.open(path_to_image))


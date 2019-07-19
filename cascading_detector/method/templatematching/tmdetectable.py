from PIL import Image

from cascading_detector.detectable import Detectable
from cascading_detector.tools.typeconv import pil_to_byte


class TMDetectable(Detectable):

    def __init__(self, class_id, template_image_pil, threshold=0.8, parent=None):
        if not Image.isImageType(template_image_pil):
            raise Exception("Template must be of type '" + str(Image) + "'!")
        width, height = template_image_pil.size
        super(TMDetectable, self).__init__(class_id, width, height, parent)

        self.threshold = threshold
        # template matching only supports greyscale images (currently)
        greyscale_image = template_image_pil.convert("L")
        # keep the image format, like png (gets lost during conversion)
        greyscale_image.format = template_image_pil.format
        self.template_image_bytes = pil_to_byte(greyscale_image)

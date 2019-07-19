import io
import numpy as np
from PIL import Image


def pil_to_byte(pil_image):
    if not Image.isImageType(pil_image):
        raise Exception("Image must be of type '" + str(Image) + "'!")
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=pil_image.format)
    return img_byte_arr.getvalue()


def pil_to_numpy(pil_image):
    if not Image.isImageType(pil_image):
        raise Exception("Image must be of type '" + str(Image) + "'!")
    return np.array(pil_image)


def byte_to_pil(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


def byte_to_numpy(image_bytes):
    return pil_to_numpy(byte_to_pil(image_bytes))


def numpy_to_pil(numpy_arr, mode=None):
    return Image.fromarray(numpy_arr, mode)


def numpy_to_byte(numpy_arr):
    return pil_to_byte(numpy_to_pil(numpy_arr))

from PIL import ImageDraw


def draw_bounding_box_on(pil_image, bounding_box):
    xmin = bounding_box["xmin"]
    xmax = bounding_box["xmax"]
    ymin = bounding_box["ymin"]
    ymax = bounding_box["ymax"]
    ImageDraw.Draw(pil_image).rectangle([(xmin, ymin), (xmax, ymax)], fill=None, outline=(255, 0, 0))


def draw_bounding_boxes_on(pil_image, bounding_box_list):
    for bounding_box in bounding_box_list:
        draw_bounding_box_on(pil_image, bounding_box)

import numpy as np
from PIL import Image
from collections import deque
from skimage.feature import match_template

from cascading_detector.detection import Detection
from cascading_detector.cdetector import CascadingDetector
from cascading_detector.tools.typeconv import byte_to_numpy
from cascading_detector.method.templatematching.tmmodel import TMModel


class TemplateMatching(CascadingDetector):

    def __init__(self, model):
        super(TemplateMatching, self).__init__()
        self.model = None
        self.set_model(model)

    def set_model(self, model):
        if not isinstance(model, TMModel):
            raise Exception("Only a model from type '" + str(TMModel) + "' can be used for Template Matching!")
        if model.detectables is None:
            raise Exception("Detectables of model must not be None!")
        self.model = model

    def detect_objects_in_pil_image(self, pil_image):
        return self.detect_objects_in_np_image(np.array(pil_image.convert("L")))

    def detect_objects_in_np_image(self, np_image):

        if np_image.ndim != 2:
            raise Exception("Current Template Matching implementation only supports greyscale images!")

        detection_dict = dict()
        if self.model is not None and isinstance(self.model, TMModel):

            for filename,_ in self.model.detectables.items():
                detectable = None
                if self.model.load_type == TMModel.LoadType.EAGER:
                    detectable = self.model.detectables[filename]
                elif self.model.load_type == TMModel.LoadType.LAZY:
                    detectable = TMModel.load_detectable(filename)
                if detectable is None:
                    continue

                base_detection = Detection(-1, 0, 0, np_image.shape[1], np_image.shape[0])
                detection_dict[detectable.class_id] = base_detection

                to_explore = deque()
                to_explore.append((base_detection, detectable))
                while to_explore:
                    parent_detection, to_detect = to_explore.popleft()
                    pxmin = int(parent_detection.bounding_box["xmin"])
                    pxmax = int(parent_detection.bounding_box["xmax"])
                    pymin = int(parent_detection.bounding_box["ymin"])
                    pymax = int(parent_detection.bounding_box["ymax"])
                    np_template_image = byte_to_numpy(to_detect.template_image_bytes)
                    detections = self.apply_template_matching(
                        to_detect.class_id,
                        np_image[pymin:pymax, pxmin:pxmax],
                        np_template_image, to_detect.threshold)
                    for detected in detections:
                        # add parent detection xmin and ymin to detections for absolute coordinates
                        detected.bounding_box["xmin"] = int(detected.bounding_box["xmin"] + pxmin)
                        detected.bounding_box["ymin"] = int(detected.bounding_box["ymin"] + pymin)
                        detected.bounding_box["xmax"] = int(detected.bounding_box["xmax"] + pxmin)
                        detected.bounding_box["ymax"] = int(detected.bounding_box["ymax"] + pymin)
                        parent_detection.add_sub_detection(detected)
                        for child in to_detect.children:
                            to_explore.append((detected, child))
        return detection_dict

    def apply_template_matching(self, cls_id, input_image_np, template_image_np, threshold):
        detections = []
        result_map = match_template(input_image_np, template_image_np, pad_input=True)
        height, width = template_image_np.shape

        locs = {}
        id = 0
        loc_x, loc_y = np.where(result_map>=threshold)
        # find adjacent points
        for x,y in zip(loc_x, loc_y):
            is_adj = False
            for key, values in locs.items():
                for value in values:
                    vx, vy = value
                    if (x + 1 == vx or x - 1 == vx or x == vx) and (y + 1 == vy or y - 1 == vy or y == vy):
                        locs[key].append((x,y))
                        is_adj = True
                        break
            if not is_adj:
                locs[id] = []
                locs[id].append((x,y))
                id += 1
        loc_centers = []
        # find points with highest confidence
        for _, values in locs.items():
            max_val = 0
            max_val_loc = (None)
            for x,y in values:
                if result_map[x][y] > max_val:
                    max_val = result_map[x][y]
                    max_val_loc = (x,y)
            loc_centers.append(max_val_loc)

        for y,x in loc_centers:
            detections.append(Detection(cls_id, x - int(width/2.0), y - int(height/2.0),
                                        x + int(width/2.0), y + int(height/2.0))
                              )
        return detections

    def generate_heatmap(self, input_image_np, template_image_np, threshold):
        result = match_template(input_image_np, template_image_np, pad_input=True)
        heatmap = Image.new("RGB", (input_image_np.shape[1], input_image_np.shape[0]))
        heatmap_pix = heatmap.load()
        for x in range(heatmap.size[0]):
            for y in range(heatmap.size[1]):
                v = int(255*result[y][x])
                if result[y][x] >= threshold:
                    heatmap_pix[x, y] = (v, 0, 0)
                elif result[y][x] > 0:
                    heatmap_pix[x, y] = (v, v, v)
                else:
                    heatmap_pix[x, y] = (0, 0, 0)
        return heatmap

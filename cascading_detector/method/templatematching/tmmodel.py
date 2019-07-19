from enum import Enum
from os.path import isfile

from cascading_detector.serializable import Serializable
from cascading_detector.method.templatematching.tmdetectable import TMDetectable


class TMModel(Serializable):

    class LoadType(Enum):
        LAZY = 1
        EAGER = 2

    def __init__(self, detectable_file_list, load_type=LoadType.LAZY):
        TMModel.check_validity_of_detectable_list(detectable_file_list)
        self.detectables = {filename: None for filename in detectable_file_list}
        self.load_type = load_type
        if self.load_type == TMModel.LoadType.EAGER:
            for detectable_filename in detectable_file_list:
                self.detectables[detectable_filename] = TMModel.load_detectable(detectable_filename)

    @classmethod
    def check_validity_of_detectable_list(cls, file_list):
        if file_list is None:
            raise Exception("Detectable file list must not be None!")
        if not isinstance(file_list, list):
            raise Exception("Detectable files names must be provided as list!")
        for file in file_list:
            if not isfile(file):
                raise Exception("Detected invalid file '" + file + "' in list of detectables!")

    @classmethod
    def load_detectable(cls, detectable_filename):
        detectable = Serializable.read_from_file(detectable_filename)
        if not isinstance(detectable, TMDetectable):
            return None
        return detectable

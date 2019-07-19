import json
import jsonpickle
from os.path import isfile


class Serializable(object):

    def __init__(self):
        pass

    def write_to_file(self, filename, override=False):
        if isfile(filename) and not override:
            raise NotADirectoryError("File '" + filename + "' already exists!")
        with open(filename, "w") as f:
            json.dump(jsonpickle.encode(self), f)

    def read_from_file(path_to_file):
        if not isfile(path_to_file):
            raise FileNotFoundError("Couldn't find file '" + path_to_file + "'")
        with open(path_to_file, "rb") as f:
            read_detectable = jsonpickle.decode(json.load(f))
        return read_detectable

    def to_json_str(self):
        return jsonpickle.encode(self)

    def to_json_dict(self):
        return json.loads(self.to_json_str())

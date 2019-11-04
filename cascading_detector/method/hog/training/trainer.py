from enum import Enum


class TrainingEntryPoint(Enum):
    """
    Possible entry points in the training process of a ClassificationTrainer.
    """
    EXTRACTING_1 = 1
    PREPROCESSING_1 = 2
    TRAINING = 3
    MINING = 4
    EXTRACTING_2 = 5
    PREPROCESSING_2 = 6
    RE_TRAINING = 7


class Trainer(object):

    def __init__(self):
        pass

    def start_training(self):
        raise Exception("Method must be implemented!")

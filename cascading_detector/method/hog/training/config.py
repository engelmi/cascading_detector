from cascading_detector.serializable import Serializable


# todo: find a more suitable method for configuration
class TrainerConfig(Serializable):

    def __init__(self):
        # name of the trainer
        self.trainer_name = ""

        # used directories
        self.base_dir = "cascading-detector"
        self.model_dir = "model"
        self.data_dir = "data"
        self.raw_data_dir = "raw-data"
        self.train_dir = "train"
        self.test_dir = "test"
        self.preprocessed_data_dir = "preprocessed-data"
        self.positives_dir = "positives"
        self.negatives_dir = "negatives"
        self.mined_negatives_dir = "mined-negatives"
        self.image_extension = ".png"

        self.result_dir = "result"

        # saved training objects (without data files)
        saved_trainer_config = "config"
        saved_labelmap = "labelmap"
        scaler_nomines = "scaler_nomines"
        pca_nomines = "pca_nomines"
        classifier_only_nomines = "classifier_only_nomines"
        classifier_all_nomines = "classifier_all_nomines"
        scaler_mines = "scaler_mines"
        pca_mines = "pca_mines"
        classifier_only_mines = "classifier_only_mines"
        classifier_all_mines = "classifier_all_mines"


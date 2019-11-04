import joblib
import numpy as np
from os import listdir
from os.path import join

from cascading_detector.method.hog.training.config import TrainerConfig
from cascading_detector.method.hog.training.trainer import Trainer, TrainingEntryPoint


class ClassificationTrainer(Trainer):

    def __init__(self, conf):
        if not isinstance(conf, TrainerConfig):
            raise Exception("")
        self.conf = conf

    def start_training(self, entry_point=TrainingEntryPoint.EXTRACTING_1):
        """
        Process for training a classifier. Saves the resulting data and classifier in the directories provided by the config.
        If overridden, call the super function to use the predefined training routine!
        :param entry_point: The entry point in the training process.
        """
        if not isinstance(entry_point, TrainingEntryPoint):
            raise Exception("Entry point not valid!")

        self.save_scikit_object(self.conf, self.conf.base_dir)

        if entry_point.value <= TrainingEntryPoint.EXTRACTING_1.value:
            X_p, Y_p, Y_labelmap = self.extract_positives()
            X_n = self.extract_negatives(X_p.shape[0])
            self.save_data(X_p, "X_p")
            self.save_data(Y_p, "Y_p")
            self.save_data(X_n, "X_n")
            self.save_labelmap(Y_labelmap, self.conf.saved_labelmap)
        if entry_point.value <= TrainingEntryPoint.PREPROCESSING_1.value:
            X_p = self.load_data("X_p")
            Y_p = self.load_data("Y_p")
            X_n = self.load_data("X_n")
            X = np.vstack((X_p, X_n))
            Y = np.hstack(( Y_p, np.zeros((X_n.shape[0])) ))
            X_preproc_nomines, scaler_nomines, pca_nomines = self.preprocess_data(X)
            self.save_data(X_preproc_nomines, "X_preproc_nomines")
            self.save_data(Y, "Y_preproc_nomines")
            self.save_scikit_object(scaler_nomines, self.conf.scaler_nomines)
            self.save_scikit_object(pca_nomines, self.conf.pca_nomines)
        if entry_point.value <= TrainingEntryPoint.TRAINING.value:
            labelmap = self.load_labelmap(self.conf.saved_labelmap)
            X_preproc_nomines = self.load_data("X_preproc_nomines")
            Y_preproc_nomines = self.load_data("Y_preproc_nomines")
            scaler_nomines = self.load_scikit_object(self.conf.scaler_nomines)
            pca_nomines = self.load_scikit_object(self.conf.pca_nomines)
            trained_classifier = self.train_classifier(X_preproc_nomines, Y_preproc_nomines)
            self.save_scikit_object(trained_classifier, self.conf.classifier_only_nomines)
            self.save_classifier(trained_classifier, scaler_nomines, pca_nomines, labelmap, self.conf.classifier_all_nomines)
        if entry_point.value <= TrainingEntryPoint.MINING.value:
            classifier, scaler, pca, labelmap = self.load_classifier(self.conf.classifier_all_nomines)
            self.mine_hard_negatives(classifier, scaler, pca, labelmap)
        if entry_point.value <= TrainingEntryPoint.EXTRACTING_2.value:
            X_hmn, Y_hmn = self.extract_mined_negatives()
            self.save_data(X_hmn, "X_hmn")
            self.save_data(Y_hmn, "Y_hmn")
        if entry_point.value <= TrainingEntryPoint.PREPROCESSING_2.value:
            X_p = self.load_data("X_p")
            Y_p = self.load_data("Y_p")
            X_n = self.load_data("X_n")
            X_hmn = self.load_data("X_hmn")
            Y_hmn = self.load_data("Y_hmn")
            X = np.vstack(( X_p, X_n ))
            Y = np.hstack(( Y_p, np.zeros((X_n.shape[0])) ))
            if X_hmn.shape[0] > 0 and Y_hmn.shape[0] > 0:
                X = np.vstack((X, X_hmn))
                Y = np.hstack((Y, Y_hmn))
            X_preproc_mines, scaler_mines, pca_mines = self.preprocess_data(X)
            self.save_data(X_preproc_mines, "X_preproc_mines")
            self.save_data(Y, "Y_preproc_mines")
            self.save_scikit_object(scaler_mines, self.conf.scaler_mines)
            self.save_scikit_object(pca_mines, self.conf.pca_mines)
        if entry_point.value <= TrainingEntryPoint.RE_TRAINING.value:
            labelmap = self.load_labelmap(self.conf.saved_labelmap)
            X_preproc_mines = self.load_data("X_preproc_mines")
            Y_preproc_mines = self.load_data("Y_preproc_mines")
            scaler_mines = self.load_scikit_object(self.conf.scaler_mines)
            pca_mines = self.load_scikit_object(self.conf.pca_mines)
            trained_classifier = self.train_classifier(X_preproc_mines, Y_preproc_mines)
            self.save_scikit_object(trained_classifier, self.conf.classifier_only_mines)
            self.save_classifier(trained_classifier, scaler_mines, pca_mines, labelmap, self.conf.classifier_all_mines)

    def extract_positives(self):
        """
        """
        raise Exception("Method must be implemented!")

    def extract_negatives(self, num_positives):
        """
        """
        raise Exception("Method must be implemented!")

    def extract_mined_negatives(self):
        """
        """
        raise Exception("Method must be implemented!")

    def preprocess_data(self, X):
        """
        """
        raise Exception("Method must be implemented!")

    def train_classifier(self, X, Y):
        """
        """
        raise Exception("Method must be implemented!")

    def mine_hard_negatives(self, classifier, scaler, pca, labelmap):
        """
        """
        raise Exception("Method must be implemented!")

    def save_labelmap(self, lblmap, name):
        """
        """
        with open(join(self.conf.base_dir, self.conf.data_dir, self.conf.train_dir, name), "w") as f:
            for id, name in lblmap.items():
                f.write("item{\n\tid: " + str(id) + "\n\tname: " + str(name) + "\n}\n")

    def load_labelmap(self, name):
        """
        """
        lblmap = {}
        with open(join(self.conf.base_dir, self.conf.data_dir, self.conf.train_dir, name), "r") as f:
            classes = f.read().replace('item{','').replace('\n','').split('}')[:-1]
            for cls in classes:
                parts = cls.split('\t')
                id, txt = parts[1], parts[2]
                id = int(id.replace('id:', ''))
                txt = txt.replace('name:', '').replace(' ', '')
                lblmap[id] = txt
        return lblmap

    def save_data(self, data, name):
        """
        """
        np.save(join(self.conf.base_dir, self.conf.data_dir, self.conf.train_dir, name), data)

    def load_data(self, name):
        """
        """
        return np.load(join(self.conf.base_dir, self.conf.data_dir, self.conf.train_dir, name + ".npy"))

    def save_scikit_object(self, obj, name):
        """
        """
        joblib.dump(obj, join(self.conf.base_dir, self.conf.model_dir, name))

    def load_scikit_object(self, name):
        """
        """
        return joblib.load(join(self.conf.base_dir, self.conf.model_dir, name))

    def save_classifier(self, classifier, scaler, pca, lblmap, name):
        """
        """
        joblib.dump((classifier, scaler, pca, lblmap), join(self.conf.base_dir, self.conf.model_dir, name))

    def load_classifier(self, name):
        """
        """
        return joblib.load(join(self.conf.base_dir, self.conf.model_dir, name))

    def load_trained_model(self, name):
        """
        """
        return joblib.load(join(self.conf.base_dir, self.conf.model_dir, name))

    def get_image_list(self, path, image_extension):
        """
        Helper function: Get a list of all images in the given path.
        :return: List of all images in given path.
        """
        return [join(path, f) for f in listdir(path) if f.endswith(image_extension)]

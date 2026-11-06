import numpy as np

class ClassificationMetrics():
    def __init__(self, y_true, y_pred, sample_weights = None, lables = None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weights = sample_weights
        self.labels = lables

    def score(measure, average = "binary"):
        if measure == "f1":
            pass
        if measure == "precision":
            pass
        if measure == "recall":
            pass

    def accuracy(matrix = True, normalize = None):
        #accuracy and confusion matrix
        pass

    def loss(normalize = True, custom = False):
        if custom == False:
            #log
            pass
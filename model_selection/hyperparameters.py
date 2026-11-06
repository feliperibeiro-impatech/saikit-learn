import numpy as np

class gridSearchCV():
    def __init__(self, estimator : object, param_grid : dict, scoring = None, refit = False, cv = 5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.score = scoring
        self.cv = cv

    def fit(self, x, y):
        return None
        self.estimator(param_grid[0][0], param_grid[1][0])
        self.estimator.fit()

    def score(self, x, y):
        return None
        self.estimator.predict()
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, name: str = 'NaiveBayes', n_features: int = 0):
        self.name = name
        self.n_features = n_features
        self.estimator = GaussianNB()
        self.parameter_grid = {}

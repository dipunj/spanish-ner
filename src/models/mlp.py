import re
from regex import P
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from src.models.perceptron import Perceptron


class MLPerceptron(Perceptron):
    """
        Wrapper for the MLPClassifier from sklearn.
        Uses Perceptron as a base class to inherit the data_pipeline method.
    """

    def __init__(self, dataset):

        super().__init__(dataset)

        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 20, ), batch_size=128, verbose=2, max_iter=10)

        # because dataset.train_X is an array of dicts
        self.vectorizer = DictVectorizer()

        self.train_X, self.train_Y = self.data_pipeline(dataset.train)
        self.dev_X, self.dev_Y = self.data_pipeline(dataset.dev)
        self.test_X, self.test_Y = self.data_pipeline(dataset.test)

    def train(self):
        """
            Train the model on the training set
        """

        print(f"Training model, size of training data: {len(self.train_X)}")

        X_vectorized = self.train_X
        if self.vectorizer:
            X_vectorized = self.vectorizer.fit_transform(self.train_X)
        else:
            raise Exception("Vectorizer not initialized")

        self.model.fit(X_vectorized, self.train_Y)

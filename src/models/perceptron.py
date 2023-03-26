import re
import sklearn.linear_model
from sklearn.feature_extraction import DictVectorizer
from src.models.base import BaseModel


class Perceptron(BaseModel):
    """
        Wrapper for the Perceptron from sklearn.
    """

    def __init__(self, dataset):

        super().__init__(dataset)
        self.model = sklearn.linear_model.Perceptron(verbose=True)
        # because dataset.train_X is an array of dicts
        self.vectorizer = DictVectorizer()
        self.train_X, self.train_Y = self.data_pipeline(dataset.train)
        self.dev_X, self.dev_Y = self.data_pipeline(dataset.dev)
        self.test_X, self.test_Y = self.data_pipeline(dataset.test)

    def data_pipeline(self, data):
        """
            Preprocess the data into the format that the model expects
        """
        X = []
        y = []

        for sentence in data:
            for i, word in enumerate(sentence):
                label = word[-1]
                X.append(self.word_to_features(sentence, i))
                y.append(label)

        return X, y

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

    def predict(self, data):
        """
            Predict the labels for the test set
        """
        # we don't call fit_transform because fit_transform will recalculate the mean and variance using test data
        # we want to use the same mean and variance as the training data
        # so we call transform instead

        X, y = self.data_pipeline(data)
        X_vectorized = self.vectorizer.transform(X)
        y_pred = self.model.predict(X_vectorized)

        return y_pred

    def get_window(self, centerWordIdx, sentence):
        """
            Returns a list of tuples (position, offset).
        """
        positions = []
        for offset in [-2, -1, 0, 1, 2]:
            offsetPosition = centerWordIdx + offset
            isPositionValid = offsetPosition >= 0 and offsetPosition < len(
                sentence)
            if isPositionValid:
                positions.append((offsetPosition, offset))

        return positions

    def is_punctuation(self, word):
        """
            Returns true if the word has a punctuation mark present(e.g. ).
        """
        # .,?!:;()'"[]{}
        punctuations = ['.', ',', '?', '!', ':', ';',
                        '(', ')', '"', "'", '[', ']', '{', '}']

        for punctuation in punctuations:
            if punctuation in word:
                return True
        return False

    def word_to_features(self, sentence, wordIdx):
        """ 
            The function generates all features
            for the word at position i in the sentence.
        """

        # features[i] is  for word at position i in the sentence
        features = []

        # the window around the token
        window = self.get_window(wordIdx, sentence)

        for position, offset in window:
            word, pos_tag, ner = sentence[position]

            feat = [
                (str(offset) + '_punctuation', self.is_punctuation(word)),
                (str(offset) + '_word', word.lower()),
                (str(offset) + '_pos_tag', pos_tag),
                (str(offset) + '_upper_case', word[0].isupper()),
            ]
            features.extend(feat)

        return dict(features)

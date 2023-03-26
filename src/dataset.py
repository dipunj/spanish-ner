
from nltk.corpus import conll2002
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def save_to_disk(data, filename):
    """
        Save the data to filename
    """

    print("Saving to data to disk", filename)
    with open(filename, 'w') as f:
        for sentence in data:
            for word, pos, label in sentence:
                f.write(f'{word} {pos} {label}\n')
            f.write('\n')


def load_data_from_disk(filename):
    """
        Load and return the data from filename
    """

    data = []
    print("Loading data from disk", filename)
    with open(filename, 'r') as f:
        sentence = []
        for line in f:
            if line == '\n':
                data.append(sentence)
                sentence = []
            else:
                word, pos, label = line.strip().split()
                sentence.append((word, pos, label))
    return data


def load_data():
    """
        Load the data from disk or download it if it's not available
    """
    raw_data_path = os.path.join(curr_dir, '..', 'data', 'raw')

    train_path = os.path.join(raw_data_path, 'train.txt')
    if not os.path.exists(train_path):
        train = list(conll2002.iob_sents('esp.train'))
        save_to_disk(train, train_path)
    else:
        train = load_data_from_disk(train_path)

    test_path = os.path.join(raw_data_path, 'test.txt')
    if not os.path.exists(test_path):
        test = list(conll2002.iob_sents('esp.testa'))
        save_to_disk(test, test_path)
    else:
        test = load_data_from_disk(test_path)

    dev_path = os.path.join(raw_data_path, 'dev.txt')
    if not os.path.exists(dev_path):
        dev = list(conll2002.iob_sents('esp.testb'))
        save_to_disk(dev, dev_path)
    else:
        dev = load_data_from_disk(dev_path)

    return NERDataset(train, dev, test)


UNIQUE_LABELS = [
    'B-LOC',
    'B-MISC',
    'B-ORG',
    'B-PER',
    'I-LOC',
    'I-MISC',
    'I-ORG',
    'I-PER',
    'O'
]


class NERDataset:
    """
        A class to hold the NER dataset
    """

    def __init__(self, train, dev, test):

        # train[0] = [
        # ('Melbourne', 'NP', 'B-LOC')
        # ('(', 'Fpa', 'O')
        # ('Australia', 'NP', 'B-LOC')
        # (')', 'Fpt', 'O')
        # (',', 'Fc', 'O')
        # ('25', 'Z', 'O')
        # ('may', 'NC', 'O')
        # ('(', 'Fpa', 'O')
        # ('EFE', 'NC', 'B-ORG')
        # (')', 'Fpt', 'O')
        # ('.', 'Fp', 'O')
        # ]

        # train is an array of sentences
        # each sentence is an array of tuples
        # each tuple is a word, its POS tag, and its NER tag

        self.train = train
        self.dev = dev
        self.test = test

        self.unique_labels = UNIQUE_LABELS

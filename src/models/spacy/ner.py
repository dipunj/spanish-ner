import os
import subprocess
from tqdm import tqdm
import spacy
from spacy.tokens import Doc
from src.models.base import BaseModel


curr_dir = os.path.dirname(os.path.realpath(__file__))


class BertNER(BaseModel):
    """
        This class is a wrapper for the spacy NER model
    """

    def __init__(self, dataset, model_path=None):
        super().__init__(dataset)

        data_path = os.path.join(curr_dir, '..', '..', '..', 'data')
        raw_data_path = os.path.join(data_path, 'raw')
        self.processed_data_path = os.path.join(data_path, 'spacy')
        self.paths = {
            'models_path': os.path.join(curr_dir, 'models'),
            'train_raw': os.path.join(raw_data_path, 'train.txt'),
            'dev_raw': os.path.join(raw_data_path, 'dev.txt'),
            'train_processed': os.path.join(self.processed_data_path, 'train.spacy'),
            'dev_processed': os.path.join(self.processed_data_path, 'dev.spacy')
        }

        self.best_model_path = model_path or os.path.join(
            self.paths['models_path'], 'model-best')

        self.data_pipeline()

    def data_pipeline(self):
        """
            Preprocess the data into the format that the model expects
        """

        if not os.path.exists(self.processed_data_path):
            # make the directory
            os.mkdir(self.processed_data_path)

        if not os.path.exists(self.paths['models_path']):
            os.mkdir(self.paths['models_path'])

        if not os.path.exists(self.paths['train_processed']):
            subprocess.call(['python', '-m', 'spacy', 'convert',
                             '-n', '10',
                             '--converter', 'iob',
                             self.paths['train_raw'], self.processed_data_path])

        if not os.path.exists(self.paths['dev_processed']):
            subprocess.call(['python', '-m', 'spacy', 'convert',
                             '-n', '10',
                             '--converter', 'iob',
                             self.paths['dev_raw'], self.processed_data_path])

    def generate_train_config(self):
        """
            Generate the training config file if it doesn't exist
        """

        self.paths['base_config_path'] = os.path.join(
            curr_dir, 'base_config.cfg')
        self.paths['config_path'] = os.path.join(curr_dir, 'config.cfg')

        if not os.path.exists(self.paths['config_path']):
            return False

        if not os.path.exists(self.paths['config_path']):
            # create the training config file
            subprocess.call(['python', '-m', 'spacy', 'init',
                            'fill-config',
                             self.paths['base_config_path'],
                             self.paths['config_path']])

        return True

    def train(self, force_retrain=False):
        """
            Train/Fine tune the model. If the model already exists, skip training
        """

        if (not force_retrain and os.path.exists(self.best_model_path)):
            print(
                "Model already exists. Skipping training. To force retrain, pass force_retrain=True")
            return

        if not self.generate_train_config():

            print("Base config file not found. Please download it, as per your system, \
                  from https://spacy.io/usage/training and place it in the models/spacy directory")
            return

        print(f"Using config file: {self.paths['config_path']} for training")
        subprocess.call(['python', '-m',
                         'spacy', 'train', self.paths['config_path'],
                         '--gpu-id', '0',
                         '--output', self.paths['models_path'],
                         '--paths.train', self.paths['train_processed'],
                         '--paths.dev', self.paths['dev_processed']])

    def predict(self, X):
        """
            Predict the labels for the test set
        """

        if (not os.path.exists(self.paths['models_path']) and not self.best_model_path):
            print(
                "Model not found, please call train the model first or supply the model path as command line argument")

        best_model = self.best_model_path
        ner = spacy.load(best_model)

        pred = []
        for sent in tqdm(X):

            words = [word for word, _, _ in sent]
            spaces = [True] * (len(words) - 1) + [False]

            # the tokenizer used by spacy doesn't tokenize the words as per the dataset
            # therefore create a tokenized doc object
            doc = ner(Doc(ner.vocab, words=words, spaces=spaces))

            for tk in doc:
                if tk.ent_iob_ == 'O':
                    pred.append('O')
                else:
                    iob = f"{tk.ent_iob_}-{tk.ent_type_}"
                    pred.append(iob)

        return pred

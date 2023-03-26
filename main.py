import sys
import os
from datetime import datetime
from src.dataset import load_data
from src.models.mlp import MLPerceptron
from src.models.perceptron import Perceptron
from src.models.spacy.ner import BertNER

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(model_path=None):

    # load the data
    print("Loading data...")
    dataset = load_data()

    print("Creating model...")
    model = BertNER(dataset, model_path)
    # model = MLPerceptron(dataset)
    # model = Perceptron(dataset)

    print("Training model...")
    model.train()

    test_on = dataset.dev

    print(f"Predicting on ... ")
    pred = model.predict(test_on)

    # print("Exporting results...")
    file_name = "results_" + datetime.now().strftime("%m-%d-%H:%M:%S") + ".txt"
    model.export_results(test_on, pred, file_name)

    print("Done!")


if __name__ == "__main__":
    best_model_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(best_model_path)

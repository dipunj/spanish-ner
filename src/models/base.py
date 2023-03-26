from abc import ABC, abstractmethod

from tqdm import tqdm
from src.dataset import NERDataset
from src.conlleval import main as final_evaluate


def print_worst_stats(mismatch_stats):
    """
        Print the mismatch stats
    """
    for k, v in mismatch_stats.items():
        max_mismatch_count = max(v.values())
        max_mismatch_label = max(v, key=v.get)

        print(f"{k} was predicted as {max_mismatch_label} {max_mismatch_count} times")


class BaseModel(ABC):
    """
        An abstract class that defines a common interface that all models must implement
    """

    def __init__(self, dataset: NERDataset) -> None:
        super().__init__()
        self.dataset = dataset

    @abstractmethod
    def data_pipeline(self, data):
        """
            Preprocess the data into the format that the model expects
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """
            Train the model on the training set
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """
            Predict on X. X will be an array of sentences, 
            where each sentence is an array of tuples (word, pos, gold)
        """
        raise NotImplementedError

    def export_results(self, X, predictions, results_file_name, print_mismatches=False):
        """
            Compares the gold NER tags from self.dataset.test to predictions 
            and Exports the results_file_name
        """
        print(f"Writing to {results_file_name}")

        j = 0
        labels = ["LOC", "MISC", "ORG", "PER", "O"]

        mismatch_sentences = []
        mismatch_stats = {label: {label: 0 for label in labels}
                          for label in labels}

        with open(results_file_name, "w", encoding="utf-8") as results_out:
            for sent in tqdm(X):

                shouldAppend = False
                word_gold_pred = []

                for (word, _, gold) in sent:
                    pred = predictions[j]
                    j += 1
                    # format is: word gold pred
                    results_out.write(f"{word}\t{gold}\t{pred}\n")
                    word_gold_pred.append((word, gold, pred))

                    if gold != pred:
                        if gold == "O":
                            gold_tail = "O"
                        else:
                            gold_tail = gold[2:]

                        if pred == "O":
                            pred_tail = "O"
                        else:
                            pred_tail = pred[2:]

                        mismatch_stats[gold_tail][pred_tail] += 1
                        shouldAppend = True

                if shouldAppend:
                    mismatch_sentences.append(word_gold_pred)

            results_out.write("\n")

        if print_mismatches:
            for sent in mismatch_sentences:
                for word, gold, pred in sent:
                    if gold != pred:
                        print(f"{word}/{gold}/{pred}")
                    else:
                        print(f"{word}")
                print("\n")

        print(f"Mismatches sentences (total = {len(mismatch_sentences)}):")
        print(f"Mismatch stats: {print_worst_stats(mismatch_stats)}")

        print(f"Evaluating {results_file_name} using conlleval.py")
        final_evaluate(["conlleval.py", results_file_name])

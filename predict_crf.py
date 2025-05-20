import argparse
import pickle
import IOB
from typing import List, Tuple


class CRFFeatures:
    def word2features(self, sent:list, i:int) -> dict:
        """
        Extracts a set of features for a given word in a sentence for CRF tagging.

        Args:
            - sent (list): A list of tuples representing the sentence, where each tuple contains word-level information.
            - i (int): The index of the word in the sentence for which to extract features.

        Returns:
            - dict: A dictionary of features for the word at position i.
        """
        word = sent[i][0]

        features = {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
        }

        if i > 0:
            word1 = sent[i - 1][0]
            features.update(
                {
                    "-1:word.lower()": word1.lower(),
                    "-1:word.istitle()": word1.istitle(),
                    "-1:word.isupper()": word1.isupper(),
                }
            )
        else:
            features["BOS"] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update(
                {
                    "+1:word.lower()": word1.lower(),
                    "+1:word.istitle()": word1.istitle(),
                    "+1:word.isupper()": word1.isupper(),
                }
            )
        else:
            features["EOS"] = True

        return features

    def sent2features(self, sent:list) -> list:
        """
        Converts a sentence into a list of feature dictionaries, one per word.

        Args:
            - sent (list): A list of tuples representing the sentence.

        Returns:
            - list: A list of dictionaries, each containing features for a word in the sentence.
        """
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent:list) -> list:
        """
        Extracts the labels from a sentence.

        Args:
            - sent (list): A list of tuples, where each tuple ends with the label.

        Returns:
            - list: A list of labels corresponding to each word in the sentence.
        """
        return [token[-1] for token in sent]


def parse_args():
    description = ""

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        default="crf.model",
        type=str,
        metavar="FILE",
        help="model file",
    )
    parser.add_argument(
        "dataset", metavar="input file", type=str, help="dataset file (IOB2)"
    )
    return parser.parse_args()


def predict(args) -> List[List[Tuple[str, str]]]:
    """
    Predict the labels for the tokens in the given dataset using a trained CRF model.

    Args:
        - args (argparse.Namespace): Command line arguments containing the model file and dataset file.

    Returns:
        - List[List[Tuple[str, str]]]: A list of sentences, where each sentence is a list of tuples.
    """
    iob = IOB.IOB()

    sentences = [
        [tuple(token) for token in sent] for sent in iob.parse_file(args.dataset)
    ]

    return predictWithSentences(sentences, args.model)


def predictWithSentences(
    sentences: List[List[Tuple[str, str]]], model: str
) -> List[List[Tuple[str, str]]]:
    """
    Predict the labels for the tokens in the given sentences using a trained CRF model.

    Args:
        - sentences (List[List[Tuple[str, str]]]): A list of sentences, where each sentence is a list of tuples.
        - model (str): The path to the trained CRF model file.

    Returns:
        - List[List[Tuple[str, str]]]: A list of sentences, where each sentence is a list of tuples.
    """
    crf = pickle.load(open(model, "rb"))
    feats = CRFFeatures()

    X = [feats.sent2features(s) for s in sentences]
    y_pred = crf.predict(X)

    toret = []

    for i, sentence in enumerate(sentences):
        currentSentence = []
        for j, token in enumerate(sentence):
            if len(token) > 1:
                currentSentence.append((token[0], token[1]))
            else:
                currentSentence.append((token[0], y_pred[i][j]))

        toret.append(currentSentence)

    return toret


if __name__ == "__main__":
    args = parse_args()
    sentences = predict(args)

    print(
        "\n\n".join(
            ["\n".join([f"{token[0]} {token[1]}" for token in s]) for s in sentences]
        ),
        end="\n\n",
    )

import os
import IOB
from predict_crf import CRFFeatures
from typing import List, Tuple


def printPercentages(sentences: List[List[Tuple[str, str]]]) -> None:
    """
    Print the percentage of tokens with and without labels in the given sentences.

    Args:
        - sentences (List[List[Tuple[str, str]]]): A list of sentences, where each sentence is a list of tuples.

    Returns:
        - None
    """
    withLabel = 0
    withoutLabel = 0

    for _, sentence in enumerate(sentences):
        for _, token in enumerate(sentence):
            if len(token) > 1:
                withLabel += 1
            else:
                withoutLabel += 1

    # Print percentages
    print(
        "Percentage with label: {:.2f}%".format(
            withLabel / (withLabel + withoutLabel) * 100
        )
    )
    print(
        "Percentage without label: {:.2f}%".format(
            withoutLabel / (withLabel + withoutLabel) * 100
        )
    )


if __name__ == "__main__":
    currentDir = os.path.dirname(os.path.abspath(__file__))

    originalFile = os.path.join(currentDir, "data", "ner-es.trainOld.csv")

    iob = IOB.IOB()
    feats = CRFFeatures()

    sentences = iob.parse_file(originalFile)

    printPercentages(sentences)

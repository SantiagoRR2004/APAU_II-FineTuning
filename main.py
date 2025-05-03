import os
import IOB
import predict_crf
from types import SimpleNamespace
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


def divideSentences(
    sentences: List[List[Tuple[str, str]]],
) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    """
    Divide the sentences into two parts based on the given percentage.

    Args:
        - sentences (List[List[Tuple[str, str]]]): A list of sentences, where each sentence is a list of tuples.

    Returns:
        - Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]: Two lists of sentences.
    """
    index = 0

    # Calculate the index to split the sentences
    for _, sentence in enumerate(sentences):
        for _, token in enumerate(sentence):
            if len(token) > 1:
                index += 1
            else:
                break

    # Split the sentences into two parts
    sentences1 = []
    sentences2 = []
    for _, sentence in enumerate(sentences):
        if index > 0:
            sentences1.append(sentence)
            index -= len(sentence)
        else:
            sentences2.append(sentence)
    return sentences1, sentences2


def saveFile(sentences: List[List[Tuple[str, str]]], filename: str) -> None:
    """
    Save the sentences to a file.

    Args:
        - sentences (List[List[Tuple[str, str]]]): A list of sentences, where each sentence is a list of tuples.
        - filename (str): The name of the file to save the sentences.

    Returns:
        - None
    """
    with open(filename, "w") as f:
        for sentence in sentences:
            for token in sentence:
                f.write("\t".join(token) + "\n")
            f.write("\n")


if __name__ == "__main__":
    currentDir = os.path.dirname(os.path.abspath(__file__))

    originalFile = os.path.join(currentDir, "data", "ner-es.trainOld.csv")

    iob = IOB.IOB()
    feats = predict_crf.CRFFeatures()

    sentences = iob.parse_file(originalFile)

    printPercentages(sentences)

    # Divide the sentences into two parts
    withLabels, unlabeled = divideSentences(sentences)
    nLabels = len(withLabels)

    saveFile(unlabeled, os.path.join(currentDir, "data", "unlabeled.csv"))

    # We make sure that the labelOptions folder exists
    os.makedirs(os.path.join(currentDir, "data", "labelOptions"), exist_ok=True)

    # Use the CRF model to predict labels for the unlabeled sentences
    cfrSentences = predict_crf.predict(
        SimpleNamespace(
            **{
                "model": os.path.join(currentDir, "crf.es.model"),
                "dataset": originalFile,
            }
        )
    )

    # Save the unlabeled part
    saveFile(
        cfrSentences[nLabels:],
        os.path.join(currentDir, "data", "labelOptions", "cfr.csv"),
    )

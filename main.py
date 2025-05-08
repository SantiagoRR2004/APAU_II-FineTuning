import os
import IOB
from typing import List, Tuple
import labelPropagation


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
                f.write(" ".join(token) + "\n")
            f.write("\n")


if __name__ == "__main__":
    currentDir = os.path.dirname(os.path.abspath(__file__))

    originalFile = os.path.join(currentDir, "data", "ner-es.trainOld.csv")

    iob = IOB.IOB()

    sentences = iob.parse_file(originalFile)

    printPercentages(sentences)

    # Divide the sentences into two parts
    withLabels, unlabeled = divideSentences(sentences)
    nLabels = len(withLabels)

    saveFile(unlabeled, os.path.join(currentDir, "data", "unlabeled.csv"))

    # All label options
    labelPropagation.labelingOptions(sentences, nLabels)

    # Get the consensus labels
    consensus = labelPropagation.getConsensus()

    # Save the fully labeled sentences
    saveFile(
        withLabels + consensus, os.path.join(currentDir, "data", "ner-es.train.csv")
    )

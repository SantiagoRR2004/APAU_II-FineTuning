import os
import predict_crf
import main
from typing import List, Tuple
from spacy.tokens import Doc
import spacy
import tqdm
import IOB
import random
import re


def labelingOptions(sentences: List[List[Tuple[str, str]]], nLabels: int) -> None:
    """
    Generate labeling options for the given sentences and save them to a file.

    All the options will be stored in the labelOptions folder.
    The options will only have the old not labeled sentences.

    Args:
        - sentences (List[List[Tuple[str, str]]]): A list of sentences, where each sentence is a list of tuples.
        - nLabels (int): The number of labeled sentences.

    Returns:
        - None
    """
    currentDir = os.path.dirname(os.path.abspath(__file__))

    # We make sure that the labelOptions folder exists
    os.makedirs(os.path.join(currentDir, "data", "labelOptions"), exist_ok=True)

    labelCRF(sentences, nLabels)
    labelSpacy(sentences, nLabels)


def labelCRF(sentences: List[List[Tuple[str, str]]], nLabels: int) -> None:
    """
    Use the CRF model to predict labels for the unlabeled sentences and save them to a file.

    Args:
        - sentences (List[List[Tuple[str, str]]]): A list of sentences, where each sentence is a list of tuples.
        - nLabels (int): The number of labeled sentences.

    Returns:
        - None
    """
    currentDir = os.path.dirname(os.path.abspath(__file__))

    # Use the CRF model to predict labels for the unlabeled sentences
    cfrSentences = predict_crf.predictWithSentences(
        sentences=sentences, model=os.path.join(currentDir, "crf.es.model")
    )

    # Save the unlabeled part
    main.saveFile(
        cfrSentences[nLabels:],
        os.path.join(currentDir, "data", "labelOptions", "cfr.csv"),
    )


def labelSpacy(sentences: List[List[Tuple[str, str]]], nLabels: int) -> None:
    """
    Use the Spacy model to predict labels for the unlabeled sentences and save them to a file.

    Args:
        - sentences (List[List[Tuple[str, str]]]): A list of sentences, where each sentence is a list of tuples.
        - nLabels (int): The number of labeled sentences.

    Returns:
        - None
    """
    currentDir = os.path.dirname(os.path.abspath(__file__))

    model_name = "es_core_news_lg"

    try:
        nlp = spacy.load(model_name)
    except OSError:
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)

    labeledSentences = []

    for sentence in tqdm.tqdm(
        sentences[nLabels:], desc="Labeling sentences with Spacy"
    ):
        tokens = [token[0] for token in sentence]

        # Create a Doc object from pre-tokenized words
        doc = Doc(nlp.vocab, words=tokens)

        labeledSentences.append(
            [
                (
                    token.text,
                    f"{token.ent_iob_}{'-' if token.ent_type_ else ''}{token.ent_type_}",
                )
                for token in nlp.get_pipe("ner")(doc)
            ]
        )

    # Save the unlabeled part
    main.saveFile(
        labeledSentences, os.path.join(currentDir, "data", "labelOptions", "spacy.csv")
    )


def getConsensus() -> List[List[Tuple[str, str]]]:
    """
    Get the consensus labels from the labeling options.
    The function reads the labeling options from the labelOptions folder
    and checks if all labels are the same.
    If they are not it randomly chooses one of the labels.

    Args:
        - None

    Returns:
        - List[List[Tuple[str, str]]]: A list of sentences, where each sentence is a list of tuples.
    """
    currentDir = os.path.dirname(os.path.abspath(__file__))

    folderPath = os.path.join(currentDir, "data", "labelOptions")
    files = {}

    for file in os.listdir(folderPath):
        if file.endswith(".csv"):
            iob = IOB.IOB()

            files[file[:-4]] = iob.parse_file(os.path.join(folderPath, file))

    emptySentences = [[(t[0], None) for t in s] for s in list(files.values())[0]]

    nAccordance = 0
    nTokens = sum([len(sentence) for sentence in emptySentences])

    for nSentence in range(len(emptySentences)):

        # Returns the number of labels that are equal
        nEqualLabels = sum(
            [
                len(set(f[nSentence][nw][1] for f in files.values())) == 1
                for nw in range(len(emptySentences[nSentence]))
            ]
        )

        nAccordance += nEqualLabels

        if nEqualLabels == len(emptySentences[nSentence]):
            # If all labels are the same, we can fill them with one
            emptySentences[nSentence] = list(files.values())[0][nSentence].copy()

        else:
            # If not, we use the solver
            emptySentences[nSentence] = solveConflicts(
                emptySentences[nSentence],
                [f[nSentence] for f in files.values()],
            )

    print(f"Percentage of unanimous labels: {nAccordance / nTokens * 100:.2f}%")

    return emptySentences


def solveConflicts(
    emptySentece: List[Tuple[str, str]], possibilities: List[List[Tuple[str, str]]]
) -> List[Tuple[str, str]]:
    """
    Solve the conflicts in the labels of the tokens.

    The occupied list needs more explanation:
    When there is a difference we choose to continue
    with one entity until it ends. If the there is
    a difference in the next token, we have to make sure
    that it is not in the middle of another entity.

    Example:
        B-LOC I-LOC O
        B-ORG I-ORG I-ORG

    If we choose the first one, we have:
        B-LOC I-LOC

    Now we can't use the second one,
    because it is in the middle of another entity,
    so we have to block it. Result:
        B-LOC I-LOC O

    Args:
        - emptySentece (List[Tuple[str, str]]): A list of tuples representing the tokens and Nones.
        - possibilities (List[List[Tuple[str, str]]]): A list of lists of tuples representing the possible labels for each token.

    Returns:
        - List[Tuple[str, str]]: A list of tuples representing the tokens and their labels.
    """
    nToken = 0
    occupied = [False] * len(possibilities)
    entityFinalPosition = [-1] * len(possibilities)

    while nToken < len(emptySentece):

        # Check if the entity has finished
        for i in range(len(possibilities)):
            if entityFinalPosition[i] < nToken:
                entityFinalPosition[i] = getEntityEnd(nToken, possibilities[i])
                occupied[i] = False

        tokenOptions = [option[nToken][1] for option in possibilities]

        if len(set(tokenOptions)) == 1:
            # If all are equal, just choose one of them
            emptySentece[nToken] = (emptySentece[nToken][0], tokenOptions[0])
            nToken += 1
        else:
            # If not, we randomly choose one of the lists
            available = [i for i, taken in enumerate(occupied) if not taken]
            nList = random.choice(available)

            # We explore the entire entity
            while entityFinalPosition[nList] >= nToken:
                emptySentece[nToken] = (
                    emptySentece[nToken][0],
                    possibilities[nList][nToken][1],
                )

                for i in range(len(possibilities)):
                    if entityFinalPosition[i] < nToken:
                        entityFinalPosition[i] = getEntityEnd(nToken, possibilities[i])

                nToken += 1

            # We block all that are not nList
            occupied = [True] * len(possibilities)
            occupied[nList] = False

    return emptySentece


def getEntityEnd(position: int, sentence: List[Tuple[str, str]]) -> int:
    """
    Get the end position of the entity in the sentence.

    The index starts at 0.

    Args:
        - position (int): The start position of the entity.
        - sentence (List[Tuple[str, str]]): A list of tuples representing the tokens and their labels.

    Returns:
        - int: The end position of the entity.
    """
    if sentence[position][1] == "O":
        return position
    else:
        # If it is a beggining we put it
        if sentence[position][1].startswith("B-"):
            position += 1

        # We continue until it isn't a I- token
        while (
            position < len(sentence)
            and sentence[position][1] == "I-" + sentence[position - 1][1][2:]
        ):
            position += 1

    return position


def checkWithRegex(
    newLabels: List[List[Tuple[str, str]]], original: List[List[Tuple[str, str]]]
) -> None:
    """
    Check the new labels with the original labels using regex.

    This just prints statistics about the labels.

    Args:
        - newLabels (List[List[Tuple[str, str]]]): A list of sentences with the new labels.
        - original (List[List[Tuple[str, str]]]): A list of sentences with the original labels.

    Returns:
        - None
    """
    originalText = "\n".join(
        [
            " ".join([f"{token[0]} {token[1]}" for token in sentence])
            for sentence in original
        ]
    )

    nSequencesTotal = 0
    nSequencesNotInOriginal = 0
    nSequencesOrgininalSame = 0
    nSequencesOrgininalDiff = 0

    for sentence in newLabels:

        # We try to find sequences
        nToken = 0

        while nToken < len(sentence):

            if sentence[nToken][1] == "O":
                nToken += 1

            else:
                end = getEntityEnd(nToken, sentence)
                nSequencesTotal += 1

                sequence = [
                    re.escape(token[0]) + r"\s+\S+" for token in sentence[nToken:end]
                ]
                pattern = r"\b" + r"\s+".join(sequence) + r"\b"

                # We use re
                regex = re.compile(pattern)
                match = list(regex.finditer(originalText))

                if match:
                    foundSame = False
                    for m in match:
                        matchedText = m.group(0).split()
                        labelsOriginal = matchedText[
                            1::2
                        ]  # Eextract every second element (the labels)

                        # We check if the labels are the same
                        if labelsOriginal == [
                            sentence[i][1] for i in range(nToken, end)
                        ]:
                            nSequencesOrgininalSame += 1
                            foundSame = True
                            break

                    if not foundSame:
                        nSequencesOrgininalDiff += 1

                # We don't find the sequence
                else:
                    nSequencesNotInOriginal += 1

                nToken = end + 1

    print(f"In the propagation {nSequencesTotal} sequences were found:")
    print(
        f"\t{100*nSequencesNotInOriginal/nSequencesTotal:.2f}% are not in the original."
    )
    print(
        f"\t{100*nSequencesOrgininalSame/nSequencesTotal:.2f}% are the same as the original."
    )
    print(
        f"\t{100*nSequencesOrgininalDiff/nSequencesTotal:.2f}% have different labels from the original."
    )

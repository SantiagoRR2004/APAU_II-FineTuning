import os
import predict_crf
import main
from typing import List, Tuple
from spacy.tokens import Doc
import spacy
import tqdm
import IOB
import random


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

    for i in range(len(emptySentences)):
        for j in range(len(emptySentences[i])):

            tokenOptions = [f[i][j][1] for f in files.values()]

            # Check if all labels are the same
            if len(set(tokenOptions)) == 1:
                emptySentences[i][j] = (emptySentences[i][j][0], tokenOptions[0])
                nAccordance += 1
            else:
                # If not, we randomly choose one of the labels
                emptySentences[i][j] = (
                    emptySentences[i][j][0],
                    random.choice(tokenOptions),
                )

    print(f"Percentage of unanimous labels: {nAccordance / nTokens * 100:.2f}%")

    return emptySentences

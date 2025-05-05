import os
import predict_crf
import main
from typing import List, Tuple
import spacy


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

    for sentence in sentences[-nLabels:]:
        doc = nlp(" ".join([token[0] for token in sentence]))
        labeledSentences.append(
            [
                (
                    token.text,
                    f"{token.ent_iob_}{'-' if token.ent_type_ else ''}{token.ent_type_}",
                )
                for token in doc
            ]
        )

    # Save the unlabeled part
    main.saveFile(
        labeledSentences, os.path.join(currentDir, "data", "labelOptions", "spacy.csv")
    )

import os
import predict_crf
import main
from typing import List, Tuple


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

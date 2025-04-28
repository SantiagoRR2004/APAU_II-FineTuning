import os
import IOB
from predict_crf import CRFFeatures

if __name__ == "__main__":
    currentDir = os.path.dirname(os.path.abspath(__file__))

    originalFile = os.path.join(currentDir, "data", "ner-es.trainOld.csv")

    iob = IOB.IOB()
    feats = CRFFeatures()

    sentences = [
        [tuple(token) for token in sent] for sent in iob.parse_file(originalFile)
    ]

    withLabel = 0
    withoutLabel = 0

    for i, sentence in enumerate(sentences):
        for j, token in enumerate(sentence):
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

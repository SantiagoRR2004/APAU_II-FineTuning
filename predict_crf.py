import argparse
import pickle
import IOB


class CRFFeatures:
    def word2features(self, sent, i):
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

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
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


def predict(args):
    iob = IOB.IOB()
    crf = pickle.load(open(args.model, "rb"))
    feats = CRFFeatures()

    sentences = [
        [tuple(token) for token in sent] for sent in iob.parse_file(args.dataset)
    ]

    X = [feats.sent2features(s) for s in sentences]
    y_pred = crf.predict(X)

    for i, sentence in enumerate(sentences):
        for j, token in enumerate(sentence):
            if len(token) > 1:
                print("{} {}".format(token[0], token[1]))
            else:
                print("{} {}".format(token[0], y_pred[i][j]))
        print()


if __name__ == "__main__":
    args = parse_args()
    predict(args)

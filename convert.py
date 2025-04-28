import argparse
import json
import IOB


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IOB format to JSON.")
    parser.add_argument("iobfile", help="Input file in IOB format")
    parser.add_argument("jsonfile", help="Output file in JSON format")
    return parser.parse_args()


def convert_to_json(ifile, ofile):
    iob = IOB.IOB()
    sentences = iob.parse_file(ifile)

    jsonl = [
        {
            "tokens": [token[0] for token in sentence],
            "labels": [token[1] for token in sentence],
        }
        for sentence in sentences
    ]

    with open(ofile, "w", encoding="utf-8") as f:
        for sentence in jsonl:
            f.write(json.dumps(sentence, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()
    convert_to_json(args.iobfile, args.jsonfile)

import argparse
import json
import IOB


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IOB format to JSONL.")
    parser.add_argument("iobfile", help="Input file in IOB format")
    parser.add_argument("jsonfile", help="Output file in JSONL format")
    return parser.parse_args()


def convert_to_json(ifile, ofile):
    iob = IOB.IOB()
    sentences = iob.parse_file(ifile)

    jsonl = []
    errors = 0

    for idx, sentence in enumerate(sentences):
        tokens = []
        labels = []
        for token in sentence:
            if len(token) == 2:
                tokens.append(token[0])
                labels.append(token[1])
            else:
                print(f"Error en la frase {idx + 1}: token mal formado {token}")
                errors += 1
        if tokens:  # Solo guardamos frases no vacías
            jsonl.append({"tokens": tokens, "labels": labels})

    with open(ofile, "w", encoding="utf-8") as f:
        for sentence in jsonl:
            f.write(json.dumps(sentence, ensure_ascii=False) + "\n")

    print(f"\nConversión terminada. Se detectaron {errors} errores de formato.\n")


if __name__ == "__main__":
    args = parse_args()
    convert_to_json(args.iobfile, args.jsonfile)

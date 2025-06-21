import argparse
import chardet
import json
import IOB
import sys


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


def checkEncoding(path: str) -> None:
    """
    Detects the encoding of a file using chardet.

    Args:
        - path (str): The path to the file to check.

    Returns:
        - None
    """
    with open(path, "rb") as f:
        raw_data = f.read(10000)  # lee los primeros 10KB
        result = chardet.detect(raw_data)

    print(
        f"Codificación detectada: {result['encoding']} con {result['confidence']*100:.2f}% de confianza"
    )


def checkUTF8(path: str) -> None:
    """
    Cheks if a file can be decoded as UTF-8 without errors.

    Args:
        - path (str): The path to the file to check.

    Returns:
        - None
    """
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, 1):
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as e:
                print(
                    f"Error decodificando UTF-8 en línea {lineno}, byte {e.start}: {e}"
                )
                sys.exit(1)
    print("¡Todo decodifica bien en UTF-8!")


if __name__ == "__main__":
    args = parse_args()
    convert_to_json(args.iobfile, args.jsonfile)

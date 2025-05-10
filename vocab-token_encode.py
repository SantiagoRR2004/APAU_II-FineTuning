import io

for path in [
    "models/roberta-base-bne-ner/vocab.json",
    "models/roberta-base-bne-ner/tokenizer.json"
]:
    # Leer como binario y decodificar como utf-8
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")

    # Sobrescribir en UTF-8
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)



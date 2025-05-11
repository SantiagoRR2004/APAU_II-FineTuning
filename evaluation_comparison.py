import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import (
    precision_score, recall_score, f1_score,
    classification_report as seqeval_classification_report
)

from predict_crf import predictWithSentences, CRFFeatures
import IOB


# --- Configuración ---
csv_path = "data/ner-es.validOld.csv"
crf_model_path = "crf.es.model"
save_report_path = "evaluation_report.txt"
hf_model_dir = "models/roberta-base-bne-ner"


# --- Leer dataset CSV IOB2 ---
def load_iob_csv(path):
    iob = IOB.IOB()
    raw_sentences = iob.parse_file(path)

    tokenized = [[(tok[0], tok[1]) if len(tok) > 1 else (tok[0], "O") for tok in sent] for sent in raw_sentences]
    tokens = [[tok for tok, _ in sent] for sent in tokenized]
    labels = [[label for _, label in sent] for sent in tokenized]
    return tokenized, tokens, labels


tokenized_sentences, tokens_list, true_labels = load_iob_csv(csv_path)

# --- CRF Predictions ---
crf_preds_tuples = predictWithSentences(tokenized_sentences, crf_model_path)
pred_labels_crf = [[label for _, label in sent] for sent in crf_preds_tuples]

# --- Hugging Face fine-tuned model ---
model = AutoModelForTokenClassification.from_pretrained(hf_model_dir)
tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# --- Hugging Face predictions ---
pred_labels_ft = []
for tokens in tokens_list:
    text = " ".join(tokens)
    preds = ner_pipeline(text)
    sentence_labels = ["O"] * len(tokens)

    for ent in preds:
        word = ent["word"]
        label = ent["entity_group"]
        iob_label = "B-" + label if label != "O" else "O"
        for i, token in enumerate(tokens):
            if sentence_labels[i] == "O" and word.lower() in token.lower():
                sentence_labels[i] = iob_label
                break

    pred_labels_ft.append(sentence_labels)

# --- Métricas globales ---
metrics = {
    "Fine-tuned": {
        "f1": f1_score(true_labels, pred_labels_ft),
        "precision": precision_score(true_labels, pred_labels_ft),
        "recall": recall_score(true_labels, pred_labels_ft),
    },
    "CRF": {
        "f1": f1_score(true_labels, pred_labels_crf),
        "precision": precision_score(true_labels, pred_labels_crf),
        "recall": recall_score(true_labels, pred_labels_crf),
    },
}

# --- Informes detallados ---
report_ft = seqeval_classification_report(true_labels, pred_labels_ft)
report_crf = seqeval_classification_report(true_labels, pred_labels_crf)

print("=== Classification Report: Fine-tuned ===")
print(report_ft)
print("=== Classification Report: CRF ===")
print(report_crf)

with open(save_report_path, "w", encoding="utf-8") as f:
    f.write("=== Classification Report: Fine-tuned ===\n")
    f.write(report_ft + "\n")
    f.write("=== Classification Report: CRF ===\n")
    f.write(report_crf + "\n")

# --- Gráfica de comparación ---
labels = ["Precision", "Recall", "F1"]
ft_scores = [metrics["Fine-tuned"][m.lower()] for m in labels]
crf_scores = [metrics["CRF"][m.lower()] for m in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width / 2, ft_scores, width, label="Fine-tuned")
ax.bar(x + width / 2, crf_scores, width, label="CRF")

ax.set_ylabel("Score")
ax.set_title("Comparison of NER Models on Validation Set")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()

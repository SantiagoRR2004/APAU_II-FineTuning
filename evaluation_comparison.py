import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report as seqeval_classification_report,
)
from predict_crf import predictWithSentences


# --- Cargar datos desde JSONL ---
def load_jsonl(path):
    tokens_list, labels_list = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens_list.append(data["tokens"])
            labels_list.append(data["labels"])
    return tokens_list, labels_list


def makeReport() -> None:
    """
    Generate and show a report comparing the performance of a fine-tuned model
    and a CRF model on a validation set for Named Entity Recognition (NER).

    Args:
        - None

    Returns:
        - None
    """
    # --- Configuración ---
    jsonl_path = "data/ner-es.valid.json"
    crf_model_path = "crf.es.model"
    hf_model_dir = "models/roberta-base-bne-ner"
    save_report_path = "evaluation_report.txt"

    tokens_list, true_labels = load_jsonl(jsonl_path)

    # --- CRF Predictions ---
    crf_input = [[(token,) for token in sentence] for sentence in tokens_list]
    crf_preds_tuples = predictWithSentences(crf_input, crf_model_path)
    pred_labels_crf = [[label for _, label in sent] for sent in crf_preds_tuples]

    # --- Hugging Face Model y Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
    model = AutoModelForTokenClassification.from_pretrained(hf_model_dir)
    model.eval()

    # --- Hugging Face predictions ---
    pred_labels_ft = []

    for tokens in tokens_list:
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,  # necesario para word_ids()
        )

        # Solo pasar input_ids y attention_mask (no offset_mapping)
        inputs = {
            k: v
            for k, v in encoding.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)
        word_ids = encoding.word_ids()

        sentence_labels = ["O"] * len(tokens)
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                pred_label_id = predictions[0][idx].item()
                label = model.config.id2label[pred_label_id]
                if label != "O" and sentence_labels[word_idx] == "O":
                    sentence_labels[word_idx] = label

        pred_labels_ft.append(sentence_labels)

    # --- Evaluación ---
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


if __name__ == "__main__":
    makeReport()

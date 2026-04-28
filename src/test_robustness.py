from pathlib import Path
import pickle
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from sklearn.model_selection import train_test_split

from model import PhishingNet
from utils import (
    evaluate_model,
    get_model_size_mb,
    save_metrics_csv,
    plot_confusion_matrix
)


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "preprocessed_emails.pkl"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_fp32.pth"
PTQ_MODEL_PATH = MODELS_DIR / "ptq_dynamic.pth"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

METRICS_PATH = RESULTS_DIR / "metrics_summary.csv"

MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_CLASSES = 2

def add_typo_to_word(word):
    """
    Adds a tiny typo by swapping two nearby letters.
    Example: account -> acocunt
    """
    if len(word) < 5:
        return word

    i = random.randint(0, len(word) - 2)
    letters = list(word)
    letters[i], letters[i + 1] = letters[i + 1], letters[i]
    return "".join(letters)

def remove_random_word(words):
    if len(words) > 5:
        idx = random.randint(0, len(words)-1)
        words.pop(idx)
    return words

def random_case(text):
    return "".join(
        c.upper() if random.random() < 0.3 else c.lower()
        for c in text
    )

def skew_email_text(text, typo_rate=0.2):
    words = text.split()
    new_words = []

    for word in words:
        if random.random() < typo_rate:
            new_words.append(add_typo_to_word(word))
        else:
            new_words.append(word)

    # randomly drop one word
    if random.random() < 0.3:
        new_words = remove_random_word(new_words)

    return " ".join(new_words)


def load_raw_test_data():
    """
    Loads original processed text and creates the same train/test split
    used in training.
    """
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X_raw = np.array(data["processed_text"])
    y = np.array(data["label"])

    _, X_test_raw, _, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_test_raw, y_test


def make_skewed_test_loader():
    """
    Creates a test loader from typo-skewed email text.
    Uses the saved TF-IDF vectorizer from baseline training.
    """
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_test_raw, y_test = load_raw_test_data()

    skewed_texts = [skew_email_text(text, typo_rate=0.20) for text in X_test_raw]

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    X_skewed = vectorizer.transform(skewed_texts).toarray()

    X_skewed = torch.tensor(X_skewed, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(X_skewed, y_test)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return loader, skewed_texts


def load_fp32_model(input_size, device):
    model = PhishingNet(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))
    model.eval()

    return model


def load_ptq_model(input_size, device):
    fp32_model = PhishingNet(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    ).to(device)

    fp32_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))
    fp32_model.eval()

    ptq_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8
    )

    ptq_model.eval()
    return ptq_model


def evaluate_and_save(model, model_version, model_size_path, test_loader, device, figure_name):
    metrics = evaluate_model(model, test_loader, device)

    metrics["model_version"] = model_version
    metrics["model_size_mb"] = get_model_size_mb(model_size_path)

    save_metrics_csv(metrics, METRICS_PATH)

    plot_confusion_matrix(
        metrics["labels"],
        metrics["predictions"],
        FIGURES_DIR / figure_name,
        title=f"{model_version} Confusion Matrix"
    )

    print(f"\n{model_version} Results on Skewed Data")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Model Size: {metrics['model_size_mb']:.4f} MB")
    print(f"Avg Inference Time/Sample: {metrics['avg_inference_time_per_sample']:.8f} seconds")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Could not find tfidf_vectorizer.pkl. Run train_baseline.py first."
        )

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Could not find baseline_fp32.pth. Run train_baseline.py first."
        )

    if not PTQ_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Could not find ptq_dynamic.pth. Run quantize_ptq.py first."
        )

    test_loader, skewed_texts = make_skewed_test_loader()

    input_size = MAX_FEATURES

    fp32_model = load_fp32_model(input_size, device)
    ptq_model = load_ptq_model(input_size, device)

    evaluate_and_save(
        model=fp32_model,
        model_version="FP32 Baseline Skewed",
        model_size_path=BASELINE_MODEL_PATH,
        test_loader=test_loader,
        device=device,
        figure_name="fp32_skewed_confusion_matrix.png"
    )

    evaluate_and_save(
        model=ptq_model,
        model_version="PTQ Dynamic Skewed",
        model_size_path=PTQ_MODEL_PATH,
        test_loader=test_loader,
        device=device,
        figure_name="ptq_skewed_confusion_matrix.png"
    )

    sample_path = RESULTS_DIR / "skewed_text_examples.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(skewed_texts[:10]):
            f.write(f"Example {i + 1}:\n{text}\n\n")

    print(f"\nSaved skewed text examples to: {sample_path}")
    print(f"Saved skewed confusion matrices to: {FIGURES_DIR}")
    print(f"Updated metrics table at: {METRICS_PATH}")


if __name__ == "__main__":
    main()
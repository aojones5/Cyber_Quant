from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from data_utils import load_data
from model import PhishingNet
from utils import (
    evaluate_model,
    get_model_size_mb,
    save_metrics_csv,
    plot_loss_curve,
    plot_confusion_matrix
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "preprocessed_emails.pkl"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_fp32.pth"
METRICS_PATH = RESULTS_DIR / "metrics_summary.csv"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    data = load_data(
        data_path=str(DATA_PATH),
        max_features=5000,
        test_size=0.2,
        random_state=42,
        batch_size=32
    )

    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    input_size = data["input_size"]

    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(data["vectorizer"], f)
    
    print(f"Saved TF-IDF vectorizer to: {vectorizer_path}")

    model = PhishingNet(
        input_size=input_size,
        hidden_size=128,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    num_epochs = 20
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), BASELINE_MODEL_PATH)
    print(f"Saved baseline model to: {BASELINE_MODEL_PATH}")

    metrics = evaluate_model(model, test_loader, device)
    metrics["model_version"] = "FP32 Baseline"
    metrics["model_size_mb"] = get_model_size_mb(BASELINE_MODEL_PATH)

    save_metrics_csv(metrics, METRICS_PATH)

    plot_loss_curve(
        epoch_losses,
        FIGURES_DIR / "baseline_loss_curve.png",
        title="FP32 Baseline Training Loss"
    )

    plot_confusion_matrix(
        metrics["labels"],
        metrics["predictions"],
        FIGURES_DIR / "baseline_confusion_matrix.png",
        title="FP32 Baseline Confusion Matrix"
    )

    print("\nBaseline Results")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Model Size: {metrics['model_size_mb']:.4f} MB")
    print(f"Avg Inference Time/Sample: {metrics['avg_inference_time_per_sample']:.8f} seconds")

    print(f"\nSaved results to: {METRICS_PATH}")
    print(f"Saved visuals to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
#uses dynamic PTQ 
from pathlib import Path
import torch
import torch.nn as nn

from data_utils import load_data
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
METRICS_PATH = RESULTS_DIR / "metrics_summary.csv"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the exact same data setup as baseline
    data = load_data(
        data_path=str(DATA_PATH),
        max_features=5000,
        test_size=0.2,
        random_state=42,
        batch_size=32
    )

    test_loader = data["test_loader"]
    input_size = data["input_size"]

    # Recreate the same FP32 model architecture
    fp32_model = PhishingNet(
        input_size=input_size,
        hidden_size=128,
        num_classes=2
    ).to(device)

    # Load trained baseline weights
    fp32_model.load_state_dict(
        torch.load(BASELINE_MODEL_PATH, map_location=device)
    )

    fp32_model.eval()

    # Apply dynamic post-training quantization to Linear layers
    ptq_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8
    )

    ptq_model.eval()

    # Save PTQ model
    torch.save(ptq_model.state_dict(), PTQ_MODEL_PATH)
    print(f"Saved PTQ model to: {PTQ_MODEL_PATH}")

    # Evaluate PTQ model using same evaluation function
    metrics = evaluate_model(ptq_model, test_loader, device)
    metrics["model_version"] = "PTQ Dynamic"
    metrics["model_size_mb"] = get_model_size_mb(PTQ_MODEL_PATH)

    # Save row to same metrics table
    save_metrics_csv(metrics, METRICS_PATH)

    # Save PTQ confusion matrix
    plot_confusion_matrix(
        metrics["labels"],
        metrics["predictions"],
        FIGURES_DIR / "ptq_confusion_matrix.png",
        title="PTQ Dynamic Confusion Matrix"
    )

    print("\nPTQ Results")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Model Size: {metrics['model_size_mb']:.4f} MB")
    print(f"Avg Inference Time/Sample: {metrics['avg_inference_time_per_sample']:.8f} seconds")

    print(f"\nSaved PTQ results to: {METRICS_PATH}")
    print(f"Saved PTQ visual to: {FIGURES_DIR / 'ptq_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
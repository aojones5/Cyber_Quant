#
import os
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def evaluate_model(model, test_loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_sample = total_time / len(all_labels)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_inference_time": total_time,
        "avg_inference_time_per_sample": avg_time_per_sample,
        "labels": all_labels,
        "predictions": all_preds
    }


def save_metrics_csv(metrics, output_path):
    file_exists = os.path.exists(output_path)

    fieldnames = [
        "model_version",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "model_size_mb",
        "total_inference_time",
        "avg_inference_time_per_sample"
    ]

    with open(output_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "model_version": metrics["model_version"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "model_size_mb": metrics["model_size_mb"],
            "total_inference_time": metrics["total_inference_time"],
            "avg_inference_time_per_sample": metrics["avg_inference_time_per_sample"]
        })


def plot_loss_curve(losses, output_path, title="Training Loss"):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(labels, predictions, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(labels, predictions)

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Ham", "Phishing"])
    plt.yticks([0, 1], ["Ham", "Phishing"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
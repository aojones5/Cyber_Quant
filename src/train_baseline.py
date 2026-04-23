import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import load_data
from model import PhishingNet


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    accuracy = 100 * correct / total
    inference_time = end_time - start_time

    return accuracy, inference_time


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cpu")  # keep CPU for fair quantization comparison
    print(f"Using device: {device}")

    data = load_data(
        data_path="data/preprocessed_emails.pkl",
        max_features=5000,
        test_size=0.2,
        random_state=42,
        batch_size=32
    )

    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    input_size = data["input_size"]

    model = PhishingNet(input_size=input_size, hidden_size=128, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 20

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
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    accuracy, inference_time = evaluate_model(model, test_loader, device)

    print(f"\nBaseline Test Accuracy: {accuracy:.2f}%")
    print(f"Baseline Inference Time: {inference_time:.4f} seconds")

    # Save model
    model_path = "models/baseline_fp32.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved baseline model to: {model_path}")

    # Save basic results text file
    results_path = "results/baseline_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Baseline Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Baseline Inference Time: {inference_time:.4f} seconds\n")

    print(f"Saved baseline results to: {results_path}")


if __name__ == "__main__":
    main()
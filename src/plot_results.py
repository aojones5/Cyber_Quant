from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics_summary.csv"

FIGURES_DIR.mkdir(exist_ok=True)


def plot_bar(df, metric, title, ylabel, filename, zoom=False):
    plt.figure(figsize=(10, 6))
    plt.bar(df["model_version"], df[metric])

    plt.title(title)
    plt.xlabel("Model Version")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")

    if zoom:
        min_val = df[metric].min()
        max_val = df[metric].max()
        padding = (max_val - min_val) * 0.25

        if padding == 0:
            padding = 0.01

        plt.ylim(min_val - padding, max_val + padding)

    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Could not find {METRICS_PATH}")

    df = pd.read_csv(METRICS_PATH)

    plot_bar(
        df,
        "accuracy",
        "Accuracy Comparison Across Model Versions",
        "Accuracy (%)",
        "accuracy_comparison_zoomed.png",
        zoom=True
    )

    plot_bar(
        df,
        "f1_score",
        "F1 Score Comparison Across Model Versions",
        "F1 Score",
        "f1_score_comparison_zoomed.png",
        zoom=True
    )

    plot_bar(
        df,
        "model_size_mb",
        "Model Size Comparison Across Model Versions",
        "Model Size (MB)",
        "model_size_comparison.png"
    )

    plot_bar(
        df,
        "avg_inference_time_per_sample",
        "Average Inference Time Per Sample",
        "Seconds Per Sample",
        "inference_time_comparison.png"
    )


if __name__ == "__main__":
    main()
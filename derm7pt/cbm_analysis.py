"""Create test-set plots and concept-intervention results for trained CBMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from derm7pt.cbm import (
    CONCEPT_COLUMNS,
    CONCEPT_LABELS,
    DIAGNOSIS_LABELS,
    _classification_metrics,
    load_data,
    make_dataset,
)


REGIMES = ("independent", "sequential", "joint")


def _predictions(model: tf.keras.Model, data, image_size: int, batch_size: int):
    images = make_dataset(data, "test", image_size, batch_size)
    values = model.predict(images, verbose=0)
    return values if isinstance(values, dict) else dict(zip(model.output_names, values))


def _concept_matrix(predictions: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [predictions[f"concept_{name}"] for name in CONCEPT_COLUMNS], axis=1
    )


def _annotated_matrix(data, indexes: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            tf.one_hot(data.concepts[name][indexes], len(CONCEPT_LABELS[name])).numpy()
            for name in CONCEPT_COLUMNS
        ],
        axis=1,
    )


def _intervene(
    model: tf.keras.Model,
    predicted: np.ndarray,
    annotated: np.ndarray,
    target: np.ndarray,
) -> dict:
    diagnosis_layer = model.get_layer("diagnosis")
    baseline = diagnosis_layer(predicted, training=False).numpy()
    base_labels = baseline.argmax(axis=1)
    base_correct = base_labels == target
    width = 0
    results = {}

    start = 0
    for name in CONCEPT_COLUMNS:
        width = len(CONCEPT_LABELS[name])
        changed = predicted.copy()
        changed[:, start:start + width] = annotated[:, start:start + width]
        start += width
        results[name] = _intervention_metrics(
            diagnosis_layer(changed, training=False).numpy(), target, base_correct
        )

    results["all_concepts"] = _intervention_metrics(
        diagnosis_layer(annotated, training=False).numpy(), target, base_correct
    )
    results["baseline"] = _intervention_metrics(baseline, target, base_correct)
    return results


def _intervention_metrics(
    probabilities: np.ndarray, target: np.ndarray, base_correct: np.ndarray
) -> dict:
    predicted = probabilities.argmax(axis=1)
    correct = predicted == target
    metrics = _classification_metrics(target, predicted)
    metrics["fixed_errors"] = int((~base_correct & correct).sum())
    metrics["new_errors"] = int((base_correct & ~correct).sum())
    return metrics


def _add_deltas(results: dict) -> None:
    for regime_results in results.values():
        baseline = regime_results["baseline"]["macro_f1"]
        baseline_accuracy = regime_results["baseline"]["accuracy"]
        for name, metrics in regime_results.items():
            if name == "concepts":
                continue
            metrics["delta_macro_f1"] = metrics["macro_f1"] - baseline
            metrics["delta_accuracy"] = metrics["accuracy"] - baseline_accuracy


def _save_plots(results: dict, output_dir: Path) -> None:
    regimes = list(results)
    colors = ("#4C78A8", "#F58518")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in zip(
        axes, ("macro_f1", "balanced_accuracy"), ("Macro F1", "Balanced accuracy")
    ):
        x = np.arange(len(regimes))
        width = 0.36
        baseline = [results[r]["baseline"][metric] for r in regimes]
        corrected = [results[r]["all_concepts"][metric] for r in regimes]
        axis.bar(x - width / 2, baseline, width, label="Predicted concepts", color=colors[0])
        axis.bar(x + width / 2, corrected, width, label="All concepts corrected", color=colors[1])
        axis.set_xticks(x, regimes, rotation=15)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Test diagnosis performance")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnosis_performance.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(regimes))
    width = 0.36
    predicted_accuracy = [results[r]["baseline"]["accuracy"] for r in regimes]
    corrected_accuracy = [results[r]["all_concepts"]["accuracy"] for r in regimes]
    supports = np.asarray(
        results[regimes[0]]["baseline"]["confusion_matrix"]
    ).sum(axis=1)
    majority_accuracy = float(supports.max() / supports.sum())

    fig, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(
        x - width / 2,
        predicted_accuracy,
        width,
        label="Predicted concepts",
        color=colors[0],
    )
    corrected_bars = axis.bar(
        x + width / 2,
        corrected_accuracy,
        width,
        label="All concepts corrected",
        color=colors[1],
    )
    axis.axhline(
        majority_accuracy,
        color="#555555",
        linestyle="--",
        label=f"Always predict majority class ({majority_accuracy:.1%})",
    )
    axis.bar_label(bars, fmt="%.1f%%", padding=3, labels=[f"{v:.1%}" for v in predicted_accuracy])
    axis.bar_label(
        corrected_bars,
        padding=3,
        labels=[f"{v:.1%}" for v in corrected_accuracy],
    )
    axis.set_xticks(x, regimes)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Test diagnosis accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "test_accuracy.png", dpi=180)
    plt.close(fig)

    concept_scores = np.array([
        [results[r]["concepts"][name]["macro_f1"] for name in CONCEPT_COLUMNS]
        for r in regimes
    ])
    fig, axis = plt.subplots(figsize=(11, 3.8))
    image = axis.imshow(concept_scores, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(CONCEPT_COLUMNS)), CONCEPT_COLUMNS, rotation=35, ha="right")
    axis.set_yticks(range(len(regimes)), regimes)
    for row in range(len(regimes)):
        for col in range(len(CONCEPT_COLUMNS)):
            axis.text(col, row, f"{concept_scores[row, col]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Macro F1")
    axis.set_title("Test concept prediction quality")
    fig.tight_layout()
    fig.savefig(output_dir / "concept_performance.png", dpi=180)
    plt.close(fig)

    interventions = (*CONCEPT_COLUMNS, "all_concepts")
    deltas = np.array([
        [results[r][name]["delta_macro_f1"] for name in interventions]
        for r in regimes
    ])
    bound = max(float(np.max(np.abs(deltas))), 0.01)
    fig, axis = plt.subplots(figsize=(11, 3.8))
    image = axis.imshow(
        deltas, vmin=-bound, vmax=bound, cmap="coolwarm", aspect="auto"
    )
    axis.set_xticks(range(len(interventions)), interventions, rotation=35, ha="right")
    axis.set_yticks(range(len(regimes)), regimes)
    for row in range(len(regimes)):
        for col in range(len(interventions)):
            axis.text(col, row, f"{deltas[row, col]:+.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Change in test macro F1")
    axis.set_title("Effect of replacing predicted concepts with annotations")
    fig.tight_layout()
    fig.savefig(output_dir / "concept_interventions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(regimes), figsize=(12, 4))
    for axis, regime in zip(axes, regimes):
        matrix = np.asarray(
            results[regime]["baseline"]["confusion_matrix"], dtype=np.int64
        )
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(regime)
        axis.set_xticks(range(len(DIAGNOSIS_LABELS)), DIAGNOSIS_LABELS, rotation=45)
        axis.set_yticks(range(len(DIAGNOSIS_LABELS)), DIAGNOSIS_LABELS)
        axis.set_xlabel("Predicted")
        for row in range(len(DIAGNOSIS_LABELS)):
            for col in range(len(DIAGNOSIS_LABELS)):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    axes[0].set_ylabel("Actual")
    fig.suptitle("Test diagnosis confusion matrices")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnosis_confusion_matrices.png", dpi=180)
    plt.close(fig)


def analyze(data_dir: Path, model_dir: Path, batch_size: int) -> dict:
    data = load_data(data_dir)
    indexes = data.splits["test"]
    output_dir = model_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    case_rows = [
        {
            "metadata_index": int(index),
            "actual_diagnosis": DIAGNOSIS_LABELS[data.diagnosis[index]],
        }
        for index in indexes
    ]

    for regime in REGIMES:
        model_path = model_dir / f"{regime}.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        image_size = int(model.inputs[0].shape[1])
        predictions = _predictions(model, data, image_size, batch_size)
        concepts = _concept_matrix(predictions)
        annotated = _annotated_matrix(data, indexes)
        diagnosis = predictions["diagnosis"].argmax(axis=1)
        concept_scores = {
            name: _classification_metrics(
                data.concepts[name][indexes],
                predictions[f"concept_{name}"].argmax(axis=1),
                CONCEPT_LABELS[name],
            )
            for name in CONCEPT_COLUMNS
        }
        interventions = _intervene(
            model, concepts, annotated, data.diagnosis[indexes]
        )
        interventions["concepts"] = concept_scores
        report[regime] = interventions

        for row, original_index in enumerate(indexes):
            case = {
                f"{regime}_predicted_diagnosis": DIAGNOSIS_LABELS[diagnosis[row]],
                f"{regime}_diagnosis_confidence": float(
                    predictions["diagnosis"][row, diagnosis[row]]
                ),
            }
            for name in CONCEPT_COLUMNS:
                concept_id = predictions[f"concept_{name}"][row].argmax()
                case[f"{regime}_{name}_predicted"] = CONCEPT_LABELS[name][concept_id]
                true_id = data.concepts[name][original_index]
                case[f"{regime}_{name}_actual"] = CONCEPT_LABELS[name][true_id]
            case_rows[row].update(case)

    _add_deltas(report)
    _save_plots(report, output_dir)
    (output_dir / "interventions.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(case_rows).to_csv(output_dir / "case_explanations.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "release_v0")
    parser.add_argument(
        "--model-dir", type=Path, default=root / "derm7pt" / "outputs" / "cbm"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = analyze(args.data_dir, args.model_dir, args.batch_size)
    for regime in REGIMES:
        baseline = report[regime]["baseline"]
        corrected = report[regime]["all_concepts"]
        print(
            f"{regime}: test macro-F1 {baseline['macro_f1']:.3f} "
            f"-> {corrected['macro_f1']:.3f} after correcting all concepts"
        )
    print(f"Plots and detailed results saved under {args.model_dir / 'analysis'}")


if __name__ == "__main__":
    main()

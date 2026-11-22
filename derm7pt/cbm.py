"""Train and assess a simple seven-concept bottleneck model for Derm7Pt."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


CONCEPTS = {
    "pigment_network": {
        "absent": "absent", "typical": "typical", "atypical": "atypical"
    },
    "streaks": {"absent": "absent", "regular": "regular", "irregular": "irregular"},
    "pigmentation": {
        "absent": "absent",
        "diffuse regular": "regular", "localized regular": "regular",
        "diffuse irregular": "irregular", "localized irregular": "irregular",
    },
    "regression_structures": {
        "absent": "absent", "blue areas": "present",
        "white areas": "present", "combinations": "present",
    },
    "dots_and_globules": {
        "absent": "absent", "regular": "regular", "irregular": "irregular"
    },
    "blue_whitish_veil": {"absent": "absent", "present": "present"},
    "vascular_structures": {
        "absent": "absent",
        "arborizing": "regular", "comma": "regular", "hairpin": "regular",
        "within regression": "regular", "wreath": "regular",
        "dotted": "irregular", "linear irregular": "irregular",
    },
}
CONCEPT_COLUMNS = tuple(CONCEPTS)
CONCEPT_LABELS = {
    name: tuple(dict.fromkeys(groups.values())) for name, groups in CONCEPTS.items()
}
DIAGNOSIS_LABELS = ("BCC", "NEV", "MEL", "MISC", "SK")
DIAGNOSES = {
    "basal cell carcinoma": "BCC",
    "blue nevus": "NEV", "clark nevus": "NEV", "combined nevus": "NEV",
    "congenital nevus": "NEV", "dermal nevus": "NEV", "recurrent nevus": "NEV",
    "reed or spitz nevus": "NEV",
    "melanoma": "MEL", "melanoma (in situ)": "MEL",
    "melanoma (less than 0.76 mm)": "MEL",
    "melanoma (0.76 to 1.5 mm)": "MEL",
    "melanoma (more than 1.5 mm)": "MEL", "melanoma metastasis": "MEL",
    "dermatofibroma": "MISC", "lentigo": "MISC", "melanosis": "MISC",
    "miscellaneous": "MISC", "vascular lesion": "MISC",
    "seborrheic keratosis": "SK",
}
DIAGNOSIS_IDS = {label: index for index, label in enumerate(DIAGNOSIS_LABELS)}
CONCEPT_IDS = {
    name: {label: index for index, label in enumerate(labels)}
    for name, labels in CONCEPT_LABELS.items()
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "release_v0"
DEFAULT_OUTPUT_DIR = ROOT / "derm7pt" / "outputs" / "cbm"
CONCEPT_OUTPUTS = tuple(f"concept_{name}" for name in CONCEPT_COLUMNS)


@dataclass
class Derm7Pt:
    paths: list[str]
    diagnosis: np.ndarray
    concepts: dict[str, np.ndarray]
    splits: dict[str, np.ndarray]


def _encode(values: pd.Series, groups: dict[str, str], ids: dict[str, int]) -> np.ndarray:
    labels = values.map(groups)
    if labels.isna().any():
        raise ValueError(f"Unknown labels: {values[labels.isna()].dropna().unique().tolist()}")
    return labels.map(ids).to_numpy(dtype=np.int32)


def _check_splits(splits: dict[str, np.ndarray], size: int) -> None:
    indexes = np.concatenate([splits[name] for name in ("train", "valid", "test")])
    if (indexes < 0).any() or (indexes >= size).any():
        raise ValueError("A split contains an out-of-range row index.")
    if len(np.unique(indexes)) != size or not np.array_equal(
        np.sort(indexes), np.arange(size)
    ):
        raise ValueError("The train/valid/test splits must be disjoint and cover all rows.")


def load_data(data_dir: Path) -> Derm7Pt:
    metadata_file = data_dir / "meta" / "meta.csv"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    frame = pd.read_csv(metadata_file)

    diagnosis = _encode(frame["diagnosis"], DIAGNOSES, DIAGNOSIS_IDS)
    concepts = {
        name: _encode(frame[name], groups, CONCEPT_IDS[name])
        for name, groups in CONCEPTS.items()
    }
    paths = [str(data_dir / "images" / str(path)) for path in frame["derm"]]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dermoscopic image(s), e.g. {missing[:3]}")

    splits = {}
    for name in ("train", "valid", "test"):
        split_file = data_dir / "meta" / f"{name}_indexes.csv"
        splits[name] = pd.read_csv(split_file)["indexes"].to_numpy(dtype=np.int64)
    _check_splits(splits, len(frame))
    return Derm7Pt(paths, diagnosis, concepts, splits)


def make_dataset(
    data: Derm7Pt,
    split: str,
    image_size: int,
    batch_size: int,
    target_names: tuple[str, ...] | None = None,
    shuffle: bool = False,
) -> tf.data.Dataset:
    indexes = data.splits[split]
    paths = np.asarray(data.paths)[indexes]
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    if target_names is not None:
        targets = {}
        if "diagnosis" in target_names:
            targets["diagnosis"] = data.diagnosis[indexes]
        for name in CONCEPT_COLUMNS:
            output = f"concept_{name}"
            if output in target_names:
                targets[output] = data.concepts[name][indexes]
        dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

    def decode(path: tf.Tensor) -> tf.Tensor:
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        return tf.cast(tf.image.resize(image, (image_size, image_size)), tf.float32) / 255

    if target_names is None:
        dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(
            lambda path, labels: (decode(path), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if shuffle:
        dataset = dataset.shuffle(len(indexes), seed=42, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_concept_model(image_size: int) -> tf.keras.Model:
    image = tf.keras.Input((image_size, image_size, 3), name="image")
    features = image
    for filters in (16, 32, 64):
        features = tf.keras.layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )(features)
        features = tf.keras.layers.MaxPooling2D()(features)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = {
        f"concept_{name}": tf.keras.layers.Dense(
            len(CONCEPT_LABELS[name]), activation="softmax", name=f"concept_{name}"
        )(features)
        for name in CONCEPT_COLUMNS
    }
    return tf.keras.Model(image, outputs, name="concept_model")


def build_cbm(concept_model: tf.keras.Model) -> tuple[tf.keras.Model, tf.keras.Model]:
    concept_values = [
        concept_model.get_layer(name).output for name in CONCEPT_OUTPUTS
    ]
    bottleneck = tf.keras.layers.Concatenate(name="concept_bottleneck")(concept_values)
    concept_input = tf.keras.Input((sum(map(len, CONCEPT_LABELS.values())),))
    diagnosis_layer = tf.keras.layers.Dense(
        len(DIAGNOSIS_LABELS), activation="softmax", name="diagnosis"
    )
    diagnosis_head = tf.keras.Model(
        concept_input, diagnosis_layer(concept_input), name="diagnosis_head"
    )
    diagnosis = diagnosis_layer(bottleneck)
    outputs = {"diagnosis": diagnosis}
    outputs.update(zip(CONCEPT_OUTPUTS, concept_values))
    cbm = tf.keras.Model(concept_model.input, outputs, name="derm7pt_cbm")
    return cbm, diagnosis_head


def _callbacks(patience: int) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]


def _fit_concepts(
    model: tf.keras.Model, data: Derm7Pt, args: argparse.Namespace
) -> dict:
    model.compile(
        optimizer="adam",
        loss={name: "sparse_categorical_crossentropy" for name in CONCEPT_OUTPUTS},
        metrics={name: ["accuracy"] for name in CONCEPT_OUTPUTS},
    )
    history = model.fit(
        make_dataset(data, "train", args.image_size, args.batch_size, CONCEPT_OUTPUTS, True),
        validation_data=make_dataset(
            data, "valid", args.image_size, args.batch_size, CONCEPT_OUTPUTS
        ),
        epochs=args.epochs,
        callbacks=_callbacks(args.patience),
        verbose=2,
    )
    return history.history


def _concept_inputs(
    model: tf.keras.Model,
    data: Derm7Pt,
    split: str,
    args: argparse.Namespace,
    use_annotations: bool,
) -> np.ndarray:
    indexes = data.splits[split]
    if use_annotations:
        return np.concatenate(
            [
                tf.one_hot(data.concepts[name][indexes], len(CONCEPT_LABELS[name])).numpy()
                for name in CONCEPT_COLUMNS
            ],
            axis=1,
        )

    predictions = model.predict(
        make_dataset(data, split, args.image_size, args.batch_size),
        verbose=0,
    )
    if not isinstance(predictions, dict):
        predictions = dict(zip(model.output_names, predictions))
    return np.concatenate([predictions[name] for name in CONCEPT_OUTPUTS], axis=1)


def _fit_diagnosis_head(
    head: tf.keras.Model,
    concept_model: tf.keras.Model,
    data: Derm7Pt,
    args: argparse.Namespace,
    use_annotations: bool,
) -> dict:
    train_x = _concept_inputs(concept_model, data, "train", args, use_annotations)
    valid_x = _concept_inputs(concept_model, data, "valid", args, use_annotations)
    head.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = head.fit(
        train_x,
        data.diagnosis[data.splits["train"]],
        validation_data=(valid_x, data.diagnosis[data.splits["valid"]]),
        epochs=args.epochs,
        callbacks=_callbacks(args.patience),
        verbose=2,
    )
    return history.history


def _classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    class_names: tuple[str, ...] = DIAGNOSIS_LABELS,
) -> dict:
    count = len(class_names)
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (actual, predicted), 1)
    scores = {}
    f1s, recalls = [], []
    for index, label in enumerate(class_names):
        tp = confusion[index, index]
        precision = tp / confusion[:, index].sum() if confusion[:, index].sum() else 0
        recall = tp / confusion[index].sum() if confusion[index].sum() else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores[label] = {
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "support": int(confusion[index].sum()),
        }
        f1s.append(f1)
        recalls.append(recall)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(
            np.mean(np.asarray(recalls)[confusion.sum(axis=1) > 0])
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class": scores,
    }


def _predict(model: tf.keras.Model, data: Derm7Pt, split: str, args: argparse.Namespace):
    images = make_dataset(data, split, args.image_size, args.batch_size)
    predictions = model.predict(images, verbose=0)
    return predictions if isinstance(predictions, dict) else dict(zip(model.output_names, predictions))


def assess(
    cbm: tf.keras.Model,
    diagnosis_head: tf.keras.Model,
    concept_model: tf.keras.Model,
    data: Derm7Pt,
    split: str,
    args: argparse.Namespace,
) -> dict:
    indexes = data.splits[split]
    predictions = _predict(cbm, data, split, args)
    result = {
        "diagnosis_from_predicted_concepts": _classification_metrics(
            data.diagnosis[indexes], predictions["diagnosis"].argmax(axis=1)
        ),
        "concepts": {},
    }
    for name in CONCEPT_COLUMNS:
        result["concepts"][name] = _classification_metrics(
            data.concepts[name][indexes],
            predictions[f"concept_{name}"].argmax(axis=1),
            CONCEPT_LABELS[name],
        )
    true_concepts = _concept_inputs(concept_model, data, split, args, use_annotations=True)
    oracle = diagnosis_head.predict(true_concepts, verbose=0).argmax(axis=1)
    result["diagnosis_from_annotated_concepts"] = _classification_metrics(
        data.diagnosis[indexes], oracle
    )
    return result


def run(args: argparse.Namespace) -> dict:
    tf.keras.utils.set_random_seed(args.seed)
    data = load_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "diagnosis_labels": list(DIAGNOSIS_LABELS),
        "regimes": {},
        "split_sizes": {name: len(indexes) for name, indexes in data.splits.items()},
    }

    separate_regimes = {"independent", "sequential"}.intersection(args.regimes)
    if separate_regimes:
        # Both regimes use a separately trained concept model. Their diagnosis
        # heads differ: annotations versus predicted concepts.
        concept_model = build_concept_model(args.image_size)
        concept_history = _fit_concepts(concept_model, data, args)
        results["separate_concept_training_history"] = concept_history
        for regime, use_annotations in (("independent", True), ("sequential", False)):
            if regime not in separate_regimes:
                continue
            tf.keras.utils.set_random_seed(args.seed)
            cbm, head = build_cbm(concept_model)
            history = _fit_diagnosis_head(
                head, concept_model, data, args, use_annotations
            )
            results["regimes"][regime] = {
                "validation": assess(cbm, head, concept_model, data, "valid", args),
                "test": assess(cbm, head, concept_model, data, "test", args),
                "training_history": history,
            }
            cbm.save(args.output_dir / f"{regime}.keras")

    if "joint" in args.regimes:
        tf.keras.utils.set_random_seed(args.seed + 1)
        joint_concepts = build_concept_model(args.image_size)
        joint, joint_head = build_cbm(joint_concepts)
        weights = {"diagnosis": 1.0}
        weights.update({
            name: args.concept_loss_weight / len(CONCEPT_OUTPUTS)
            for name in CONCEPT_OUTPUTS
        })
        joint.compile(
            optimizer="adam",
            loss={name: "sparse_categorical_crossentropy" for name in weights},
            loss_weights=weights,
            metrics={name: ["accuracy"] for name in weights},
        )
        joint_history = joint.fit(
            make_dataset(
                data, "train", args.image_size, args.batch_size,
                ("diagnosis", *CONCEPT_OUTPUTS), True,
            ),
            validation_data=make_dataset(
                data, "valid", args.image_size, args.batch_size,
                ("diagnosis", *CONCEPT_OUTPUTS),
            ),
            epochs=args.epochs,
            callbacks=_callbacks(args.patience),
            verbose=2,
        )
        results["regimes"]["joint"] = {
            "validation": assess(joint, joint_head, joint_concepts, data, "valid", args),
            "test": assess(joint, joint_head, joint_concepts, data, "test", args),
            "training_history": joint_history.history,
        }
        joint.save(args.output_dir / "joint.keras")

    results["regime_definitions"] = {
        "independent": "Train concepts from images and diagnosis from annotated concepts.",
        "sequential": "Train concepts from images, then diagnosis from predicted concepts.",
        "joint": "Train concept and diagnosis losses together end-to-end.",
    }
    results["note"] = "Research use only; not validated for clinical diagnosis."
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved three models and assessment to {args.output_dir}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concept-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=("independent", "sequential", "joint"),
        default=("independent", "sequential", "joint"),
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size) < 1 or args.patience < 0:
        parser.error("epochs, batch-size, and image-size must be positive; patience cannot be negative.")
    if args.concept_loss_weight < 0:
        parser.error("concept-loss-weight cannot be negative.")
    return args


if __name__ == "__main__":
    run(parse_args())

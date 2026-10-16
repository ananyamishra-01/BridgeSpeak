#!/usr/bin/env python3
"""
BridgeSpeak model training script (v2).

Trains a BiLSTM multi-task classifier (urgency + intent) on the expanded dataset.
Designed to avoid overfitting / underfitting:
  - Train / validation / test split (70 / 15 / 15)
  - Class weights for the minority 'urgent' class
  - Dropout + L2 regularization
  - Early stopping on validation loss
  - Learning-rate reduction on plateau
  - Explicit evaluation metrics (precision/recall/F1) especially for urgent class

Usage:
  python train_model.py [--epochs 40] [--batch 32]

Requires: tensorflow, numpy
Outputs:
  ../model/bridgespeak_model/          (Keras SavedModel)
  ../model/bridgespeak_tfjs/           (TF.js conversion)
  ../model/bridgespeak_metadata.json
  ../model/training_report.json
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Numpy 1.x compatibility shims (needed on some environments with numpy 2.x)
# ---------------------------------------------------------------------------
import numpy as np
for _old, _new in [
    ("complex_", "complex128"),
    ("float_", "float64"),
    ("string_", "bytes_"),
    ("unicode_", "str_"),
]:
    if not hasattr(np, _old):
        try:
            setattr(np, _old, getattr(np, _new))
        except Exception:
            pass

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
except Exception as e:
    raise SystemExit(
        "TensorFlow is required. Install with:\n"
        "  pip install 'tensorflow>=2.12,<2.17' 'numpy<2'\n"
        f"Original error: {e}"
    )

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset.json"
OUT_DIR = ROOT / "model"
SEED = 42
MAX_SEQ_LEN = 12
EMBED_DIM = 64
LSTM_UNITS = 48
DROPOUT = 0.35
L2 = 1e-4


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = data["examples"]
    urgency_classes = data["urgency_classes"]
    intent_classes = data["intent_classes"]
    return examples, urgency_classes, intent_classes


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if t]


def build_vocab(tokenized: list[list[str]], min_freq: int = 1) -> dict[str, int]:
    cnt = Counter(t for seq in tokenized for t in seq)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and w not in word2idx:
            word2idx[w] = len(word2idx)
    return word2idx


def encode_seq(tokens: list[str], word2idx: dict, max_len: int) -> list[int]:
    ids = [word2idx.get(t, 1) for t in tokens][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


def stratified_split(examples, urgency_classes, intent_classes, ratios=(0.7, 0.15, 0.15)):
    """Stratify primarily on urgency (the critical minority class)."""
    by_urg = {u: [] for u in urgency_classes}
    for i, ex in enumerate(examples):
        by_urg[ex["urgency"]].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for u, idxs in by_urg.items():
        random.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1])) if n > 2 else 0
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def compute_class_weights(labels: list[int], n_classes: int) -> dict[int, float]:
    """Inverse-frequency weights, clipped to avoid extreme values."""
    cnt = Counter(labels)
    total = len(labels)
    weights = {}
    for c in range(n_classes):
        freq = cnt.get(c, 1)
        w = total / (n_classes * freq)
        weights[c] = float(np.clip(w, 0.5, 8.0))
    return weights


def build_model(vocab_size: int, n_intent: int) -> keras.Model:
    inp = layers.Input(shape=(MAX_SEQ_LEN,), dtype="int32", name="input")
    x = layers.Embedding(
        vocab_size, EMBED_DIM, mask_zero=True,
        embeddings_regularizer=regularizers.l2(L2),
        name="embedding",
    )(inp)
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_UNITS,
            return_sequences=False,
            dropout=DROPOUT,
            recurrent_dropout=0.15,
            kernel_regularizer=regularizers.l2(L2),
            name="lstm",
        ),
        name="bilstm",
    )(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Dropout(DROPOUT * 0.7)(x)

    urgency_out = layers.Dense(1, activation="sigmoid", name="urgency")(x)
    intent_out = layers.Dense(n_intent, activation="softmax", name="intent")(x)

    model = keras.Model(inp, [urgency_out, intent_out], name="bridgespeak_bilstm")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    examples, urgency_classes, intent_classes = load_dataset(DATA_PATH)
    print(f"Loaded {len(examples)} examples")
    print("Urgency distribution:", Counter(e["urgency"] for e in examples))
    print("Intent distribution:", Counter(e["intent"] for e in examples))

    tokenized = [tokenize(e["text"]) for e in examples]
    word2idx = build_vocab(tokenized)
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size}")

    X = np.array([encode_seq(t, word2idx, MAX_SEQ_LEN) for t in tokenized], dtype=np.int32)
    y_urg = np.array([urgency_classes.index(e["urgency"]) for e in examples], dtype=np.float32)
    y_int = np.array([intent_classes.index(e["intent"]) for e in examples], dtype=np.int32)

    train_idx, val_idx, test_idx = stratified_split(examples, urgency_classes, intent_classes)
    print(f"Split sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    yu_train, yu_val, yu_test = y_urg[train_idx], y_urg[val_idx], y_urg[test_idx]
    yi_train, yi_val, yi_test = y_int[train_idx], y_int[val_idx], y_int[test_idx]

    urg_weights = compute_class_weights(yu_train.astype(int).tolist(), 2)
    int_weights = compute_class_weights(yi_train.tolist(), len(intent_classes))
    print("Urgency class weights:", urg_weights)
    print("Intent class weights:", int_weights)

    # Sample weights for urgency head (binary) – emphasise minority class
    sample_w_urg = np.array([urg_weights[int(y)] for y in yu_train], dtype=np.float32)
    sample_w_int = np.array([int_weights[int(y)] for y in yi_train], dtype=np.float32)

    model = build_model(vocab_size, len(intent_classes))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "urgency": "binary_crossentropy",
            "intent": "sparse_categorical_crossentropy",
        },
        loss_weights={"urgency": 1.5, "intent": 1.0},  # prioritise urgency correctness
        metrics={
            "urgency": [
                keras.metrics.BinaryAccuracy(name="acc"),
                keras.metrics.Precision(name="prec"),
                keras.metrics.Recall(name="rec"),
            ],
            "intent": ["accuracy"],
        },
    )
    model.summary()

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(OUT_DIR / "best_checkpoint.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        {"urgency": yu_train, "intent": yi_train},
        sample_weight={"urgency": sample_w_urg, "intent": sample_w_int},
        validation_data=(X_val, {"urgency": yu_val, "intent": yi_val}),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=1,
    )

    # ---- Evaluation on held-out test set ----
    print("\n=== Test-set evaluation ===")
    results = model.evaluate(X_test, {"urgency": yu_test, "intent": yi_test}, verbose=0)
    metric_names = model.metrics_names
    test_metrics = dict(zip(metric_names, results))
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Confusion-style analysis for urgency (threshold 0.55 – slightly conservative)
    urg_pred_prob = model.predict(X_test, verbose=0)[0].flatten()
    urg_pred = (urg_pred_prob >= 0.55).astype(int)
    tp = int(((urg_pred == 1) & (yu_test == 1)).sum())
    fp = int(((urg_pred == 1) & (yu_test == 0)).sum())
    tn = int(((urg_pred == 0) & (yu_test == 0)).sum())
    fn = int(((urg_pred == 0) & (yu_test == 1)).sum())
    print(f"\nUrgency confusion (threshold 0.55): TP={tp} FP={fp} TN={tn} FN={fn}")
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    # ---- Save artefacts ----
    keras_path = OUT_DIR / "bridgespeak_model"
    model.export(str(keras_path))  # SavedModel
    model.save(str(OUT_DIR / "bridgespeak.keras"))

    metadata = {
        "word2idx": word2idx,
        "urgency_classes": urgency_classes,
        "intent_classes": intent_classes,
        "max_seq_len": MAX_SEQ_LEN,
        "urgency_threshold": 0.55,
        "stop_words": [],  # kept empty – model is trained end-to-end
        "keep_words": ["good", "bad", "not", "want", "need", "help", "no", "yes"],
        "version": "2.0",
        "vocab_size": vocab_size,
        "notes": "Retrained on expanded balanced dataset. Urgency threshold raised to 0.55 to reduce false positives.",
    }
    with open(OUT_DIR / "bridgespeak_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report = {
        "n_examples": len(examples),
        "vocab_size": vocab_size,
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "urgency_confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "urgency_f1": float(f1),
        "final_epoch": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
    }
    with open(OUT_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # TF.js conversion
    try:
        import tensorflowjs as tfjs
        tfjs_path = OUT_DIR / "bridgespeak_tfjs"
        tfjs.converters.save_keras_model(model, str(tfjs_path))
        print(f"TF.js model written to {tfjs_path}")
        # Keras 3 exports aren't loadable by tf.loadLayersModel until rewritten
        import subprocess, sys
        (tfjs_path / "model.keras3.json").unlink(missing_ok=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fix_tfjs_model.py"), str(tfjs_path / "model.json")], check=True)
    except Exception as e:
        print(f"TF.js conversion skipped ({e}). You can convert later with:")
        print("  tensorflowjs_converter --input_format=keras bridgespeak.keras bridgespeak_tfjs/")

    print("\nDone. Artefacts in", OUT_DIR)


if __name__ == "__main__":
    main()

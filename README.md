# BridgeSpeak v2 — Local Model Only

Assistive communication tool for deaf / mute users.  
Everything runs offline with a local BiLSTM classifier + template-based sentence reconstruction.

## Project layout

```
bridgespeak/
├── bridgespeak.html          # Frontend (open in browser)
├── data/
│   └── dataset.json          # Expanded training dataset (548 examples)
├── model/
│   ├── bridgespeak_metadata.json
│   └── bridgespeak_tfjs/     # TensorFlow.js model weights
└── scripts/
    └── train_model.py        # Retraining pipeline
```

## What changed

1. **No API** – no API key, no network calls for generation.
2. **Dataset expanded** – `data/dataset.json` (548 labelled examples, heavily weighted toward `not_urgent` to fix false positives).
3. **False-urgent bug fixed** – classification now requires:
   - model probability ≥ 0.55 **and**
   - presence of a true emergency keyword, **or**
   - very high confidence (≥ 0.85).
   Mild modifiers (`mild`, `slight`, `not`, `later` …) cancel urgency when no emergency keyword is present.
4. **Sentence reconstruction** – template engine driven by predicted intent + keywords (no LLM).
5. **Training script** – stratified split, class weights, dropout, L2, early stopping, ReduceLROnPlateau → protects against over/under-fitting.

## Running the app

Serve the folder over HTTP (TF.js cannot load models from `file://`):

```bash
cd bridgespeak
python3 -m http.server 8080
# open http://localhost:8080/bridgespeak.html
```

## Retraining

```bash
# Requires TensorFlow 2.12–2.16 and numpy < 2
pip install 'tensorflow>=2.12,<2.17' 'numpy<2' tensorflowjs
cd bridgespeak
python scripts/train_model.py --epochs 40 --batch 32
```

The script writes a new TF.js model into `model/bridgespeak_tfjs/` and updates `bridgespeak_metadata.json`.

It then runs `scripts/fix_tfjs_model.py`, which rewrites the Keras 3 `model.json` into the format
TensorFlow.js can load (without it, the browser falls back to rule-based mode with
"An InputLayer should be passed either a `batchInputShape` or an `inputShape`"). The original
export is kept as `model.keras3.json`. Run it manually if you convert a model yourself:

```bash
python scripts/fix_tfjs_model.py model/bridgespeak_tfjs/model.json
```

## Design notes on overfitting / underfitting

| Technique              | Purpose                                      |
|------------------------|----------------------------------------------|
| 70/15/15 stratified split | Reliable validation & test metrics        |
| Class weights          | Counter the minority `urgent` class          |
| Dropout 0.35 + recurrent dropout | Regularisation                    |
| L2 weight decay        | Further regularisation                       |
| Early stopping         | Stop when val loss stops improving           |
| ReduceLROnPlateau      | Fine-tune when plateauing                    |
| Urgency loss weight 1.5| Prioritise correct urgency decisions         |
| Expanded non-urgent data | Reduce false-positive urgent predictions  |

##IoT Hardware
This project is connected with a Linguistic Hand Glove, an IoT device which will track the sign languages and upload the output as the keywords in the file.

#!/usr/bin/env python3
"""
Make a Keras 3 TF.js export loadable by TensorFlow.js (tfjs-layers 4.x).

tensorflowjs_converter run under Keras 3 writes a topology that tf.loadLayersModel
cannot parse ("An InputLayer should be passed either a `batchInputShape` or an
`inputShape`"). This rewrites model.json in place into the Keras 2 style that
TF.js expects. Weights (.bin) are untouched. Run it after every retrain:

  python scripts/fix_tfjs_model.py [path/to/model.json]

A backup of the original is kept as model.keras3.json.
"""

from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "model" / "bridgespeak_tfjs" / "model.json"


def simplify(obj):
    """Recursively turn Keras 3 serialized objects into Keras 2 form."""
    if isinstance(obj, list):
        return [simplify(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    # DTypePolicy object -> plain dtype string
    if obj.get("class_name") == "DTypePolicy":
        return obj["config"]["name"]
    out = {}
    for k, v in obj.items():
        if k in ("module", "registered_name", "build_config", "quantization_config", "optional"):
            continue
        out[k] = simplify(v)
    return out


def convert_inbound(nodes):
    """Keras 3 {"args": [keras_tensor...], "kwargs": {}} -> Keras 2 [[[name, node, tensor, kwargs]]]."""
    converted = []
    for node in nodes:
        if isinstance(node, list):  # already Keras 2 style
            converted.append(node)
            continue
        kwargs = {k: v for k, v in node.get("kwargs", {}).items() if v is not None and k != "training"}
        entries = []
        for arg in node.get("args", []):
            args = arg if isinstance(arg, list) else [arg]
            for a in args:
                name, idx, t_idx = a["config"]["keras_history"]
                entries.append([name, idx, t_idx, kwargs])
        converted.append(entries)
    return converted


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    model = json.loads(path.read_text(encoding="utf-8"))
    topo = model["modelTopology"]
    if topo.get("keras_version", "").startswith("2"):
        print("Already Keras 2 format, nothing to do.")
        return

    backup = path.with_name("model.keras3.json")
    if not backup.exists():
        shutil.copy(path, backup)

    cfg = topo["model_config"]["config"]
    for layer in cfg["layers"]:
        layer["inbound_nodes"] = convert_inbound(layer.get("inbound_nodes", []))
        c = layer["config"]
        if layer["class_name"] == "InputLayer" and "batch_shape" in c:
            c["batch_input_shape"] = c.pop("batch_shape")
            c["dtype"] = "int32"  # token ids
        layer["config"] = simplify(c)
        if layer["class_name"] == "Bidirectional":
            # TF.js names the wrapped layers "forward_<name>" / "backward_<name>"
            layer["config"]["layer"]["config"]["name"] = "lstm"
            layer["config"].pop("backward_layer", None)

    # Keras 3 weight names: bilstm/forward_lstm/lstm_cell/kernel -> TF.js: bilstm/forward_lstm/kernel
    for group in model["weightsManifest"]:
        for w in group["weights"]:
            w["name"] = w["name"].replace("/lstm_cell/", "/")

    # input_layers / output_layers are [name, 0, 0] (single) or a list of them
    for key in ("input_layers", "output_layers"):
        v = cfg[key]
        if v and isinstance(v[0], str):
            cfg[key] = [v]

    topo["keras_version"] = "2.15.0"
    topo["model_config"]["config"] = cfg
    path.write_text(json.dumps(model), encoding="utf-8")
    print(f"Converted {path} (backup: {backup.name})")


if __name__ == "__main__":
    main()

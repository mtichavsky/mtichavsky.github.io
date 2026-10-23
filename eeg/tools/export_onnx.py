"""
Export the in-ear binary model to ONNX and generate golden fixtures for the JS port.

Dev-only; the website itself needs none of this. Run from the repo root:

    python eeg/tools/export_onnx.py --code <eeg/code> --checkpoint <run>/fold_N_best.pth

Writes:
    eeg/models/inear_binary.onnx
    eeg/js/coefficients.js           scipy filter coefficients
    eeg/tests/fixtures/golden.json   per-stage reference outputs of the Python pipeline
    eeg/tests/fixtures/sample.csv    synthetic IDUN-format recording (no patient data)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def synthetic_recording(fs: int = 250, seconds: int = 45, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Alpha + beta + 50 Hz mains + drift + noise, in microvolt-ish units."""
    rng = np.random.default_rng(seed)
    t = np.arange(fs * seconds) / fs
    x = (
        20 * np.sin(2 * np.pi * 10 * t)
        + 6 * np.sin(2 * np.pi * 22 * t + 1.0)
        + 8 * np.sin(2 * np.pi * 50 * t)
        + 40 * t / seconds
        + rng.normal(0, 8, t.size)
    )
    return t, x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, type=Path, help="eeg/code checkout")
    ap.add_argument("--checkpoint", required=True, type=Path, help="fold_N_best.pth to export")
    ap.add_argument("--model", default="CNNAttnS", help="architecture name in thesis.model_factory")
    args = ap.parse_args()
    sys.path.insert(0, str(args.code))

    import torch
    from scipy.signal import butter, filtfilt, iirnotch
    from scipy.signal import detrend as sp_detrend
    from scipy.stats import zscore

    from thesis.dataset import IDUNDataset, SpectrogramDataset
    from thesis.model_factory import create_model
    from thesis.stft import FREQ_CUTOFF_HZ, spectrogram_shape

    # Checkpoints record their spectrogram cutoff; a missing key means the 70 Hz default.
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    if ckpt.get("freq_cutoff_hz", FREQ_CUTOFF_HZ) != FREQ_CUTOFF_HZ:
        sys.exit(f"{args.checkpoint} was trained with a different freq_cutoff_hz; the JS port assumes 70 Hz")
    SPEC_SHAPE = spectrogram_shape(FREQ_CUTOFF_HZ)

    out_models = ROOT / "models"
    fixtures = ROOT / "tests" / "fixtures"
    out_models.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)

    # --- fixtures: synthetic CSV through the real Python pipeline -------------------------
    t, x = synthetic_recording()
    x = np.round(x, 6)  # what the CSV stores, so stage references see identical input
    csv_path = fixtures / "sample.csv"
    with open(csv_path, "w") as f:
        f.write("timestamp,ch1\n")
        for ti, xi in zip(t, x):
            f.write(f"{ti:.4f},{xi:.6f}\n")

    chunks = IDUNDataset.load_and_preprocess_idun_file(csv_path)  # (N, 1, 2500)
    specs = SpectrogramDataset.convert_to_spectrograms(chunks, source_fs=250.0)

    # Intermediate stages on the full signal, to localise any JS drift.
    z = zscore(x)
    d = sp_detrend(z, type="linear")
    nyq = 125.0
    b, a = butter(4, [1.0 / nyq, 70.0 / nyq], btype="band")
    bp = filtfilt(b, a, d)
    bn, an = iirnotch(50.0, Q=30, fs=250)
    notched = filtfilt(bn, an, bp)

    golden = {
        "spec_shape": list(SPEC_SHAPE),
        "freq_cutoff_hz": FREQ_CUTOFF_HZ,
        "butter_ba": {"b": b.tolist(), "a": a.tolist()},
        "notch_ba": {"b": bn.tolist(), "a": an.tolist()},
        "stages": {
            "zscore": z.tolist(),
            "detrend": d.tolist(),
            "bandpass": bp.tolist(),
            "notch": notched.tolist(),
        },
        "num_chunks": int(chunks.shape[0]),
        "chunk0": chunks[0, 0].tolist(),
        "spec0": specs[0][0].numpy().tolist(),
        "logits": None,
    }

    # --- model ----------------------------------------------------------------------------
    model = create_model(
        model_name=args.model,
        spec_shape=SPEC_SHAPE,
        dropout=0.0,
        num_classes=2,
        device=torch.device("cpu"),
        in_channels=1,
        pretrained_checkpoint=str(args.checkpoint),
    )
    model.eval()
    path = out_models / "inear_binary.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 1, *SPEC_SHAPE),
        str(path),
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={"spectrogram": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Parity: onnxruntime vs torch on real spectrograms.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batch = torch.stack(specs)  # (N, 1, H, W)
    with torch.no_grad():
        ref = model(batch).numpy()
    got = sess.run(None, {"spectrogram": batch.numpy()})[0]
    err = float(np.abs(ref - got).max())
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} ({params} params, epoch {ckpt.get('epoch')}) -> {path.name} "
          f"{path.stat().st_size / 1e6:.2f} MB, max |torch-onnx| = {err:.2e}")
    assert err < 1e-4, "ONNX export diverges from PyTorch"
    golden["logits"] = ref.tolist()

    coeffs = (
        "// Generated by tools/export_onnx.py from scipy (butter(4, [1, 70] Hz, band) and\n"
        "// iirnotch(50 Hz, Q=30)) at fs = 250 Hz. Do not edit.\n"
        f"export const BANDPASS = {json.dumps({'b': b.tolist(), 'a': a.tolist()})};\n"
        f"export const NOTCH = {json.dumps({'b': bn.tolist(), 'a': an.tolist()})};\n"
    )
    (ROOT / "js" / "coefficients.js").write_text(coeffs)

    with open(fixtures / "golden.json", "w") as f:
        json.dump(golden, f)
    print(f"fixtures: {csv_path.name}, golden.json ({golden['num_chunks']} chunks)")


if __name__ == "__main__":
    main()

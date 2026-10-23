// Run: node --test eeg/tests/parity.test.mjs
// Checks the JS pipeline against scipy/PyTorch reference output (fixtures/golden.json).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  zscore, detrendLinear, filtfilt, logSpectrogram, BANDPASS, NOTCH,
} from "../js/dsp.js";
import { parseIdunCsv, preprocess, filterSignal, aggregate } from "../js/pipeline.js";

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, "fixtures/golden.json"), "utf8"));
const csv = readFileSync(join(here, "fixtures/sample.csv"), "utf8");

const maxAbsDiff = (a, b) => {
  assert.equal(a.length, b.length, "length mismatch");
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

test("filter stages match scipy", () => {
  const { signal } = parseIdunCsv(csv);
  const z = zscore(signal);
  assert.ok(maxAbsDiff(z, golden.stages.zscore) < 1e-9, "zscore");
  const d = detrendLinear(z);
  assert.ok(maxAbsDiff(d, golden.stages.detrend) < 1e-9, "detrend");
  const bp = filtfilt(BANDPASS.b, BANDPASS.a, d);
  assert.ok(maxAbsDiff(bp, golden.stages.bandpass) < 1e-8, "bandpass");
  const nt = filtfilt(NOTCH.b, NOTCH.a, bp);
  assert.ok(maxAbsDiff(nt, golden.stages.notch) < 1e-8, "notch");
});

test("chunk and spectrogram match the Python pipeline", () => {
  const p = preprocess(csv);
  assert.equal(p.spectrograms.length, golden.num_chunks);
  assert.deepEqual([p.bins, p.frames], golden.spec_shape);
  const ref = golden.spec0.flat();
  assert.ok(maxAbsDiff(p.spectrograms[0], ref) < 1e-4, "spectrogram of chunk 0");
});

test("end-to-end logits match PyTorch through onnxruntime-web", async () => {
  const ort = await import("../vendor/ort/ort.wasm.min.mjs");
  ort.env.wasm.numThreads = 1;
  const p = preprocess(csv);
  const bytes = readFileSync(join(here, "../models/inear_binary.onnx"));
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  for (let i = 0; i < p.spectrograms.length; i++) {
    const input = new ort.Tensor("float32", p.spectrograms[i], [1, 1, p.bins, p.frames]);
    const out = await session.run({ spectrogram: input });
    const diff = maxAbsDiff(Array.from(out.logits.data), golden.logits[i]);
    assert.ok(diff < 1e-3, `chunk ${i}: logits differ by ${diff}`);
  }
});

test("majority vote breaks ties by first seen", () => {
  const r = aggregate([[0, 1], [1, 0], [1, 0], [0, 1]]);
  assert.equal(r.finalPrediction, 1); // counts tie 2-2; Counter keeps first-seen order
  assert.equal(r.finalClassName, "Pathological");
});

test("rejects non-IDUN csv", () => {
  assert.throws(() => parseIdunCsv("Fp1,Fp2\n1,2\n"), /ch1/);
});

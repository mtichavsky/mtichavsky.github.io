// Runs preprocessing and inference off the main thread. All work is local; nothing is uploaded.

import * as ort from "./vendor/ort/ort.wasm.min.mjs";
import { preprocess, aggregate } from "./js/pipeline.js";

// GitHub Pages cannot send the COOP/COEP headers that threaded WASM needs, and the models are
// tiny, so run single-threaded.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;

let session = null;

self.onmessage = async ({ data: { text } }) => {
  try {
    self.postMessage({ type: "step", step: 0 });
    const pre = preprocess(text);

    self.postMessage({ type: "step", step: 1 });
    session ??= ort.InferenceSession.create(
      new URL("./models/inear_binary.onnx", import.meta.url).href, { executionProviders: ["wasm"] });
    const sess = await session;

    self.postMessage({ type: "step", step: 2 });
    const logits = [];
    for (const spec of pre.spectrograms) {
      const input = new ort.Tensor("float32", spec, [1, 1, pre.bins, pre.frames]);
      const out = await sess.run({ spectrogram: input });
      logits.push(Array.from(out.logits.data));
    }

    self.postMessage({
      type: "done",
      result: aggregate(logits),
      spectrograms: pre.spectrograms,
      bins: pre.bins,
      frames: pre.frames,
      seconds: pre.seconds,
      fsEstimate: pre.fsEstimate,
    });
  } catch (err) {
    self.postMessage({ type: "error", message: err.message ?? String(err) });
  }
};

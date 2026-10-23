// IDUN CSV -> model-ready spectrograms, mirroring IDUNDataset.load_and_preprocess_idun_file and
// thesis.inference (no quality file, as in the API), plus majority-vote aggregation.

import {
  BANDPASS, NOTCH, CHUNK_SAMPLES, FS,
  zscore, detrendLinear, filtfilt, logSpectrogram,
} from "./dsp.js";

export const CLASS_NAMES = ["Healthy", "Pathological"];

/** Parse an IDUN CSV (header with `ch1`, optional `timestamp`) into a signal and a rate estimate. */
export function parseIdunCsv(text) {
  const lines = text.split(/\r?\n/);
  const header = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  const ch = header.indexOf("ch1");
  if (ch < 0) {
    throw new Error('No "ch1" column found. Expected an IDUN in-ear CSV with a ch1 column.');
  }
  const ts = header.indexOf("timestamp");

  const signal = [];
  let t0 = NaN, t1 = NaN;
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const cols = lines[i].split(",");
    const v = Number(cols[ch]);
    if (!Number.isFinite(v)) continue;
    signal.push(v);
    if (ts >= 0) {
      const t = Number(cols[ts]);
      if (Number.isFinite(t)) { if (Number.isNaN(t0)) t0 = t; t1 = t; }
    }
  }
  const fsEstimate = signal.length > 1 && t1 > t0 ? (signal.length - 1) / (t1 - t0) : null;
  return { signal: Float64Array.from(signal), fsEstimate };
}

/** z-score -> linear detrend -> 1-70 Hz band-pass -> 50 Hz notch (whole recording). */
export function filterSignal(raw) {
  let x = zscore(raw);
  x = detrendLinear(x);
  x = filtfilt(BANDPASS.b, BANDPASS.a, x);
  return filtfilt(NOTCH.b, NOTCH.a, x);
}

/** Non-overlapping 10 s chunks, each z-scored on its own; a trailing partial chunk is dropped. */
export function chunkSignal(x) {
  const chunks = [];
  for (let i = 0; i + CHUNK_SAMPLES <= x.length; i += CHUNK_SAMPLES) {
    chunks.push(zscore(x.subarray(i, i + CHUNK_SAMPLES)));
  }
  return chunks;
}

/** Full preprocessing: returns {spectrograms: Float32Array[], bins, frames, fsEstimate}. */
export function preprocess(csvText) {
  const { signal, fsEstimate } = parseIdunCsv(csvText);
  if (signal.length < CHUNK_SAMPLES) {
    throw new Error(
      `Recording is too short: ${(signal.length / FS).toFixed(1)} s, need at least 10 s.`);
  }
  const chunks = chunkSignal(filterSignal(signal));
  const specs = chunks.map(logSpectrogram);
  return {
    spectrograms: specs.map((s) => s.data),
    bins: specs[0].bins,
    frames: specs[0].frames,
    fsEstimate,
    seconds: signal.length / FS,
  };
}

/** Majority vote over per-chunk argmax, same tie-break as collections.Counter.most_common. */
export function aggregate(logitsPerChunk) {
  const names = CLASS_NAMES;
  const preds = logitsPerChunk.map((l) => l.indexOf(Math.max(...l)));
  const counts = names.map((_, k) => preds.filter((p) => p === k).length);
  // Counter.most_common keeps first-seen order on ties, so only a strictly larger count wins.
  const firstSeen = [...new Set(preds)];
  const final = firstSeen.reduce((best, k) => (counts[k] > counts[best] ? k : best), firstSeen[0]);
  return {
    finalPrediction: final,
    finalClassName: names[final],
    totalChunks: preds.length,
    classCounts: Object.fromEntries(names.map((n, k) => [n, counts[k]])),
    classPercentages: Object.fromEntries(names.map((n, k) => [n, (counts[k] / preds.length) * 100])),
    chunkPredictions: preds,
  };
}

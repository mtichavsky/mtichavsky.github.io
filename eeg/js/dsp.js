// Signal processing ported from the Python training pipeline (scipy), so a browser-side
// spectrogram equals the one the model saw in training. Verified against golden fixtures in
// tests/. Everything is float64 until the final tensor, like scipy.

export const FS = 250;
export const CHUNK_SAMPLES = 10 * FS;
export const NPERSEG = 256;
export const NOVERLAP = 192;
export const HOP = NPERSEG - NOVERLAP;
// Highest spectrogram frequency kept (thesis/stft.py FREQ_CUTOFF_HZ): bins 0..69.34 Hz -> 72 rows.
export const FREQ_CUTOFF_HZ = 70;
export const NUM_FREQ_BINS = Math.floor(FREQ_CUTOFF_HZ / (FS / NPERSEG)) + 1;

export { BANDPASS, NOTCH } from "./coefficients.js";

export function zscore(x) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const std = Math.sqrt(v / n); // ddof=0, like scipy.stats.zscore
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) / std;
  return out;
}

/** scipy.signal.detrend(x, type="linear"): subtract the least-squares line. */
export function detrendLinear(x) {
  const n = x.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += i; sy += x[i]; sxx += i * i; sxy += i * x[i];
  }
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const intercept = (sy - slope * sx) / n;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = x[i] - (slope * i + intercept);
  return out;
}

/** Solve A x = y by Gaussian elimination with partial pivoting (A is n x n, row-major arrays). */
function solve(A, y) {
  const n = y.length;
  const M = A.map((row, i) => [...row, y[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/** scipy.signal.lfilter_zi: steady-state initial conditions for a unit step. */
export function lfilterZi(b, a) {
  const n = Math.max(a.length, b.length);
  const bb = [...b, ...new Array(n - b.length).fill(0)].map((v) => v / a[0]);
  const aa = [...a, ...new Array(n - a.length).fill(0)].map((v) => v / a[0]);
  const m = n - 1;
  // (I - A^T) zi = B - a[1:] * b[0], with A the companion matrix of a.
  const IminusAT = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let aT = 0; // A^T[i][j] = A[j][i]; A[0][*] = -a[1:], A[k][k-1] = 1 (k >= 1)
      if (j === 0) aT = -aa[i + 1];
      else if (i === j - 1) aT = 1;
      return (i === j ? 1 : 0) - aT;
    }));
  const rhs = Array.from({ length: m }, (_, i) => bb[i + 1] - aa[i + 1] * bb[0]);
  return solve(IminusAT, rhs);
}

/** scipy.signal.lfilter with initial state zi (direct form II transposed). Returns [y, zf]. */
export function lfilter(b, a, x, zi) {
  const n = Math.max(a.length, b.length);
  const bb = [...b, ...new Array(n - b.length).fill(0)].map((v) => v / a[0]);
  const aa = [...a, ...new Array(n - a.length).fill(0)].map((v) => v / a[0]);
  const z = Float64Array.from(zi);
  const y = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = bb[0] * xi + (n > 1 ? z[0] : 0);
    for (let k = 0; k < n - 2; k++) z[k] = bb[k + 1] * xi + z[k + 1] - aa[k + 1] * yi;
    if (n > 1) z[n - 2] = bb[n - 1] * xi - aa[n - 1] * yi;
    y[i] = yi;
  }
  return y;
}

/** scipy.signal.filtfilt defaults: odd padding of 3*max(len(a), len(b)) samples, lfilter_zi. */
export function filtfilt(b, a, x) {
  const padlen = 3 * Math.max(a.length, b.length);
  const n = x.length;
  if (n <= padlen) throw new Error(`Signal too short for filtering (${n} samples).`);
  const ext = new Float64Array(n + 2 * padlen);
  for (let i = 0; i < padlen; i++) {
    ext[i] = 2 * x[0] - x[padlen - i];
    ext[padlen + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  }
  ext.set(x, padlen);

  const zi = lfilterZi(b, a);
  let y = lfilter(b, a, ext, zi.map((v) => v * ext[0]));
  y.reverse();
  y = lfilter(b, a, y, zi.map((v) => v * y[0]));
  y.reverse();
  return y.slice(padlen, padlen + n);
}

// ---- STFT -----------------------------------------------------------------------------------

/** Periodic Hamming window (scipy get_window default, fftbins=True). */
const WINDOW = Float64Array.from({ length: NPERSEG }, (_, i) =>
  0.54 - 0.46 * Math.cos((2 * Math.PI * i) / NPERSEG));
const WINDOW_SUM = WINDOW.reduce((s, v) => s + v, 0);

/** In-place iterative radix-2 complex FFT. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
        const vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
      }
    }
  }
}

/**
 * log1p(|scipy.signal.stft(x, fs, nperseg=256, noverlap=192, window="hamming")|), keeping rows
 * up to FREQ_CUTOFF_HZ, as compute_log_spectrogram does. Returns a row-major Float32Array of shape [bins, frames] plus the shape.
 */
export function logSpectrogram(x) {
  // boundary="zeros" (NPERSEG/2 each side), then padded=True up to a whole number of hops.
  let total = x.length + NPERSEG;
  const rem = (total - NPERSEG) % HOP;
  if (rem) total += HOP - rem;
  const sig = new Float64Array(total);
  sig.set(x, NPERSEG / 2);

  const frames = 1 + (total - NPERSEG) / HOP;
  const bins = NUM_FREQ_BINS;
  const out = new Float32Array(bins * frames);
  const re = new Float64Array(NPERSEG), im = new Float64Array(NPERSEG);
  for (let f = 0; f < frames; f++) {
    const start = f * HOP;
    for (let i = 0; i < NPERSEG; i++) { re[i] = sig[start + i] * WINDOW[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < bins; k++) {
      out[k * frames + f] = Math.log1p(Math.hypot(re[k], im[k]) / WINDOW_SUM);
    }
  }
  return { data: out, bins, frames };
}

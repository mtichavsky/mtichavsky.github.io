const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

let busy = false;

/** Show exactly one state of the analysis pane: empty, running, failed or result. */
function setState(name) {
  for (const id of ["empty", "running", "failed", "result"]) $(id).hidden = id !== name;
}

function setStep(step) {
  document.querySelectorAll(".steps li").forEach((li) => {
    const n = Number(li.dataset.step);
    li.classList.toggle("done", n < step);
    li.classList.toggle("active", n === step);
  });
}

function setBusy(on) {
  busy = on;
  $("sample").disabled = on;
}

function run(name, text) {
  if (busy) return;
  setBusy(true);
  $("file-card").hidden = false;
  $("f-name").textContent = name;
  $("f-name").title = name;
  for (const id of ["f-length", "f-rate", "f-segments"]) $(id).textContent = "…";
  $("f-rate").classList.remove("warn");
  setStep(0);
  setState("running");
  if (matchMedia("(max-width: 820px)").matches) {
    $("output-h").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth" });
  }
  worker.postMessage({ text });
}

function fail(message) {
  setBusy(false);
  $("error").textContent = message;
  for (const id of ["f-length", "f-rate", "f-segments"]) $(id).textContent = "–";
  setState("failed");
}

worker.onmessage = ({ data }) => {
  if (data.type === "step") return setStep(data.step);
  if (data.type === "error") {
    return fail(`${data.message} Check that the file is an IDUN in-ear CSV and try again.`);
  }
  setBusy(false);
  show(data);
};

worker.onerror = () => fail(
  "The analysis could not start. This page needs a recent browser (Chrome, Edge, Firefox or Safari 15+).");

function show({ result, spectrograms, bins, frames, seconds, fsEstimate }) {
  const names = Object.keys(result.classCounts);
  const rateOff = fsEstimate && Math.abs(fsEstimate - 250) > 5;

  $("f-length").textContent = seconds >= 120
    ? `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s` : `${seconds.toFixed(1)} s`;
  $("f-rate").textContent = fsEstimate ? `${fsEstimate.toFixed(0)} Hz` : "250 Hz (assumed)";
  $("f-rate").classList.toggle("warn", Boolean(rateOff));
  $("f-segments").textContent = `${result.totalChunks} × 10 s`;

  $("verdict-name").textContent = result.finalClassName;
  $("verdict-name").dataset.class = result.finalClassName;
  const hits = result.classCounts[result.finalClassName];
  $("verdict-sub").textContent =
    `${hits} of ${result.totalChunks} ten-second segments were classified as ${result.finalClassName.toLowerCase()}; the majority decides.`;

  $("bars").replaceChildren(...names.map((name) => {
    const pct = result.classPercentages[name];
    const li = document.createElement("li");
    li.dataset.class = name;
    li.innerHTML = '<span></span><span class="track"><span class="fill"></span></span><span class="pct"></span>';
    li.children[0].textContent = name;
    li.children[2].textContent = `${Math.round(pct)}%`;
    requestAnimationFrame(() => { li.querySelector(".fill").style.width = `${pct}%`; });
    return li;
  }));

  // One cell per segment, the same width as that segment's slice of the spectrogram above.
  $("strip").replaceChildren(...result.chunkPredictions.map((p, i) => {
    const li = document.createElement("li");
    li.dataset.class = names[p];
    li.textContent = result.totalChunks <= 40 ? names[p][0] : "";
    li.title = `${i * 10}–${(i + 1) * 10} s: ${names[p]}`;
    return li;
  }));
  $("t-end").textContent = `${result.totalChunks * 10} s`;

  drawSpectrogram(spectrograms, bins, frames);
  const dropped = Math.round((seconds - result.totalChunks * 10) * 10) / 10;
  $("spec-note").textContent =
    "Spectrogram of the filtered signal as the model sees it, one block per segment." +
    (dropped > 0 ? ` The last ${dropped} s are shorter than a segment and were not used.` : "") +
    (rateOff ? ` Timestamps suggest ${fsEstimate.toFixed(0)} Hz, not 250 Hz, so the output may be unreliable.` : "");
  setState("result");
}

// Dark teal -> green -> pale yellow, scaled to the recording's 99th percentile.
const RAMP = [[26, 56, 70], [47, 158, 107], [243, 240, 200]];
function colour(t) {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2);
  const f = x - i;
  return RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
}

function drawSpectrogram(specs, bins, frames) {
  const width = specs.length * frames;
  const canvas = $("spec");
  canvas.width = width;
  canvas.height = bins;
  const all = specs.flatMap((s) => Array.from(s)).sort((a, b) => a - b);
  const hi = all[Math.floor(all.length * 0.99)] || 1;
  const img = new ImageData(width, bins);
  specs.forEach((s, c) => {
    for (let k = 0; k < bins; k++) {
      for (let f = 0; f < frames; f++) {
        const [r, g, b] = colour(s[k * frames + f] / hi);
        const o = ((bins - 1 - k) * width + c * frames + f) * 4; // low frequencies at the bottom
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
      }
    }
  });
  canvas.getContext("2d").putImageData(img, 0, 0);
}

// ---- input ----------------------------------------------------------------------------------

async function loadFile(file) {
  if (file) run(file.name, await file.text());
}

$("file").addEventListener("change", (e) => {
  loadFile(e.target.files[0]);
  e.target.value = ""; // allow picking the same file again
});

const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));

$("sample").addEventListener("click", async () => {
  const res = await fetch(new URL("./tests/fixtures/sample.csv", import.meta.url));
  run("synthetic-sample.csv", await res.text());
});

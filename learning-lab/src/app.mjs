import {
  LESSON_STEPS,
  advanceLesson,
  buildPixelWork,
  createLessonState,
  nextGridIndex,
  recordPrediction,
  retreatLesson,
} from "./lesson-model.mjs";
import {
  CODE_SAMPLES,
  getLessonCopy,
  getPredictionFeedback,
} from "./lesson-content.mjs";

const STEP_LABELS = ["Story", "Predict", "Simulate", "Code", "Run", "Explain", "Challenge"];
const state = {
  lesson: createLessonState(),
  simulationTimers: [],
  simulationHasRun: false,
  codeTab: "python",
  blockSize: 8,
  capabilities: null,
  executionResult: null,
  executionError: null,
  executionRunning: false,
};

const elements = {
  back: document.querySelector("#back-button"),
  next: document.querySelector("#next-button"),
  content: document.querySelector("#step-content"),
  count: document.querySelector("#step-count"),
  mode: document.querySelector("#execution-mode"),
  progress: document.querySelector("#progress-list"),
  simulation: document.querySelector("#simulation-panel"),
  grid: document.querySelector("#pixel-grid"),
  inspector: document.querySelector("#thread-inspector"),
  status: document.querySelector("#simulation-status"),
  simulationHeading: document.querySelector("#simulation-heading"),
  run: document.querySelector("#run-simulation"),
  reset: document.querySelector("#reset-simulation"),
};

function currentStep() {
  return LESSON_STEPS[state.lesson.stepIndex];
}

function clearSimulationTimers() {
  state.simulationTimers.forEach(window.clearTimeout);
  state.simulationTimers = [];
}

function renderProgress() {
  elements.progress.innerHTML = STEP_LABELS.map((label, index) => {
    const status = index < state.lesson.stepIndex
      ? "complete"
      : index === state.lesson.stepIndex
        ? "current"
        : "upcoming";
    const marker = status === "complete" ? "✓" : String(index + 1).padStart(2, "0");
    return `<li class="progress-item progress-${status}" ${status === "current" ? 'aria-current="step"' : ""}>
      <span class="progress-marker">${marker}</span><span>${label}</span>
    </li>`;
  }).join("");
}

function storyMarkup() {
  return `<div class="copy-column">
    <p class="lede">Imagine an 8 × 8 picture with no color yet. One CPU worker could visit all 64 pixels in sequence—or we could give every pixel its own GPU worker.</p>
    <div class="story-comparison" aria-label="One sequential worker compared with many parallel workers">
      <article class="story-card">
        <span class="story-number">1</span>
        <div><strong>CPU worker</strong><p>Walks across the picture one pixel at a time.</p></div>
      </article>
      <span class="versus" aria-hidden="true">versus</span>
      <article class="story-card story-card-accent">
        <span class="story-number">64</span>
        <div><strong>GPU workers</strong><p>Each receives one pixel and the same simple instruction.</p></div>
      </article>
    </div>
    <p class="learning-note"><strong>Notice:</strong> GPUs are useful when lots of independent data needs the same operation.</p>
  </div>`;
}

function predictionMarkup() {
  const selected = state.lesson.prediction;
  const feedback = selected ? `<p class="feedback">${getPredictionFeedback(selected)}</p>` : "";
  return `<div class="copy-column">
    <p class="lede">Both teams must color exactly 64 pixels. Before seeing the work map, choose what you expect.</p>
    <fieldset class="prediction-options">
      <legend class="sr-only">Which approach will finish first?</legend>
      ${[
        ["cpu", "One CPU worker", "A powerful worker handles every pixel in order."],
        ["gpu", "Many GPU workers", "Small workers each handle one independent pixel."],
        ["same", "About the same", "The total amount of coloring work is unchanged."],
      ].map(([value, title, detail]) => `<button type="button" class="prediction-card ${selected === value ? "is-selected" : ""}" data-prediction="${value}" aria-pressed="${selected === value}">
        <span class="prediction-radio" aria-hidden="true"></span><span><strong>${title}</strong><small>${detail}</small></span>
      </button>`).join("")}
    </fieldset>
    ${feedback}
  </div>`;
}

function simulationMarkup() {
  return `<div class="copy-column compact-copy">
    <p class="lede">Every square below is both a pixel and a job. Run the concept simulation to watch 64 threads claim 64 jobs.</p>
    <p class="learning-note"><strong>This is not a benchmark.</strong> Animation time helps us see assignment order; it does not represent GPU execution speed.</p>
  </div>`;
}

function codeMarkup() {
  const tabs = ["python", "pytorch", "triton", "cuda"];
  return `<div class="copy-column code-column">
    <p class="lede">The operation stays the same as we move closer to the hardware. What changes is how explicitly we describe the workers.</p>
    <div class="code-tabs" role="group" aria-label="Implementation level">
      ${tabs.map((tab) => `<button type="button" class="code-tab ${state.codeTab === tab ? "is-selected" : ""}" aria-pressed="${state.codeTab === tab}" data-code-tab="${tab}">${tab === "pytorch" ? "PyTorch" : tab[0].toUpperCase() + tab.slice(1)}</button>`).join("")}
    </div>
    <pre class="code-panel" tabindex="0"><code>${escapeHtml(CODE_SAMPLES[state.codeTab])}</code></pre>
    <p class="code-caption">${codeCaption(state.codeTab)}</p>
  </div>`;
}

function explainMarkup() {
  return `<div class="copy-column">
    <p class="lede">A GPU kernel is one instruction run by many workers. Three ideas connect the code to the picture.</p>
    <div class="concept-grid">
      <article><span class="concept-icon">ID</span><strong>Identity</strong><p>Each worker asks, “Which thread am I?”</p></article>
      <article><span class="concept-icon">→</span><strong>Destination</strong><p>The thread ID maps to one pixel in memory.</p></article>
      <article><span class="concept-icon">⌁</span><strong>Boundary</strong><p>A mask stops extra workers from touching pixels outside the image.</p></article>
    </div>
    <div class="mapping-strip"><span>Metal <code>grid + thread_position</code></span><span aria-hidden="true">↔</span><span>Triton <code>program_id + arange</code></span><span aria-hidden="true">↔</span><span>CUDA <code>blockIdx + threadIdx</code></span><small>Metal dispatches threads in threadgroups; a Triton program processes vector lanes; CUDA organizes individual threads into blocks. They solve the same indexing problem with different execution models.</small></div>
  </div>`;
}

function providerById(id) {
  return state.capabilities?.providers?.find((provider) => provider.id === id);
}

function runMarkup() {
  const apple = providerById("apple-mlx");
  const modal = providerById("modal-triton");
  const appleReady = apple?.available === true;
  const modalReady = modal?.available === true;
  return `<div class="copy-column run-column">
    <p class="lede">The animation showed the mapping. Now choose hardware and measure a real kernel. Both backends return the same correctness and timing format.</p>
    <div class="provider-grid">
      <article class="provider-card">
        <span class="provider-kind">Local · no cloud charge</span>
        <h2>Apple GPU · MLX</h2>
        <p>Runs a custom Metal kernel on this Mac using explicit grid and threadgroup sizes.</p>
        <p class="provider-status ${appleReady ? "is-ready" : ""}">${apple ? (appleReady ? "Ready on this Mac" : apple.reason) : "Checking local companion…"}</p>
        <button class="button button-primary" type="button" data-run-provider="apple-mlx" ${appleReady && !state.executionRunning ? "" : "disabled"}>Run on your Apple GPU</button>
      </article>
      <article class="provider-card">
        <span class="provider-kind">Cloud · explicit confirmation</span>
        <h2>NVIDIA L4 · Triton</h2>
        <p>Runs the equivalent masked Triton kernel through your authenticated Modal account.</p>
        <p class="provider-status ${modalReady ? "is-ready" : ""}">${modal ? (modalReady ? "Modal CLI detected" : modal.reason) : "Checking local companion…"}</p>
        <label class="cost-confirm"><input id="modal-confirm" type="checkbox"> Modal uses billable NVIDIA L4 compute. I want to launch one run.</label>
        <button class="button button-quiet" type="button" data-run-provider="modal-triton" ${modalReady && !state.executionRunning ? "" : "disabled"}>Run once on Modal</button>
      </article>
    </div>
    <div id="gpu-run-status" class="gpu-run-status" aria-live="polite">${executionStatusMarkup()}</div>
  </div>`;
}

function executionStatusMarkup() {
  if (state.executionRunning) return "Compiling and running the kernel…";
  if (state.executionError) return `<strong>Run unavailable</strong><span>${escapeHtml(state.executionError)}</span>`;
  if (!state.executionResult) return "Choose an available backend when you are ready.";
  const result = state.executionResult;
  const latency = result.measurements[0].value;
  return `<div class="result-heading"><span class="result-check">✓</span><div><strong>Correct output on ${escapeHtml(result.device)}</strong><span>Measured GPU execution · not simulation</span></div></div>
    <dl class="result-grid"><div><dt>Backend</dt><dd>${escapeHtml(result.implementation)}</dd></div><div><dt>Latency</dt><dd>${latency.toFixed(4)} ms</dd></div><div><dt>Max error</dt><dd>${result.correctness.max_abs_error}</dd></div><div><dt>Checksum</dt><dd>${result.output.checksum}</dd></div></dl>`;
}

function challengeMarkup() {
  return `<div class="copy-column">
    <p class="lede">The picture still has 64 pixels. Change how many workers form a block, then rerun the same work map.</p>
    <div class="challenge-row">
      <label for="block-size"><strong>Teaching group size</strong><small>Metal uses a threadgroup; Triton rounds this to a power-of-two vector width; CUDA uses threads per block.</small></label>
      <select id="block-size">
        ${[4, 8, 10, 16].map((size) => `<option value="${size}" ${state.blockSize === size ? "selected" : ""}>${size}</option>`).join("")}
      </select>
    </div>
    <p class="learning-note"><strong>Your checkpoint:</strong> “A GPU divides similar work among many workers, and every worker needs an index that identifies its data.”</p>
  </div>`;
}

function escapeHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function codeCaption(tab) {
  return {
    python: "One worker visits every row and column in sequence.",
    pytorch: "PyTorch describes the whole operation and dispatches GPU work for us.",
    triton: "A program ID selects a block; one program handles a vector of pixel offsets and masks overflow lanes.",
    cuda: "CUDA builds a global pixel index from a block ID and a thread ID.",
  }[tab];
}

function renderStep() {
  clearSimulationTimers();
  const step = currentStep();
  const copy = getLessonCopy(step);
  const content = {
    story: storyMarkup,
    predict: predictionMarkup,
    simulate: simulationMarkup,
    code: codeMarkup,
    run: runMarkup,
    explain: explainMarkup,
    challenge: challengeMarkup,
  }[step]();

  elements.count.textContent = `Step ${state.lesson.stepIndex + 1} of ${LESSON_STEPS.length}`;
  elements.mode.textContent = step === "run" ? "Measured execution" : "Concept simulation";
  elements.content.innerHTML = `<div class="step-heading"><span class="eyebrow">${copy.eyebrow}</span><h1 id="step-title">${copy.title}</h1></div>${content}`;
  elements.simulation.hidden = !new Set(["simulate", "challenge"]).has(step);
  elements.back.disabled = state.lesson.stepIndex === 0;
  elements.next.textContent = step === "challenge" ? "Finish lesson ✓" : "Continue →";
  renderProgress();
  bindStepEvents();

  if (!elements.simulation.hidden) {
    resetSimulation();
  }
}

function bindStepEvents() {
  document.querySelectorAll("[data-prediction]").forEach((button) => {
    button.addEventListener("click", () => {
      const prediction = button.dataset.prediction;
      state.lesson = recordPrediction(state.lesson, prediction);
      renderStep();
      document.querySelector(`[data-prediction="${prediction}"]`)?.focus();
    });
  });
  document.querySelectorAll("[data-code-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.codeTab = button.dataset.codeTab;
      renderStep();
      document.querySelector(`[data-code-tab="${state.codeTab}"]`)?.focus();
    });
  });
  document.querySelector("#block-size")?.addEventListener("change", (event) => {
    state.blockSize = Number(event.target.value);
    resetSimulation();
  });
  document.querySelectorAll("[data-run-provider]").forEach((button) => {
    button.addEventListener("click", () => runRealGpu(button.dataset.runProvider));
  });
}

async function loadCapabilities() {
  try {
    const response = await fetch("/api/capabilities");
    if (!response.ok) throw new Error("Companion API unavailable");
    state.capabilities = await response.json();
  } catch (_error) {
    state.capabilities = { providers: [
      { id: "apple-mlx", available: false, reason: "Start with `python learning-lab/server.py` to enable real GPU runs." },
      { id: "modal-triton", available: false, reason: "Start the companion server and authenticate Modal first." },
    ] };
  }
  if (currentStep() === "run") renderStep();
}

async function runRealGpu(provider) {
  const confirmed = provider === "modal-triton"
    ? document.querySelector("#modal-confirm")?.checked === true
    : false;
  if (provider === "modal-triton" && !confirmed) {
    state.executionError = "Confirm the billable Modal L4 run before launching.";
    renderStep();
    return;
  }
  state.executionRunning = true;
  state.executionError = null;
  state.executionResult = null;
  renderStep();
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, group_size: state.blockSize, confirmed: provider === "modal-triton" }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "GPU execution failed");
    state.executionResult = payload;
  } catch (error) {
    state.executionError = error.message;
  } finally {
    state.executionRunning = false;
    renderStep();
  }
}

function resetSimulation() {
  clearSimulationTimers();
  state.simulationHasRun = false;
  const work = buildPixelWork({ width: 8, height: 8, blockSize: state.blockSize });
  elements.grid.innerHTML = work.threads.map((thread, index) =>
    `<button type="button" tabindex="${index === 0 ? 0 : -1}" class="pixel ${thread.active ? "" : "is-masked"}" data-thread="${thread.threadId}" aria-label="${thread.active ? `Pixel ${thread.x}, ${thread.y}; waiting` : `Thread ${thread.threadId}; masked because it is outside the image`}"></button>`,
  ).join("");
  const maskedCount = work.launchedThreads - work.totalPixels;
  elements.simulationHeading.textContent = `${work.totalPixels} pixels · ${work.launchedThreads} launched threads`;
  elements.status.textContent = `Ready. ${work.totalPixels} pixels are waiting; ${work.blockCount} blocks of ${state.blockSize} launch${maskedCount ? `, with ${maskedCount} overflow threads stopped by the mask` : ""}.`;
  elements.inspector.innerHTML = `<span class="section-label">Thread inspector</span><strong>Select a pixel</strong><p>Run the simulation, then choose any square to trace its worker.</p>`;
  elements.run.disabled = false;
  bindPixels(work);
}

function runSimulation() {
  resetSimulation();
  const work = buildPixelWork({ width: 8, height: 8, blockSize: state.blockSize });
  elements.run.disabled = true;
  elements.status.textContent = "Assigning one active thread to every pixel…";
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const delay = reducedMotion ? 0 : 28;

  work.threads.filter((thread) => thread.active).forEach((thread, index) => {
    const timer = window.setTimeout(() => {
      const pixel = elements.grid.querySelector(`[data-thread="${thread.threadId}"]`);
      pixel.classList.add("is-painted", `block-${thread.blockId % 4}`);
      pixel.setAttribute("aria-label", `Pixel ${thread.x}, ${thread.y}; painted by thread ${thread.threadId} in block ${thread.blockId}`);
      if (index === work.totalPixels - 1) {
        state.simulationHasRun = true;
        elements.run.disabled = false;
        elements.status.textContent = `Complete: ${work.totalPixels} active threads painted ${work.totalPixels} pixels. This animation shows mapping, not elapsed GPU time.`;
      }
    }, index * delay);
    state.simulationTimers.push(timer);
  });
}

function bindPixels(work) {
  const pixels = [...elements.grid.querySelectorAll(".pixel")];
  pixels.forEach((pixel, index) => {
    pixel.addEventListener("click", () => {
      const thread = work.threads[Number(pixel.dataset.thread)];
      elements.grid.querySelectorAll(".pixel").forEach((item) => item.classList.remove("is-inspected"));
      pixel.classList.add("is-inspected");
      elements.inspector.innerHTML = thread.active ? `<span class="section-label">Thread inspector</span><strong>Thread ${thread.threadId}</strong><dl>
        <div><dt>Block</dt><dd>${thread.blockId}</dd></div>
        <div><dt>Lane in block</dt><dd>${thread.laneId}</dd></div>
        <div><dt>Pixel</dt><dd>(${thread.x}, ${thread.y})</dd></div>
      </dl><p><code>pixel = ${thread.threadId}</code></p>` : `<span class="section-label">Thread inspector</span><strong>Thread ${thread.threadId} is masked</strong><p>Its offset is outside the 64-pixel image, so the boundary mask prevents a memory write.</p>`;
    });
    pixel.addEventListener("keydown", (event) => {
      if (!new Set(["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "Home", "End"]).has(event.key)) return;
      event.preventDefault();
      const targetIndex = nextGridIndex({ current: index, key: event.key, columns: 8, total: pixels.length });
      pixels.forEach((item, itemIndex) => item.tabIndex = itemIndex === targetIndex ? 0 : -1);
      pixels[targetIndex].focus();
    });
  });
}

elements.next.addEventListener("click", () => {
  if (currentStep() === "challenge") {
    elements.content.innerHTML = `<div class="completion"><span class="completion-mark">✓</span><h1 id="step-title">Trail marker reached</h1><p>You mapped threads to pixels and connected Python, Triton, and CUDA indexing.</p><button class="button button-primary" type="button" id="restart-lesson">Run the lesson again</button></div>`;
    elements.simulation.hidden = true;
    elements.next.hidden = true;
    elements.back.hidden = true;
    document.querySelector("#restart-lesson").addEventListener("click", () => window.location.reload());
    return;
  }

  try {
    state.lesson = advanceLesson(state.lesson);
    renderStep();
  } catch (error) {
    const firstChoice = document.querySelector("[data-prediction]");
    firstChoice?.focus();
    elements.content.querySelector(".prediction-options")?.setAttribute("aria-invalid", "true");
  }
});

elements.back.addEventListener("click", () => {
  if (state.lesson.stepIndex === 0) return;
  state.lesson = retreatLesson(state.lesson);
  renderStep();
});
elements.run.addEventListener("click", runSimulation);
elements.reset.addEventListener("click", resetSimulation);

renderStep();
loadCapabilities();

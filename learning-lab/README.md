# GPU Safari Learning Lab

The learning lab is a platform-neutral, browser-based companion to GPU Safari's runnable experiments. Its first lesson, **Paint Pixels in Parallel**, introduces thread-to-data mapping before asking learners to configure a GPU provider.

The animation is a concept simulation, not a hardware benchmark. It never invents GPU timing data.

## Run the concept-only lab

From the repository root:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/learning-lab/>.

No JavaScript packages or GPU are required. Run the lesson-model tests with:

```bash
cd learning-lab
npm test
```

Run the full repository test suite with:

```bash
python -m pytest -q
```

## Run a real Metal kernel on Apple silicon

On an Apple silicon Mac with macOS 14 or newer, create an isolated environment and start the companion server:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r learning-lab/requirements-mac.txt
python learning-lab/server.py
```

Open <http://127.0.0.1:8000>, reach the **Run** step, and choose **Run on your Apple GPU**. The custom MLX Metal kernel returns measured latency, device information, correctness, and an output checksum. Tiny teaching kernels are dominated by dispatch overhead, so treat the timing as an observation rather than a performance score.

## Compare with NVIDIA through Modal

Install and authenticate Modal as described in [`platforms/modal/README.md`](../platforms/modal/README.md), then start the same companion server. The Modal option is cost-gated in the interface: it launches billable NVIDIA L4 compute only after the learner checks the confirmation box and clicks **Run once on Modal**.

The Apple and NVIDIA paths share one result contract while keeping their execution models distinct: Metal grids and threadgroups, Triton programs and vector lanes, and CUDA blocks and threads are related concepts—not interchangeable names.

## Current scope

- Progressive story, prediction, simulation, code, explanation, and challenge stages
- Accessible 8×8 thread-to-pixel work map
- Python, PyTorch, Triton, and CUDA concept comparison
- Real Apple GPU execution through an MLX custom Metal kernel
- Explicitly confirmed Modal Triton execution on NVIDIA L4
- Provider-neutral correctness and timing results
- Adjustable block size
- Responsive layout and reduced-motion support

The browser simulation remains available when neither real backend is configured.

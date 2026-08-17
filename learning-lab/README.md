# GPU Safari Learning Lab

The learning lab is a platform-neutral, browser-based companion to GPU Safari's runnable experiments. Its first lesson, **Paint Pixels in Parallel**, introduces thread-to-data mapping before asking learners to configure a GPU provider.

The animation is a concept simulation, not a hardware benchmark. It never invents GPU timing data.

## Run locally

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

## Current scope

- Progressive story, prediction, simulation, code, explanation, and challenge stages
- Accessible 8×8 thread-to-pixel work map
- Python, PyTorch, Triton, and CUDA concept comparison
- Adjustable block size
- Responsive layout and reduced-motion support

Real GPU execution and provider adapters are intentionally outside this first vertical slice.

# Modal

Install dependencies and authenticate once:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m modal setup
```

Run the CUDA reduction experiment:

```bash
modal run platforms/modal/reduction.py
```

This command launches billable L4 compute and normally completes within a few minutes. It uses a 10-minute timeout, a 2-second scale-down window, and no warm containers. The launcher compiles the experiment, runs correctness checks before benchmarking, validates the returned JSON against `benchmarks/schema/result-v1.schema.json`, and scales to zero.

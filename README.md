# Code Studio — Local Python Coding Assistant

A small, self-hosted Python coding assistant built around a decoder-only transformer implemented in PyTorch. The training checkpoint contains approximately 143 million parameters; the CPU deployment uses an INT8 dynamically quantized checkpoint. A Flask interface handles code requests, supports curated-example retrieval (28 verified functions and 5 project templates in the current build), and allows generated snippets/projects to be downloaded.

This is an experimental learning project, **not a general-purpose production coding model**. Generated code can be incomplete or incorrect. For example, the current web-scraper benchmark prompt produces a partial solution even with the full-precision model.

## Architecture

- `gpt_model.py`: transformer architecture (10 layers, 16 attention heads, 1,024 hidden dimensions, 512-token context).
- `09_inference.py`: checkpoint loading, generation and nearest-match retrieval.
- `06_prepare_instruct_data.py`: curated examples and data-preparation logic needed by inference.
- `code_checks.py`: syntax and missing-import checks on generated Python.
- `backend.py` and `static/`: Flask API and browser interface.
- `export/`: tokenizer assets. The model checkpoint itself is **not included in this repository**.

## Run locally

1. Install a compatible Python environment with PyTorch CPU and the deployment dependencies:

   ```powershell
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements-deploy.txt
   ```

2. Put **your own trusted** INT8 checkpoint at `export/model_int8.pt`. The checkpoint is not distributed here. PyTorch checkpoint loading in this project uses `weights_only=False`: **never load an untrusted .pt file**.
3. Start the app from this repository's root:

   ```powershell
   python backend.py
   ```

4. Open `http://127.0.0.1:5000`.

This ZIP is a deployment snapshot, not a complete training repository: the earlier pretraining, instruction tuning and quantization scripts are not part of the attached snapshot. Without the trained checkpoint the interface cannot generate model responses.

## Deployment and security

The original demo runs on a CPU laptop with Windows Task Scheduler, and is proxied over HTTPS using Tailscale Funnel. A permanently available demo requires that laptop to remain powered on, connected and awake.

**Do not expose this snapshot to the public internet without adding access control first.** The present API has no login: users of a public Funnel endpoint can send requests and read or clear shared chat history. The public demo URL is deliberately not listed here until that is fixed. For private use, prefer access restricted to your own tailnet rather than a public Funnel.

## Benchmark caveat

A single matching output from full precision and INT8 is not a quality benchmark: retrieval may return the same curated response without sampling the model. Compare CPU-vs-CPU latency and a held-out set of prompts that do not match the retrieval examples, and assess functional correctness by running tests. The reported parameter counts for quantized PyTorch modules are not directly comparable to the original parameter count.

## Included / excluded

Published: Python source, web UI, tokenizer assets and deployment requirements. Excluded: 202 MB checkpoint, conversation history, local environment files, training checkpoints and UI backups. No model weight or training data redistribution rights are claimed.

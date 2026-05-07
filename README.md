# PACZero: PAC-Private Fine-Tuning of Language Models via Sign Quantization

This repository is the official code release for the paper *PACZero: PAC-Private Fine-Tuning of Language Models via Sign
Quantization*. It contains the training/evaluation code for both the PAC-MI
and PAC-ZPL variants of our mechanism, our in-house reproductions of the
two DP zeroth-order baselines we compare against (DPZero, DP-AggZO), and a
script per cell reported in the paper.

> **Where to run from.** This repository's contents (this `README.md`,
> `paczo/`, `src/`, `scripts/`, `setup.sh`, `requirements.txt`) sit at the
> top level of the repo. After `git clone`, **`cd` into the cloned directory**
> (whatever you named it) and run every command from there. Scripts under
> `scripts/` self-resolve the repo root from their own location, so the
> directory name doesn't matter — only the cwd at invocation does.

## Repository layout

```
paczo_prod/
├── paczo/                 # PACZero training entrypoints + sign-quantized PAC trainer
│   ├── run_paczo.py       # PAC-MI and PAC-ZPL entrypoint (single-direction K=1)
│   ├── run_paczo_kvar.py  # K-aggregation extension (K ∈ {4, 16})
│   ├── pac_trainer.py
│   ├── pac_trainer_kvar.py
│   └── pac_utils.py       # binary-channel MI calibration + subset bookkeeping
├── src/                   # Task / template / model utilities (forked, see below)
├── baselines/             # populated by setup.sh on first run; not vendored
├── scripts/               # One bash script per cell reported in the paper
│   ├── headline/          # Table 1 (headline SST-2 + SQuAD cells)
│   ├── ablation/          # plateaus / clip / rank / LR sweeps / mechanism / T-ladders / K-ablation
│   └── baselines/         # in-house DP-cliff cells (DPZero K=1, DP-AggZO K=64)
├── setup.sh               # clones the DP-AggZO baseline at a pinned commit
├── README.md
├── SCRIPTS.md             # script ↔ paper cell map
└── requirements.txt
```

## Provenance of forked code

The training scaffolding under `src/` is derived from the public DP-AggZO
codebase (Bao et al. 2025), which in turn declares itself derived from
DPZero (Liu et al. 2024) and MeZO (Malladi et al. 2023). The DP-AggZO
baseline used for the DP-cliff reproductions in Table 3 is **not vendored**
in this repository — `setup.sh` clones it from its public source at a
pinned commit on first run (see Setup below).

| File | Upstream | Modification |
|---|---|---|
| `src/ht_opt.py` | DP-AggZO, `opt/src/ht_opt.py`. Originally derived from MeZO. | Unmodified. |
| `src/lora.py` | DP-AggZO, `opt/src/lora.py`. | Unmodified. |
| `src/metrics.py` | DP-AggZO, `opt/src/metrics.py`. | Unmodified. |
| `src/prefix.py` | DP-AggZO, `opt/src/prefix.py`. | Unmodified. |
| `src/tasks.py` | DP-AggZO, `opt/src/tasks.py`. | Minor edits: trimmed unused dataset shims. |
| `src/templates.py` | DP-AggZO, `opt/src/templates.py`. | Minor edits: trimmed unused templates. |
| `src/utils.py` | DP-AggZO, `opt/src/utils.py`. | Minor edits: removed dead helpers; unchanged signatures. |
| `baselines/dp-aggzo/` | DP-AggZO, full repository. Cloned by `setup.sh` at the pinned commit; not vendored in this repo. | Unmodified. |
| `paczo/*.py` | This repository (original). | Original work. The trainer subclasses Hugging Face `transformers.Trainer` and reuses the MeZO-style perturbation pattern. |

The mechanism, the binary-channel MI calibration (`pac_utils.py`), the
PAC-ZPL releaser, the M-subset bookkeeping, and the K-aggregation extension
(`run_paczo_kvar.py` / `pac_trainer_kvar.py`) are all original to this
repository. We do not include any model weights.

## Setup

Tested on Linux with Python 3.9, CUDA 12.1, and a single GPU per process.
OPT-1.3B fits on 16 GB VRAM; OPT-6.7B requires ≥40 GB VRAM.

After cloning this repository, **all commands below should be run from the
repository root** (the directory containing this `README.md`). Every script
under `scripts/` resolves the repo root from its own location, so the directory
you clone into can be named anything — but the cwd at invocation time should
be the repo root.

```bash
# Inside the cloned repository root:

# 1. Isolated Python environment (uv recommended; conda or venv work too).
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Clone the DP-AggZO baseline at the pinned commit (only needed if you
#    plan to run the DP-cliff baselines under scripts/baselines/).
bash setup.sh
```

`setup.sh` is idempotent — re-running it is a no-op if the baseline is
already at the pinned commit. The PACZero scripts under `scripts/headline/`
and `scripts/ablation/` do **not** depend on it.

If `uv` is unavailable, plain `python3.9 -m venv .venv && pip install -r requirements.txt` works. PyTorch wheels for CUDA 12.1 may need a `--extra-index-url` depending on your platform.

## Datasets

Both SST-2 and SQuAD are loaded automatically from Hugging Face Datasets on
first run. Task-side preprocessing follows DP-AggZO / MeZO and is implemented
in `src/tasks.py` and `src/templates.py`.

## Training

Every cell in the paper has a corresponding script under `scripts/` (mapped
in `SCRIPTS.md`). All scripts are runnable from the repository root. To
reproduce a single cell — for example, the SST-2 1.3B LoRA PAC-MI MI=0.33
headline cell at seed 0 — from the cloned repo root:

```bash
bash scripts/headline/sst2_1p3b_lora_pacmi_mi033_s0.sh
```

Each script auto-detects the repository root from its own location, so it
works regardless of what you named the cloned directory or where you placed it.

Each script writes outputs to `result/runs/<run_id>/`:

* `metrics.json` — final test/dev metrics.
* `checkpoints/` — Hugging Face Trainer checkpoints (saved at every `eval_steps`,
  `save_total_limit=3`; the dev-best checkpoint is loaded at end-of-run).

A 1–2 minute smoke check that exercises the PAC-MI pipeline at tiny step
budget is provided as a single command:

```bash
bash scripts/smoke.sh
```

To run any other script as a smoke check, set the `SMOKE` environment
variable:

```bash
SMOKE=1 SMOKE_STEPS=5 SMOKE_EVAL=5 bash scripts/headline/sst2_1p3b_ft_paczpl_s1.sh
```

## Evaluation

Test/dev metrics are written into `result/runs/<run_id>/metrics.json` at
the end of training. The `metric_for_best_model` is `eval_loss` for SST-2
and the generation-F1 (`eval_f1`) for SQuAD; the dev-best checkpoint is
loaded back as the final model and evaluated once on the held-out test
split. There is no separate evaluation script.

## Pre-trained models

We do not release any fine-tuned checkpoints. All cells reported in the
paper start from the public OPT-1.3B and OPT-6.7B base models on Hugging
Face Hub.

## Results

The paper reports four headline configurations on each of two tasks (SST-2,
SQuAD) at two model scales (OPT-1.3B, OPT-6.7B) and two parameter tracks
(LoRA r=8, full-parameter FT). Headline numbers, per-seed values, and the
reproduction commands are mapped to the corresponding scripts in
`SCRIPTS.md`. The most compute-intensive ablations (the FT plateau at
1.3B, the canonical PAC-ZPL T-ladders, the K-aggregation 6.7B cells) are
broken into per-cell scripts so that any subset can be re-run independently.

## License

The forked DP-AggZO baseline retains its upstream license (see
`baselines/dp-aggzo/LICENSE`). The PACZero-original code is released for
review under an MIT-style license (full text to be added in the
camera-ready release).


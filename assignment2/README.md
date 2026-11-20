# Assignment 2: English vs Bengali prompts

Compare local Ollama answers for 50 aligned English/Bengali prompt pairs.

## Run

Install Ollama and Python 3.10 or later. Download the model once:

```bash
ollama pull qwen2.5:0.5b
```

Ollama must be running. If needed, run `ollama serve` in another terminal.
From this folder:

```bash
python runner.py
```

Use `python3 runner.py` if your system has no `python` command. No pip packages
or virtual environment are required. You can also run
`python assignment2/runner.py` from the repository root.

The runner processes both languages once (100 answers), prints prompts, answers,
token counts and timing, and saves all output beside `runner.py` by default.
To save into a folder, edit the variable near the top of `runner.py`:

```python
OUTPUT_DIR = None       # Default: beside runner.py
# OUTPUT_DIR = "my-run"  # Instead: assignment2/my-run/ (created automatically)
```

Relative folder names are resolved beside the script; absolute paths also work.
The dataset is always read from beside `runner.py`.
Expect roughly 8–15 minutes on an i5-10310U laptop, excluding model download;
background load affects timing.

**Each run replaces the five generated files in the selected output folder.** Copy them elsewhere first if
you want to keep an earlier run. A failed or interrupted run preserves partial
results and marks metadata `failed`; running again starts from the beginning.
Both comparison files are generated only after all pairs finish.

## Files

| File | Contents |
|---|---|
| `dataset.tsv` | Pair ID, category, English prompt and Bengali (`bangla`) prompt |
| `runner.py` | Runs Ollama and generates every output file |
| `requirements.txt` | Notes that no third-party Python packages are needed |
| `results.csv` | All prompts and answers, input/output tokens, estimated cost and seconds per call |
| `raw.json` | JSON array containing every saved answer and original Ollama response |
| `metadata.json` | Model/version, settings, timestamps, dataset hash and completion status |
| `comparison.csv` | One row per pair comparing tokens, elapsed seconds and estimated cost |
| `summary.csv` | Cumulative English/Bengali totals and average timing across all prompts |

In `comparison.csv`, each metric has English and Bengali values, a difference
(Bengali minus English), and a ratio (Bengali divided by English). Total tokens
means input plus output tokens. A ratio is blank when its English denominator
is zero. `latency_s` includes model unload, load, inference and local HTTP.

`summary.csv` compares prompt counts, summed input/output/total tokens, summed
latency, mean latency per prompt and summed estimated cost. Each metric has
English and Bengali values, a difference (Bengali minus English), a ratio
(Bengali divided by English), and percent change relative to English. Ratios
use language totals, not averages of per-pair ratios. A zero English value
leaves ratio and percent change blank. Summed latency covers measured calls,
not the entire script's wall-clock time.

Both languages use the same system instruction, temperature 0, seed 42,
4096-token context, batch size 8 and maximum 128 output tokens. Each call unloads
the model before inference and uses `keep_alive=0`. Counts come directly from
Ollama, including chat-template/system overhead. `raw.json` retains
`done_reason`; `length` means the answer hit the generation limit.

Costs are course estimates ($0.15 per million input tokens, $0.60 per million
output tokens), not paid API charges. The small model can produce incorrect or
poor Bengali answers; review answers before drawing quality conclusions.

The included saved results and metadata come from the completed run
`output-20261120-213817` (run 1). They were preserved during simplification;
`raw.json` was converted from its JSONL records and `comparison.csv` calculated
from those measurements. Running the script replaces these with a new run.

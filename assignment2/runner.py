"""Run all English/Bangla prompt pairs: python runner.py"""

import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = "3b_run"  # None saves beside this script; e.g. "my-run" saves under assignment2/my-run.
MODEL = "qwen2.5:3b"
HOST = "http://127.0.0.1:11434"
SYSTEM_PROMPT = "You are a helpful assistant. Give a correct, concise answer in the language of the user's prompt."
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 128, "num_ctx": 4096, "num_batch": 8}
FIELDS = ["id", "language", "prompt", "output", "input_tokens", "output_tokens", "cost_usd", "latency_s"]


def request(endpoint, data=None):
    """Call the local Ollama server, allowing up to ten minutes per request."""
    req = Request(
        HOST + "/api/" + endpoint,
        data=None if data is None else json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=600) as response:
        return json.load(response)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dataset():
    with (ROOT / "dataset.tsv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["id", "category", "english", "bangla"]:
            raise ValueError("dataset.tsv must have id, category, english, bangla columns")
        pairs = list(reader)
    if not pairs or len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Dataset must contain pairs with unique IDs")
    if any(not str(p.get(k) or "").strip() for p in pairs for k in ("id", "category", "english", "bangla")):
        raise ValueError("Dataset contains an empty field")
    return pairs


def output_folder():
    """Resolve relative output folders beside the script, regardless of cwd."""
    return ROOT if OUTPUT_DIR is None else ROOT / Path(OUTPUT_DIR)


def write_comparison(pairs, rows):
    """Compare each aligned pair; ratios are Bangla divided by English."""
    indexed = {(str(r["id"]), r["language"]): r for r in rows}
    expected = {(p["id"], lang) for p in pairs for lang in ("english", "bangla")}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Comparison requires exactly one result per language per pair")
    comparisons = []
    for pair in pairs:
        english = indexed[(pair["id"], "english")]
        bangla = indexed[(pair["id"], "bangla")]
        result = {"id": pair["id"], "category": pair["category"],
                  "english_prompt": pair["english"], "bangla_prompt": pair["bangla"]}
        for metric in ("input_tokens", "output_tokens", "total_tokens", "latency_s", "cost_usd"):
            def value(row):
                if metric == "total_tokens":
                    return int(row["input_tokens"]) + int(row["output_tokens"])
                return int(row[metric]) if metric.endswith("tokens") else float(row[metric])
            en, bn = value(english), value(bangla)
            result.update({f"english_{metric}": en, f"bangla_{metric}": bn,
                           f"{metric}_difference": bn - en,
                           f"{metric}_ratio": bn / en if en else ""})
        comparisons.append(result)
    with (output_folder() / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


def write_summary(rows):
    """Compare language totals, using ratios of totals rather than pair ratios."""
    totals = {}
    for language in ("english", "bangla"):
        selected = [r for r in rows if r["language"] == language]
        if not selected:
            raise ValueError(f"Summary requires results for {language}")
        inputs = sum(int(r["input_tokens"]) for r in selected)
        outputs = sum(int(r["output_tokens"]) for r in selected)
        latency = sum(float(r["latency_s"]) for r in selected)
        totals[language] = {
            "prompt_count": len(selected),
            "input_tokens": inputs,
            "output_tokens": outputs,
            "total_tokens": inputs + outputs,
            "total_latency_s": latency,
            "mean_latency_s": latency / len(selected),
            "total_cost_usd": sum(float(r["cost_usd"]) for r in selected),
        }
    with (output_folder() / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "metric", "english", "bangla", "difference", "ratio", "percent_change",
        ])
        writer.writeheader()
        for metric, en in totals["english"].items():
            bn = totals["bangla"][metric]
            writer.writerow({"metric": metric, "english": en, "bangla": bn,
                             "difference": bn - en, "ratio": bn / en if en else "",
                             "percent_change": (bn - en) / en * 100 if en else ""})


def main():
    pairs = load_dataset()
    # Check the server and model before replacing any saved files.
    version, models = request("version"), request("tags")
    if not any(m.get("name") == MODEL for m in models.get("models", [])):
        raise ValueError(f"Model missing. Run: ollama pull {MODEL}")
    output = output_folder()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": MODEL, "server_version": version, "models": models,
        "system_prompt": SYSTEM_PROMPT, "options": OPTIONS, "runs_requested": 1,
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "dataset_sha256": hashlib.sha256((ROOT / "dataset.tsv").read_bytes()).hexdigest(),
        "pairs": len(pairs), "calls_completed": 0, "python": platform.python_version(),
        "platform": platform.platform(),
        "output_dir": str(output),
        "latency_policy": "cold start, includes explicit unload, load and HTTP",
        "cache_policy": "unload before every request, keep_alive=0, no shared conversation",
        "cost_policy": "course estimate: $0.15/million input tokens, $0.60/million output tokens; no API charge",
    }
    rows, raw_records = [], []
    started = time.perf_counter()
    write_json(output / "metadata.json", metadata)
    write_json(output / "raw.json", raw_records)
    (output / "comparison.csv").unlink(missing_ok=True)
    (output / "summary.csv").unlink(missing_ok=True)
    try:
        with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            handle.flush()
            for index, pair in enumerate(pairs):
                languages = ("english", "bangla") if index % 2 == 0 else ("bangla", "english")
                for language in languages:
                    call_started = time.perf_counter()
                    request("generate", {"model": MODEL, "prompt": "", "keep_alive": 0, "stream": False})
                    raw = request("chat", {
                        "model": MODEL, "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": pair[language]},
                        ], "options": OPTIONS, "keep_alive": 0, "stream": False,
                    })
                    elapsed = time.perf_counter() - call_started
                    inputs, outputs = raw.get("prompt_eval_count"), raw.get("eval_count")
                    if (raw.get("done") is not True or type(inputs) is not int or inputs <= 0
                            or type(outputs) is not int or outputs < 0):
                        raise ValueError("Ollama returned an incomplete response or invalid token counts")
                    if raw.get("prompt_eval_cached_count", 0):
                        raise ValueError("Ollama reported cached prompt tokens")
                    row = {"id": pair["id"], "language": language, "prompt": pair[language],
                           "output": raw["message"]["content"], "input_tokens": inputs,
                           "output_tokens": outputs, "cost_usd": (inputs * .15 + outputs * .60) / 1_000_000,
                           "latency_s": elapsed}
                    rows.append(row)
                    raw_records.append({**row, "category": pair["category"],
                                        "done_reason": raw.get("done_reason", "unknown"), "raw": raw})
                    writer.writerow(row)
                    handle.flush()
                    write_json(output / "raw.json", raw_records)
                    metadata["calls_completed"] = len(rows)
                    write_json(output / "metadata.json", metadata)
                    print(f"[{len(rows)}/{len(pairs) * 2}] Pair {pair['id']} ({language}) — "
                          f"input={inputs}, output={outputs}, time={elapsed:.2f}s", flush=True)
                    print(f"Prompt: {pair[language]}\nAnswer: {row['output']}\n", flush=True)
        write_comparison(pairs, rows)
        write_summary(rows)
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["elapsed_s"] = time.perf_counter() - started
        write_json(output / "metadata.json", metadata)
    print(f"Saved results.csv, raw.json, metadata.json, comparison.csv and summary.csv in {output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Partial results are saved; the next run starts again.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Run failed: {exc}\nCheck that Ollama is running (ollama serve).", file=sys.stderr)
        raise SystemExit(1)

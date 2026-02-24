import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict

# -----------------------------
# USER CONFIG
# -----------------------------
TSLIB_DIR = Path(__file__).resolve().parent

ROOT_PATH = Path(r"C:\Users\kasgr\Documents\Masteroppgave\master_repository\Master-s-Thesis\data\processed")
DATA_PATH = "parquet_data_karmoy_2024_L42_processed.csv"

N_FEATURES = 91
TASK_NAME = "short_term_forecast"
FEATURES_MODE = "M"
TARGET = "OT"
FREQ = "t"
SEQ_LEN = 48
LABEL_LEN = 24
PRED_LEN = 1

USE_GPU = False
NUM_WORKERS = 0
ITR = 1
PATIENCE = 5
FINAL_EPOCHS = 20
DES = "final"

# --- point this to a specific sweep run folder ---
# Example: TSLIB_DIR / "checkpoints" / "sweeps" / "20260220_115730"
SWEEP_RUN_DIR = TSLIB_DIR / "checkpoints" / "sweeps" / "YOUR_SWEEP_RUN_ID_HERE"

# Selection mode
BEST_PER_MODEL = True          # True = one best config per architecture
TOP_K_OVERALL = 3              # used only if BEST_PER_MODEL=False

# -----------------------------
# Paths inside the sweep run
# -----------------------------
SWEEP_RESULTS = SWEEP_RUN_DIR / "sweep_results.jsonl"
FINAL_DIR = SWEEP_RUN_DIR / "final_eval"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_RESULTS = FINAL_DIR / "final_results.json"
LOG_DIR = FINAL_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no} in {path}: {e}") from e
    return rows


def read_metrics(metrics_path: Path) -> Dict[str, Optional[float]]:
    """
    Reads metrics.jsonl produced by Exp_Short_Term_Forecast.
    Returns:
      best_val: best validation loss seen (from epoch_end rows)
      test_mse: test mse (from test_end row)
    """
    best_val = None
    test_mse = None

    if not metrics_path.exists():
        return {"best_val": None, "test_mse": None}

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("event") == "epoch_end":
                bv = obj.get("best_val")
                if isinstance(bv, (int, float)):
                    best_val = float(bv)

            if obj.get("event") == "test_end":
                tm = obj.get("test_mse")
                if isinstance(tm, (int, float)):
                    test_mse = float(tm)

    return {"best_val": best_val, "test_mse": test_mse}


def pick_best_per_model(trials: List[Dict[str, Any]], require_success: bool = True) -> List[Dict[str, Any]]:
    """
    One best trial per model type.
    Ranking: lowest best_val_loss wins. (Tie-breaker: lower seconds, then model_id.)
    If require_success=True: only consider returncode==0; if none for a model, fall back to all.
    """
    by_model = defaultdict(list)
    for t in trials:
        m = t.get("model")
        if m is None:
            continue
        by_model[m].append(t)

    winners = []
    for model, items in by_model.items():
        ok = [x for x in items if x.get("returncode") == 0 and x.get("best_val_loss") is not None]
        pool = ok if (require_success and ok) else items

        def key(x: Dict[str, Any]):
            return (
                x.get("best_val_loss", float("inf")),
                x.get("seconds", float("inf")),
                str(x.get("model_id", "")),
            )

        winners.append(min(pool, key=key))

    winners.sort(key=lambda x: (x.get("model", ""), x.get("best_val_loss", float("inf"))))
    return winners


def pick_top_k_overall(trials: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    ok = [t for t in trials if t.get("returncode") == 0 and t.get("best_val_loss") is not None]
    ok.sort(key=lambda x: x["best_val_loss"])
    return ok[:k]


def build_cmd(trial: Dict[str, Any], exp_dir: Path, metrics_path: Path) -> List[str]:
    # Keep model_id short; it still identifies the config
    model_id = f"final_{trial['model_id']}"

    cmd = [
        sys.executable, "run.py",
        "--task_name", TASK_NAME,
        "--is_training", "1",
        "--data", "MYDATA",
        "--root_path", str(ROOT_PATH),
        "--data_path", DATA_PATH,

        "--model", trial["model"],
        "--model_id", model_id,

        "--features", FEATURES_MODE,
        "--target", TARGET,
        "--freq", FREQ,

        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(PRED_LEN),

        "--enc_in", str(N_FEATURES),
        "--dec_in", str(N_FEATURES),
        "--c_out", str(N_FEATURES),

        "--train_epochs", str(FINAL_EPOCHS),
        "--batch_size", str(trial["batch_size"]),
        "--learning_rate", str(trial["learning_rate"]),

        "--d_model", str(trial["d_model"]),
        "--e_layers", str(trial["e_layers"]),
        "--dropout", str(trial["dropout"]),

        "--itr", str(ITR),
        "--patience", str(PATIENCE),
        "--num_workers", str(NUM_WORKERS),
        "--des", DES,

        "--run_test", "1",
        "--quiet",

        "--exp_dir", str(exp_dir),
        "--metrics_path", str(metrics_path),
    ]

    if USE_GPU:
        cmd += ["--use_gpu", "--gpu", "0"]
    else:
        cmd += ["--no_use_gpu"]

    return cmd




def run_cmd_to_log(cmd: List[str], cwd: Path, log_file: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_file.open("w", encoding="utf-8") as lf:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        return p.wait()


def main():
    if not SWEEP_RUN_DIR.exists():
        raise FileNotFoundError(f"Missing sweep run dir: {SWEEP_RUN_DIR}")

    if not SWEEP_RESULTS.exists():
        raise FileNotFoundError(f"Missing {SWEEP_RESULTS}. Did you point SWEEP_RUN_DIR correctly?")

    trials = load_jsonl(SWEEP_RESULTS)
    if not trials:
        raise RuntimeError(f"No trials found in {SWEEP_RESULTS}")

    # Choose configs
    if BEST_PER_MODEL:
        selected = pick_best_per_model(trials, require_success=True)
    else:
        selected = pick_top_k_overall(trials, TOP_K_OVERALL)

    if not selected:
        raise RuntimeError("No usable trials selected (check sweep results/return codes).")

    print(f"Final eval will run {len(selected)} configs under: {FINAL_DIR}")

    final_rows: List[Dict[str, Any]] = []

    for i, trial in enumerate(selected, start=1):
        model = trial["model"]
        model_id = trial["model_id"]

        # Each final run gets its own folder
        run_name = f"{model}__{model_id}"
        exp_dir = FINAL_DIR / "runs" / run_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = exp_dir / "metrics.jsonl"
        log_file = LOG_DIR / f"{run_name}.log"

        print(f"[{i}/{len(selected)}] RUN {model} | {model_id}")

        cmd = build_cmd(trial, exp_dir=exp_dir, metrics_path=metrics_path)

        start = time.time()
        rc = run_cmd_to_log(cmd, cwd=TSLIB_DIR, log_file=log_file)
        seconds = time.time() - start

        metrics = read_metrics(metrics_path)

        row = {
            "model": model,
            "sweep_model_id": model_id,
            "final_model_id": f"final_{model_id}",
            "exp_dir": str(exp_dir),
            "metrics_path": str(metrics_path),
            "log_file": str(log_file),
            "returncode": rc,
            "seconds": seconds,

            # sweep hyperparams
            "learning_rate": trial.get("learning_rate"),
            "batch_size": trial.get("batch_size"),
            "d_model": trial.get("d_model"),
            "e_layers": trial.get("e_layers"),
            "dropout": trial.get("dropout"),

            # results from metrics.jsonl
            "best_val_loss": metrics["best_val"],
            "test_mse": metrics["test_mse"],
        }

        final_rows.append(row)

        print(f"    rc={rc}  best_val={row['best_val_loss']}  test_mse={row['test_mse']}  time={seconds:.1f}s")

    FINAL_RESULTS.write_text(json.dumps(final_rows, indent=2), encoding="utf-8")
    print("\nSaved:", FINAL_RESULTS)

    # Ranking
    def rank_key(r: Dict[str, Any]):
        return (
            r["test_mse"] if r["test_mse"] is not None else float("inf"),
            r["best_val_loss"] if r["best_val_loss"] is not None else float("inf"),
        )

    print("\nFinal ranking (by test_mse, then best_val_loss):")
    for r in sorted(final_rows, key=rank_key):
        print(f"  {r['model']:12s}  test_mse={r['test_mse']}  best_val={r['best_val_loss']}  secs={r['seconds']:.1f}")


if __name__ == "__main__":
    main()

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict
import hashlib

import numpy as np

def short_hash(s: str, n: int = 10) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


# -----------------------------
# USER CONFIG
# -----------------------------
TSLIB_DIR = Path(__file__).resolve().parent

ROOT_PATH = Path(r"C:\Users\kasgr\Documents\Masteroppgave\master_repository\Master-s-Thesis\data\processed")
DATA_PATH = "data_karmoy_2024_L42_processed.csv"

N_FEATURES = 90
TASK_NAME = "long_term_forecast"
FEATURES_MODE = "M"
TARGET = "OT"
FREQ = "t"
SEQ_LEN = 48
LABEL_LEN = 24
PRED_LEN = 1

USE_GPU = False
NUM_WORKERS = 0
ITR = 1
SEEDS = [42, 43, 44]
PATIENCE = 5
FINAL_EPOCHS = 50
DES = "final"

WRITE_LOGS = False          # if True -> write .log files for all runs
WRITE_LOGS_ON_FAIL = True   # if True -> only write logs when rc != 0




# --- point this to a specific sweep run folder ---

# Example: TSLIB_DIR / "checkpoints" / "sweeps" / "20260220_115730"
SWEEP_RUN_DIR = TSLIB_DIR / "checkpoints" / "sweeps" / "AMy_M_Linear_Regression_20260226_160718"

# Selection mode
BEST_PER_MODEL = True          # True = one best config per architecture
TOP_K_OVERALL = 1              # used only if BEST_PER_MODEL=False

# -----------------------------
# Paths inside the sweep run
# -----------------------------
SWEEP_RESULTS = SWEEP_RUN_DIR / "sweep_results.jsonl"
# Short output path under checkpoints (NOT inside sweeps/<long>/...)
FINAL_DIR = TSLIB_DIR / "checkpoints" / "final_eval" / SWEEP_RUN_DIR.name
FINAL_DIR.mkdir(parents=True, exist_ok=True)


FINAL_RESULTS = FINAL_DIR / "final_results.json"

LOG_DIR = FINAL_DIR / "logs"
if WRITE_LOGS or WRITE_LOGS_ON_FAIL:
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
    best_val = None
    out = {
        "best_val": None,
        "test_mse": None,
        "test_rmse": None,
        "test_mae": None,
        "test_mape": None,
        "test_smape": None,
        "test_mspe": None,
        "test_dtw": None,
    }

    if not metrics_path.exists():
        return out

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
                for k in list(out.keys()):
                    v = obj.get(k)
                    if isinstance(v, (int, float)):
                        out[k] = float(v)

    out["best_val"] = best_val
    return out



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


def add_if_present(cmd: List[str], trial: Dict[str, Any], key: str, flag: str):
    if key in trial and trial[key] is not None:
        cmd += [flag, str(trial[key])]

def build_cmd(trial: Dict[str, Any], exp_dir: Path, metrics_path: Path, seed: int) -> List[str]:
    # short model id so paths/checkpoints don't get long
    model_id = f"final_{short_hash(trial['model_id'], n=12)}_s{seed}"


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
        "--dropout", str(trial.get("dropout", 0.0)),

        "--itr", str(ITR),
        "--seed", str(seed),
        "--patience", str(PATIENCE),
        "--num_workers", str(NUM_WORKERS),

        "--des", DES,

        "--run_test", "1",
        "--quiet",

        "--exp_dir", str(exp_dir),
        "--metrics_path", str(metrics_path),
        "--save_test_outputs",
    ]

    # Optional regularization (if present in sweep results)
    add_if_present(cmd, trial, "weight_decay", "--weight_decay")

    # Transformer-ish (ONLY if present)
    add_if_present(cmd, trial, "d_model", "--d_model")
    add_if_present(cmd, trial, "e_layers", "--e_layers")
    add_if_present(cmd, trial, "d_layers", "--d_layers")
    add_if_present(cmd, trial, "n_heads", "--n_heads")
    add_if_present(cmd, trial, "d_ff", "--d_ff")

    # TimeMixer-specific (ONLY if present)
    add_if_present(cmd, trial, "down_sampling_method", "--down_sampling_method")
    add_if_present(cmd, trial, "down_sampling_layers", "--down_sampling_layers")
    add_if_present(cmd, trial, "down_sampling_window", "--down_sampling_window")
    add_if_present(cmd, trial, "channel_independence", "--channel_independence")

    # LSTM-specific (ONLY if present)
    add_if_present(cmd, trial, "d_mark", "--d_mark")
    add_if_present(cmd, trial, "lstm_hidden", "--lstm_hidden")
    add_if_present(cmd, trial, "lstm_layers", "--lstm_layers")

    if USE_GPU:
        cmd += ["--use_gpu", "--gpu", "0"]
    else:
        cmd += ["--no_use_gpu"]

    return cmd



def rank_key(r: Dict[str, Any]):
    return (
        r.get("test_mse_median") if r.get("test_mse_median") is not None else float("inf"),
        r.get("test_mse_mean") if r.get("test_mse_mean") is not None else float("inf"),
    )
def run_cmd_maybe_log(cmd: List[str], cwd: Path, log_file: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if not WRITE_LOGS and not WRITE_LOGS_ON_FAIL:
        # completely silent
        p = subprocess.Popen(cmd, cwd=str(cwd),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             text=True, env=env)
        return p.wait()

    # stream to a temp file
    tmp_log = log_file.with_suffix(log_file.suffix + ".tmp")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with tmp_log.open("w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, cwd=str(cwd),
                             stdout=lf, stderr=subprocess.STDOUT,
                             text=True, env=env)
        rc = p.wait()

    if rc == 0 and WRITE_LOGS_ON_FAIL and not WRITE_LOGS:
        # success: delete temp log
        try:
            tmp_log.unlink()
        except Exception:
            pass
    else:
        # keep log (rename tmp -> final)
        try:
            if log_file.exists():
                log_file.unlink()
            tmp_log.rename(log_file)
        except Exception:
            pass

    return rc


def mean_std(vals: List[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return {"mean": None, "std": None, "median": None}
    arr = np.array(xs, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "median": float(np.median(arr))}



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

    seed_rows: List[Dict[str, Any]] = []   # store per-seed runs
    agg_rows: List[Dict[str, Any]] = []    # store aggregated per config

    for i, trial in enumerate(selected, start=1):
        model = trial["model"]
        model_id = trial["model_id"]

        run_tag = short_hash(model_id)  # stable short id based on sweep model_id
        cfg_dir = FINAL_DIR / model / run_tag
        cfg_dir.mkdir(parents=True, exist_ok=True)

        # store full sweep trial info once
        (cfg_dir / "trial.json").write_text(json.dumps(trial, indent=2), encoding="utf-8")

        print(f"[{i}/{len(selected)}] CONFIG {model} | {model_id} | tag={run_tag}")

        # run each seed into its own folder
        per_seed_metrics: List[Dict[str, Any]] = []
        per_seed_seconds: List[float] = []

        for seed in SEEDS:
            exp_dir = cfg_dir / f"seed{seed}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = exp_dir / "metrics.jsonl"
            log_file = LOG_DIR / f"{model}_{run_tag}_seed{seed}.log"

            print(f"    -> seed {seed}  exp_dir={exp_dir}")

            cmd = build_cmd(trial, exp_dir=exp_dir, metrics_path=metrics_path, seed=seed)

            start = time.time()
            rc = run_cmd_maybe_log(cmd, cwd=TSLIB_DIR, log_file=log_file)

            seconds = time.time() - start

            m = read_metrics(metrics_path)

            seed_row = {
                "model": model,
                "sweep_model_id": model_id,
                "config_tag": run_tag,
                "seed": seed,

                "exp_dir": str(exp_dir),
                "metrics_path": str(metrics_path),
                "log_file": str(log_file) if ((WRITE_LOGS) or (WRITE_LOGS_ON_FAIL and rc != 0)) else None,

                "returncode": rc,
                "seconds": seconds,

                # hyperparams
                "learning_rate": trial.get("learning_rate"),
                "batch_size": trial.get("batch_size"),
                "dropout": trial.get("dropout"),
                "weight_decay": trial.get("weight_decay"),
                "lstm_hidden": trial.get("lstm_hidden"),
                "lstm_layers": trial.get("lstm_layers"),
                "d_mark": trial.get("d_mark"),

                # metrics
                "best_val_loss": m.get("best_val"),
                "test_mse": m.get("test_mse"),
                "test_rmse": m.get("test_rmse"),
                "test_mae": m.get("test_mae"),
                "test_mape": m.get("test_mape"),
                "test_smape": m.get("test_smape"),
                "test_mspe": m.get("test_mspe"),
                "test_dtw": m.get("test_dtw"),
            }

            seed_rows.append(seed_row)
            per_seed_metrics.append(seed_row)
            per_seed_seconds.append(seconds)

            print(f"       rc={rc}  test_mse={seed_row['test_mse']}  best_val={seed_row['best_val_loss']}  secs={seconds:.1f}")

        # aggregate across seeds (ignore missing/failed metrics)
        def collect(key: str) -> List[Optional[float]]:
            return [r.get(key) for r in per_seed_metrics if r.get("returncode") == 0]

        agg = {
            "model": model,
            "sweep_model_id": model_id,
            "config_tag": run_tag,
            "config_dir": str(cfg_dir),

            # hyperparams (same for all seeds)
            "learning_rate": trial.get("learning_rate"),
            "batch_size": trial.get("batch_size"),
            "dropout": trial.get("dropout"),
            "weight_decay": trial.get("weight_decay"),
            "lstm_hidden": trial.get("lstm_hidden"),
            "lstm_layers": trial.get("lstm_layers"),
            "d_mark": trial.get("d_mark"),

            "n_seeds": len(SEEDS),
            "n_success": sum(1 for r in per_seed_metrics if r.get("returncode") == 0),
        }

        for k in ["test_mse", "test_rmse", "test_mae", "test_mape", "test_smape", "test_mspe", "test_dtw", "best_val_loss"]:
            stats = mean_std(collect(k))
            agg[f"{k}_mean"] = stats["mean"]
            agg[f"{k}_std"] = stats["std"]
            agg[f"{k}_median"] = stats["median"]

        agg["seconds_mean"] = float(np.mean(per_seed_seconds)) if per_seed_seconds else None
        agg_rows.append(agg)

    # write per-seed and aggregated results
    (FINAL_DIR / "seed_results.json").write_text(json.dumps(seed_rows, indent=2), encoding="utf-8")
    FINAL_RESULTS.write_text(json.dumps(agg_rows, indent=2), encoding="utf-8")
    print("\nSaved:")
    print("  ", FINAL_DIR / "seed_results.json")
    print("  ", FINAL_RESULTS)



    print("\nFinal ranking (by median test_mse, then mean test_mse):")
    for r in sorted(agg_rows, key=rank_key):
        print(
            f"  {r['model']:18s} "
            f"median={r.get('test_mse_median')}  mean={r.get('test_mse_mean')}±{r.get('test_mse_std')}  "
            f"n_ok={r.get('n_success')}/{r.get('n_seeds')}  tag={r.get('config_tag')}"
        )





if __name__ == "__main__":
    main()

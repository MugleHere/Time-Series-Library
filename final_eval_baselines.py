import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# -----------------------------
# USER CONFIG (match final_eval.py!)
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
DES = "final_baselines"

BASELINE_MODELS = ["AMy_Average_baseline", "AMy_Naive_baseline"]

# Logging behavior (same idea as final_eval.py)
WRITE_LOGS = False          # if True -> write .log files for all runs
WRITE_LOGS_ON_FAIL = True   # if True -> only write logs when rc != 0

# Where to write outputs
OUT_DIR = TSLIB_DIR / "checkpoints" / "final_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUT_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

RUNS_DIR = RUN_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = RUN_DIR / "logs"
if WRITE_LOGS or WRITE_LOGS_ON_FAIL:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

SEED_RESULTS = RUN_DIR / "seed_results.json"   # kept for schema compatibility (1 "seed" per model)
FINAL_RESULTS = RUN_DIR / "final_results.json"


# -----------------------------
# Helpers
# -----------------------------
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


def mean_std(vals: List[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return {"mean": None, "std": None, "median": None}
    arr = np.array(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def run_cmd_maybe_log(cmd: List[str], cwd: Path, log_file: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if not WRITE_LOGS and not WRITE_LOGS_ON_FAIL:
        p = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            text=True, env=env
        )
        return p.wait()

    tmp_log = log_file.with_suffix(log_file.suffix + ".tmp")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with tmp_log.open("w", encoding="utf-8") as lf:
        p = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=lf, stderr=subprocess.STDOUT,
            text=True, env=env
        )
        rc = p.wait()

    if rc == 0 and WRITE_LOGS_ON_FAIL and not WRITE_LOGS:
        # success: delete temp log
        try:
            tmp_log.unlink()
        except Exception:
            pass
    else:
        # keep log
        try:
            if log_file.exists():
                log_file.unlink()
            tmp_log.rename(log_file)
        except Exception:
            pass

    return rc


def build_baseline_cmd(model: str, model_id: str, exp_dir: Path, metrics_path: Path) -> List[str]:
    cmd = [
        sys.executable, "run.py",
        "--task_name", TASK_NAME,

        # Baseline = test-only
        "--is_training", "0",
        "--run_test", "1",

        "--data", "MYDATA",
        "--root_path", str(ROOT_PATH),
        "--data_path", DATA_PATH,

        "--model", model,
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

        # parser expects these; baselines ignore them
        "--batch_size", "32",
        "--learning_rate", "0.001",
        "--d_model", "256",
        "--e_layers", "1",
        "--dropout", "0.0",

        "--itr", str(ITR),
        "--num_workers", str(NUM_WORKERS),
        "--des", DES,

        # IMPORTANT: no --seed here (removed)

        "--quiet",
        "--exp_dir", str(exp_dir),
        "--metrics_path", str(metrics_path),

        "--baseline_mode",
    ]

    if USE_GPU:
        cmd += ["--use_gpu", "--gpu", "0"]
    else:
        cmd += ["--no_use_gpu"]

    return cmd


def main():
    print(f"Baseline eval run: {RUN_ID}")
    print(f"Outputs: {RUN_DIR}")

    seed_rows: List[Dict[str, Any]] = []
    agg_rows: List[Dict[str, Any]] = []

    for i, model in enumerate(BASELINE_MODELS, start=1):
        # Match final_eval layout: runs/<model>/<tag>
        run_tag = model  # stable tag (no hyperparams)
        cfg_dir = RUNS_DIR / model / run_tag
        cfg_dir.mkdir(parents=True, exist_ok=True)

        trial_info = {
            "model": model,
            "config_tag": run_tag,
            "task_name": TASK_NAME,
            "data_path": DATA_PATH,
            "seq_len": SEQ_LEN,
            "label_len": LABEL_LEN,
            "pred_len": PRED_LEN,
            "features": FEATURES_MODE,
            "target": TARGET,
            "freq": FREQ,
            "n_features": N_FEATURES,
            "baseline_mode": True,
            "seeded": False,
        }
        (cfg_dir / "trial.json").write_text(json.dumps(trial_info, indent=2), encoding="utf-8")

        print(f"[{i}/{len(BASELINE_MODELS)}] MODEL {model} | tag={run_tag}")

        # single run (no seeds)
        exp_dir = cfg_dir / "run"
        exp_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = exp_dir / "metrics.jsonl"
        log_file = LOG_DIR / f"{model}_{run_tag}.log"
        model_id = f"baseline_{model}"

        cmd = build_baseline_cmd(
            model=model,
            model_id=model_id,
            exp_dir=exp_dir,
            metrics_path=metrics_path,
        )

        print(f"    -> run  exp_dir={exp_dir}")
        start = time.time()
        rc = run_cmd_maybe_log(cmd, cwd=TSLIB_DIR, log_file=log_file)
        seconds = time.time() - start

        m = read_metrics(metrics_path)

        # keep seed_results.json schema-compatible: just one entry, seed=None
        seed_row = {
            "model": model,
            "sweep_model_id": None,
            "config_tag": run_tag,
            "seed": None,

            "exp_dir": str(exp_dir),
            "metrics_path": str(metrics_path),
            "log_file": str(log_file),

            "returncode": rc,
            "seconds": seconds,

            # hyperparams (none for baselines)
            "learning_rate": None,
            "batch_size": None,
            "dropout": None,
            "weight_decay": None,
            "lstm_hidden": None,
            "lstm_layers": None,
            "d_mark": None,

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

        print(f"       rc={rc}  test_mse={seed_row['test_mse']}  secs={seconds:.1f}")

        # agg row: treat as 1 "seed"
        agg = {
            "model": model,
            "sweep_model_id": None,
            "config_tag": run_tag,
            "config_dir": str(cfg_dir),

            # hyperparams
            "learning_rate": None,
            "batch_size": None,
            "dropout": None,
            "weight_decay": None,
            "lstm_hidden": None,
            "lstm_layers": None,
            "d_mark": None,

            "n_seeds": 1,
            "n_success": 1 if rc == 0 else 0,
        }

        for k in ["test_mse", "test_rmse", "test_mae", "test_mape", "test_smape", "test_mspe", "test_dtw", "best_val_loss"]:
            stats = mean_std([seed_row.get(k)] if rc == 0 else [])
            agg[f"{k}_mean"] = stats["mean"]
            agg[f"{k}_std"] = stats["std"]
            agg[f"{k}_median"] = stats["median"]

        agg["seconds_mean"] = seconds
        agg_rows.append(agg)

    SEED_RESULTS.write_text(json.dumps(seed_rows, indent=2), encoding="utf-8")
    FINAL_RESULTS.write_text(json.dumps(agg_rows, indent=2), encoding="utf-8")

    print("\nSaved:")
    print("  ", SEED_RESULTS)
    print("  ", FINAL_RESULTS)

    print("\nFinal ranking (by median test_mse, then mean test_mse):")

    def rank_key(r: Dict[str, Any]):
        return (
            r.get("test_mse_median") if r.get("test_mse_median") is not None else float("inf"),
            r.get("test_mse_mean") if r.get("test_mse_mean") is not None else float("inf"),
        )

    for r in sorted(agg_rows, key=rank_key):
        print(
            f"  {r['model']:24s} "
            f"median={r.get('test_mse_median')}  mean={r.get('test_mse_mean')}±{r.get('test_mse_std')}  "
            f"n_ok={r.get('n_success')}/{r.get('n_seeds')}  tag={r.get('config_tag')}"
        )


if __name__ == "__main__":
    main()

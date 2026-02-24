import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# -----------------------------
# USER CONFIG (match final_eval.py!)
# -----------------------------
TSLIB_DIR = Path(__file__).resolve().parent

ROOT_PATH = Path(r"C:\Users\kasgr\Documents\Masteroppgave\master_repository\Master-s-Thesis\data\processed")
DATA_PATH = "parquet_data_karmoy_2024_L42_processed.csv"

N_FEATURES = 91
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

# Where to write outputs
OUT_DIR = TSLIB_DIR / "checkpoints" / "final_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUT_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

RUNS_DIR = RUN_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = RUN_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FINAL_RESULTS = RUN_DIR / "final_results.json"


# -----------------------------
# Helpers
# -----------------------------
def read_metrics(metrics_path: Path) -> Dict[str, Optional[float]]:
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


def build_baseline_cmd(model: str, exp_dir: Path, metrics_path: Path) -> List[str]:
    # Dummy params included only because your args parser expects them.
    # Baselines will ignore d_model/e_layers/dropout.
    cmd = [
        sys.executable, "run.py",
        "--task_name", TASK_NAME,

        # IMPORTANT: test-only (no training)
        "--is_training", "0",
        "--run_test", "1",

        "--data", "MYDATA",
        "--root_path", str(ROOT_PATH),
        "--data_path", DATA_PATH,

        "--model", model,
        "--model_id", f"baseline_{model}",

        "--features", FEATURES_MODE,
        "--target", TARGET,
        "--freq", FREQ,

        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(PRED_LEN),

        "--enc_in", str(N_FEATURES),
        "--dec_in", str(N_FEATURES),
        "--c_out", str(N_FEATURES),

        "--batch_size", "32",
        "--learning_rate", "0.001",
        "--d_model", "256",
        "--e_layers", "1",
        "--dropout", "0.0",

        "--itr", str(ITR),
        "--num_workers", str(NUM_WORKERS),
        "--des", DES,

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
    rows: List[Dict[str, Any]] = []
    print(f"Baseline eval run: {RUN_ID}")
    print(f"Outputs: {RUN_DIR}")

    for i, model in enumerate(BASELINE_MODELS, start=1):
        run_name = f"{model}__testonly"
        exp_dir = RUNS_DIR / run_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = exp_dir / "metrics.jsonl"
        log_file = LOG_DIR / f"{run_name}.log"

        cmd = build_baseline_cmd(model, exp_dir=exp_dir, metrics_path=metrics_path)

        print(f"[{i}/{len(BASELINE_MODELS)}] RUN {model}")
        start = time.time()
        rc = run_cmd_to_log(cmd, cwd=TSLIB_DIR, log_file=log_file)
        seconds = time.time() - start

        metrics = read_metrics(metrics_path)

        row = {
            "model": model,
            "model_id": f"baseline_{model}",
            "exp_dir": str(exp_dir),
            "metrics_path": str(metrics_path),
            "log_file": str(log_file),
            "returncode": rc,
            "seconds": seconds,
            "test_mse": metrics["test_mse"],
        }
        rows.append(row)
        print(f"    rc={rc} test_mse={row['test_mse']} time={seconds:.1f}s")

    FINAL_RESULTS.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Saved:", FINAL_RESULTS)


if __name__ == "__main__":
    main()

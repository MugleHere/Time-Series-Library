import os
import sys
import json
import time
import itertools
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


# -----------------------------
# USER CONFIG
# -----------------------------
TSLIB_DIR = Path(__file__).resolve().parent  # folder containing run.py

#ROOT_PATH = Path(r"C:\Users\kasgr\Documents\Masteroppgave\master_repository\Master-s-Thesis\third_party\tslib\data_")
# VM ubuntu
ROOT_PATH = Path("/home/ubuntu/Time-Series-Library/data_")
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
PATIENCE = 2
EPOCHS = 10  # sweep epochs
HEARTBEAT_SEC = 30  # how often sweep prints progress

MODELS = ["AMy_lstm"] #, "AMy_M_Linear_Regression"]
#MODELS  ["TimesNet", "DLinear", "TimeXer, "TimeMixer"]  # ADVANCED MODELS

LEARNING_RATES = [1e-5]#, 5e-4, 1e-3]
BATCH_SIZES = [16]#, 32]
DROPOUTS = [0.0]
WEIGHT_DECAYS = [1e-5]

DEFAULT_GRID = {
    "learning_rate": LEARNING_RATES,
    "batch_size": BATCH_SIZES,
    "dropout": DROPOUTS,
    "weight_decay": WEIGHT_DECAYS,
}

MODEL_GRIDS = {
    "TimeMixer": {
        **DEFAULT_GRID,
        "down_sampling_method": ["avg"],
        "down_sampling_layers": [1, 2],
        "down_sampling_window": [2],
        "channel_independence": [0],
    },
    "Informer": {
        **DEFAULT_GRID,
    },
    "DLinear": {
        **DEFAULT_GRID,
    },
    "AMy_lstm": {
        **DEFAULT_GRID,
        "d_mark": [5],                # set to your stamp dim (you printed 5)
        "lstm_hidden": [64],#, 128, 256],
        "lstm_layers": [1],     
    },
}
def short_name(name: str) -> str:
    return name.replace("AMy_", "").replace("_", "")



# Output root
OUT_DIR = TSLIB_DIR / "checkpoints" / "sweeps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_tag = "_".join(MODELS)
RUN_ID = f"{model_tag}_{time.strftime('%Y%m%d_%H%M%S')}"

RUN_DIR = OUT_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_DIR = RUN_DIR / "trials"
TRIALS_DIR.mkdir(parents=True, exist_ok=True)

FAIL_DIR = RUN_DIR / "failures"
FAIL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSONL = RUN_DIR / "sweep_results.jsonl"
RESULTS_SUMMARY = RUN_DIR / "sweep_summary.json"

# Make TSLib separate runs (still useful even though we isolate with exp_dir)
DES = f"sweep_{RUN_ID}"

LOG_DIR = RUN_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)



# -----------------------------
# Helpers
# -----------------------------
def fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds == float("inf"):
        return "?"
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"


def safe_read_last_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    """Read last JSON object from a JSONL file efficiently."""
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end == 0:
                return None
            size = min(16384, end)
            f.seek(-size, os.SEEK_END)
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def read_best_val_from_metrics(metrics_path: Path) -> Optional[float]:
    """Scan metrics.jsonl and return the best_val from the last epoch_end (fast enough; files are small)."""
    if not metrics_path.exists():
        return None
    best = None
    try:
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
                        best = float(bv)
        return best
    except Exception:
        return best


def write_failure_tail(trial_id: str, metrics_path: Path) -> Optional[str]:
    """If a trial fails, save last few metrics lines (or nothing if file missing)."""
    if not metrics_path.exists():
        return None
    try:
        lines = metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = "\n".join(lines[-60:])
        out = FAIL_DIR / f"{trial_id}_metrics_tail.txt"
        out.write_text(tail, encoding="utf-8")
        return str(out)
    except Exception:
        return None


def build_trial_id(model: str, params: dict) -> str:
    base = (
        f"{model}"
        f"_lr{params['learning_rate']}"
        f"_bs{params['batch_size']}"
        f"_do{params.get('dropout', 0.0)}"
        f"_wd{params.get('weight_decay', 0.0)}"
        f"_pl{PRED_LEN}"
        f"_sl{SEQ_LEN}"
    )
    if model == "TimeMixer":
        base += (
            f"_dsm{params['down_sampling_method']}"
            f"_dsl{params['down_sampling_layers']}"
            f"_dsw{params['down_sampling_window']}"
            f"_ci{params.get('channel_independence', 0)}"
        )
    if model == "AMy_lstm":  
        base += f"_hm{params['lstm_hidden']}_ly{params['lstm_layers']}_dm{params['d_mark']}"
    return base






@dataclass
class Trial:
    # always-set sweep params
    model: str
    model_id: str
    learning_rate: float
    batch_size: int
    dropout: float
    exp_dir: str
    metrics_path: str

    # TimeMixer-only (defaults so LinearRegression doesn’t care)
    down_sampling_method: str = "avg"
    down_sampling_layers: int = 0
    down_sampling_window: int = 1
    channel_independence: int = 0

    # results/runtime
    best_val_loss: Optional[float] = None
    returncode: int = -1
    seconds: float = 0.0
    fail_tail_file: Optional[str] = None

    d_mark: int = 5
    lstm_hidden: int = 128
    lstm_layers: int = 1
    weight_decay: float = 0.0








def build_cmd(trial: Trial) -> list[str]:
    cmd = [
        sys.executable, "run.py",
        "--task_name", TASK_NAME,
        "--is_training", "1",
        "--data", "MYDATA",
        "--root_path", str(ROOT_PATH),
        "--data_path", DATA_PATH,

        "--model", trial.model,
        "--model_id", trial.model_id,

        "--features", FEATURES_MODE,
        "--target", TARGET,
        "--freq", FREQ,

        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(PRED_LEN),

        "--enc_in", str(N_FEATURES),
        "--dec_in", str(N_FEATURES),
        "--c_out", str(N_FEATURES),

        "--train_epochs", str(EPOCHS),
        "--batch_size", str(trial.batch_size),
        "--learning_rate", str(trial.learning_rate),

        "--dropout", str(trial.dropout),

        "--itr", str(ITR),
        "--patience", str(PATIENCE),
        "--num_workers", str(NUM_WORKERS),

        "--des", DES,
        "--run_test", "0",

        "--quiet",
        "--exp_dir", trial.exp_dir,
        "--metrics_path", trial.metrics_path,

        "--weight_decay", str(trial.weight_decay),
    ]


    if trial.model == "TimeMixer":
        cmd += [
            "--down_sampling_method", trial.down_sampling_method,
            "--down_sampling_layers", str(trial.down_sampling_layers),
            "--down_sampling_window", str(trial.down_sampling_window),
            "--channel_independence", str(trial.channel_independence),
        ]
    if trial.model == "AMy_lstm":
        cmd += [
            "--d_mark", str(trial.d_mark),
            "--lstm_hidden", str(trial.lstm_hidden),
            "--lstm_layers", str(trial.lstm_layers),
        ]



    if USE_GPU:
        cmd += ["--use_gpu", "--gpu", "0"]
    else:
        cmd += ["--no_use_gpu"]

    return cmd




def run_trial_with_heartbeat(cmd: List[str], cwd: Path, metrics_path: Path,
                             trial_index: int, total_trials: int,
                             avg_trial_sec: Optional[float]) -> Tuple[int, float, Optional[float]]:
    """
    Run run.py quietly. While running, every HEARTBEAT_SEC, read metrics.jsonl last line and print progress.
    Returns: (rc, seconds, best_val_loss)
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    last_beat = 0.0

    log_file = LOG_DIR / f"{metrics_path.parent.name}.log"
    with log_file.open("w", encoding="utf-8") as lf:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )


    # Loop until process ends
    while True:
        rc = p.poll()
        now = time.time()
        if rc is not None:
            break

        if now - last_beat >= HEARTBEAT_SEC:
            last_beat = now
            elapsed = now - start

            last = safe_read_last_jsonl(metrics_path)
            epoch = None
            best_val = None
            loss = None

            if last:
                epoch = last.get("epoch")
                best_val = last.get("best_val")
                loss = last.get("loss")

            eta_total = None
            if avg_trial_sec is not None:
                remaining_trials = total_trials - trial_index
                eta_total = remaining_trials * avg_trial_sec

            msg = f"[{trial_index:03d}/{total_trials:03d}] running  elapsed={fmt_hms(elapsed)}"
            if epoch is not None:
                msg += f"  epoch={epoch}/{EPOCHS}"
            if isinstance(best_val, (int, float)):
                msg += f"  best_val={best_val:.6f}"
            elif isinstance(loss, (int, float)):
                msg += f"  loss={loss:.6f}"
            if eta_total is not None:
                msg += f"  ETA_total~{fmt_hms(eta_total)}"

            print(msg)

        time.sleep(1.0)


    seconds = time.time() - start
    best_val_loss = read_best_val_from_metrics(metrics_path)
    return rc, seconds, best_val_loss

def trial_to_record(trial: Trial) -> dict:
    base = {
        "model": trial.model,
        "model_id": trial.model_id,
        "learning_rate": trial.learning_rate,
        "batch_size": trial.batch_size,
        "dropout": trial.dropout,
        "weight_decay": trial.weight_decay,
        "exp_dir": trial.exp_dir,
        "metrics_path": trial.metrics_path,
        "best_val_loss": trial.best_val_loss,
        "returncode": trial.returncode,
        "seconds": trial.seconds,
        "fail_tail_file": trial.fail_tail_file,
    }

    if trial.model == "TimeMixer":
        base.update({
            "down_sampling_method": trial.down_sampling_method,
            "down_sampling_layers": trial.down_sampling_layers,
            "down_sampling_window": trial.down_sampling_window,
            "channel_independence": trial.channel_independence,
        })

    if trial.model == "AMy_lstm":  
        base.update({
            "d_mark": trial.d_mark,
            "lstm_hidden": trial.lstm_hidden,
            "lstm_layers": trial.lstm_layers,
        })

    return base



def grid_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    combos = itertools.product(*[grid[k] for k in keys])
    return [dict(zip(keys, values)) for values in combos]

def main():
    trials: list[Trial] = []

    for model in MODELS:
        grid = MODEL_GRIDS.get(model, DEFAULT_GRID)

        combos = grid_combinations(grid)

        for params in combos:
            trial_id = build_trial_id(model, params)

            exp_dir = str(TRIALS_DIR / trial_id)
            metrics_path = str(Path(exp_dir) / "metrics.jsonl")

            # Create Trial with shared fields + model-specific params (with defaults)
            t = Trial(
                model=model,
                model_id=trial_id,
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                dropout=params.get("dropout", 0.0),
                exp_dir=exp_dir,
                metrics_path=metrics_path,
            )
            t.weight_decay = params.get("weight_decay", 0.0)

            if model == "AMy_lstm":  
                t.d_mark = params["d_mark"]
                t.lstm_hidden = params["lstm_hidden"]
                t.lstm_layers = params["lstm_layers"]

            if model == "TimeMixer":
                t.down_sampling_method = params["down_sampling_method"]
                t.down_sampling_layers = params["down_sampling_layers"]
                t.down_sampling_window = params["down_sampling_window"]
                t.channel_independence = params.get("channel_independence", 0)


            trials.append(t)

    # then loop through trials exactly as you already do


    total = len(trials)
    print(f"Sweep run: {RUN_ID}")
    print(f"Trials: {total}")
    print(f"Results: {RESULTS_JSONL}")

    durations: List[float] = []

    with open(RESULTS_JSONL, "w", encoding="utf-8") as f_jsonl:
        for i, trial in enumerate(trials, start=1):
            Path(trial.exp_dir).mkdir(parents=True, exist_ok=True)
            metrics_path = Path(trial.metrics_path)

            avg_trial_sec = (sum(durations) / len(durations)) if durations else None

            print(f"\n[{i:03d}/{total:03d}] START {trial.model} | {trial.model_id}")

            cmd = build_cmd(trial)

            rc, secs, best_val = run_trial_with_heartbeat(
                cmd=cmd,
                cwd=TSLIB_DIR,
                metrics_path=metrics_path,
                trial_index=i,
                total_trials=total,
                avg_trial_sec=avg_trial_sec,
            )

            trial.returncode = rc
            trial.seconds = secs
            trial.best_val_loss = best_val
            durations.append(secs)

            if rc != 0:
                trial.fail_tail_file = write_failure_tail(trial.model_id, metrics_path)

            print(f"[{i:03d}/{total:03d}] DONE  rc={rc}  best_val={best_val}  time={fmt_hms(secs)}")

            f_jsonl.write(json.dumps(trial_to_record(trial)) + "\n")

            f_jsonl.flush()

    # Summarize
    rows = [json.loads(line) for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in rows if r.get("returncode") == 0 and r.get("best_val_loss") is not None]
    ok_sorted = sorted(ok, key=lambda r: r["best_val_loss"])

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": RUN_ID,
        "total_trials": total,
        "successful_trials": len(ok_sorted),

        # all trials as-run (includes failures)
        "all_trials": rows,

        # all successful trials sorted by val loss
        "all_success_by_val": ok_sorted,
    }
    summary["best_overall"] = ok_sorted[0] if ok_sorted else None
    summary["best_per_model"] = {}
    for r in ok_sorted:
        m = r["model"]
        if m not in summary["best_per_model"]:
            summary["best_per_model"][m] = r



    RESULTS_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved summary:", RESULTS_SUMMARY)
    print("Top 5 by validation loss:")
    for r in ok_sorted[:5]:
        print(f"  {r['model']:12s}  val={r['best_val_loss']:.6f}  id={r['model_id']}")

    print(f"\nAll outputs for this sweep are under:\n  {RUN_DIR}")


if __name__ == "__main__":
    main()

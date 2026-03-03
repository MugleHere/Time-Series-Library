import json
import time
from pathlib import Path

import optuna

import run
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


TSLIB_DIR = Path(__file__).resolve().parent
OUT_DIR = TSLIB_DIR / "checkpoints" / "optuna"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Fixed experiment settings (yours) ----
COMMON = dict(
    task_name="long_term_forecast",
    is_training=1,
    data="MYDATA",
    root_path="/home/ubuntu/Time-Series-Library/data_",
    data_path="data_karmoy_2024_L42_processed.csv",

    features="M",
    target="OT",
    freq="t",
    seq_len=48,
    label_len=24,
    pred_len=1,

    enc_in=90,
    dec_in=90,
    c_out=90,

    itr=1,
    patience=2,
    train_epochs=10,   # tuning epochs
    num_workers=0,

    run_test=0,
    quiet=True,

    # Force CPU
    no_use_gpu=True,
)

#MODELS = ["AMy_lstm"]  # add more like "TimeMixer", "DLinear", ...
MODELS = ["DLinear", "TimeXer", "TimeMixer", "AMy_M_Linear_Regression", "FEDformer", "PatchTST", "iTransformer","Nonstationary_Transformer"]  #Removed "TimesNet" because of speed


def suggest_params(trial: optuna.Trial, model: str) -> dict:
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]), #Removed 8 as was in the four first models
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
    }

    if model == "AMy_lstm":
        p.update({
            "d_mark": 5,
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 64, 128, 256]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        })


    if model == "TimeMixer":
        p.update({
            "down_sampling_method": "avg",
            "down_sampling_layers": 1,
            "down_sampling_window": 2,
            "channel_independence": 0,
        })


    return p
import traceback
def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def make_args(model: str, model_id: str, exp_dir: Path, metrics_path: Path, params: dict):
    args_list = []

    # common args -> CLI list
    for k, v in COMMON.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            # these are flags in your argparse; include if True
            if v:
                args_list.append(flag)
        else:
            args_list += [flag, str(v)]

    args_list += ["--model", model, "--model_id", model_id]
    args_list += ["--exp_dir", str(exp_dir), "--metrics_path", str(metrics_path)]
    args_list += ["--des", "optuna"]

    # hyperparams
    args_list += ["--learning_rate", str(params["learning_rate"])]
    args_list += ["--batch_size", str(params["batch_size"])]
    args_list += ["--dropout", str(params["dropout"])]
    args_list += ["--weight_decay", str(params["weight_decay"])]

    if model == "AMy_lstm":
        args_list += ["--d_mark", str(params["d_mark"])]
        args_list += ["--lstm_hidden", str(params["lstm_hidden"])]
        args_list += ["--lstm_layers", str(params["lstm_layers"])]

    if model == "TimeMixer":
        args_list += ["--down_sampling_method", str(params["down_sampling_method"])]
        args_list += ["--down_sampling_layers", str(params["down_sampling_layers"])]
        args_list += ["--down_sampling_window", str(params["down_sampling_window"])]
        args_list += ["--channel_independence", str(params["channel_independence"])]

    return run.parse_args(args_list)


def read_best_val(metrics_path: Path) -> float | None:
    best = None
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("event") == "epoch_end":
                bv = obj.get("best_val")
                if isinstance(bv, (int, float)):
                    best = float(bv)
    return best


def tune_model(model: str, n_trials: int = 50):
    model_dir = OUT_DIR / model
    trials_dir = model_dir / "trials"
    model_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    FAIL_JSONL = model_dir / "failures.jsonl"

    storage = f"sqlite:///{(model_dir / 'optuna.db').as_posix()}"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model}_tune",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3,
            max_resource=COMMON["train_epochs"],
            reduction_factor=3,
        ),
    )

    def objective(trial: optuna.Trial):
        params = suggest_params(trial, model)

        run_id = f"trial_{trial.number:05d}_{int(time.time())}"
        exp_dir = trials_dir / run_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = exp_dir / "metrics.jsonl"
        args = make_args(model, run_id, exp_dir, metrics_path, params)

        exp = Exp_Long_Term_Forecast(args)

        def epoch_cb(epoch: int, m: dict):
            val = float(m["val_loss"])
            trial.report(val, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        try:
            exp.train(setting=f"{model}_{run_id}", epoch_cb=epoch_cb)
            best_val = read_best_val(metrics_path)
            if best_val is None:
                raise optuna.exceptions.TrialPruned()
            return best_val

        except optuna.exceptions.TrialPruned:
            # Record prune reason (optional: read last metrics line)
            append_jsonl(FAIL_JSONL, {
                "event": "pruned",
                "model": model,
                "trial_number": trial.number,
                "run_id": run_id,
                "params": params,
                "metrics_path": str(metrics_path),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            raise

        except KeyboardInterrupt:
            # Let study.optimize(catch=...) stop cleanly
            raise

        except Exception as e:
            tb = traceback.format_exc()
            # Save traceback inside trial folder too
            (exp_dir / "exception.txt").write_text(tb, encoding="utf-8")

            append_jsonl(FAIL_JSONL, {
                "event": "failed",
                "model": model,
                "trial_number": trial.number,
                "run_id": run_id,
                "params": params,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback_file": str(exp_dir / "exception.txt"),
                "metrics_path": str(metrics_path),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

            # For known “bad config” errors, prune instead of killing the whole run
            msg = str(e)
            if isinstance(e, (RuntimeError, IndexError)) and (
                "Output size is too small" in msg
                or "Calculated output size" in msg
                or "list index out of range" in msg
            ):
                raise optuna.exceptions.TrialPruned()

            # Otherwise mark as failed trial but continue (Optuna treats exception as FAIL)
            raise

    
    # Count finished trials (COMPLETE, PRUNED, FAIL)
    done = sum(t.state.is_finished() for t in study.trials)
    remaining = max(0, n_trials - done)

    print(f"[{model}] trials done={done} target={n_trials} remaining={remaining}")

    if remaining <= 0:
        # Still write best_params.json in case it doesn't exist yet
        best = {"best_value": study.best_value, "best_params": study.best_params}
        (model_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"[{model}] already complete. saved: {model_dir / 'best_params.json'}")
        return


    study.optimize(objective, n_trials=remaining,catch=(KeyboardInterrupt,))

    best = {"best_value": study.best_value, "best_params": study.best_params}
    (model_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print(f"\n[{model}] BEST val={study.best_value:.6f}")
    print(f"[{model}] saved: {model_dir / 'best_params.json'}")


def main():
    for m in MODELS:
        tune_model(m, n_trials=50)


if __name__ == "__main__":
    main()

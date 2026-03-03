import json
import time
from pathlib import Path

import run
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


TSLIB_DIR = Path(__file__).resolve().parent

# Where Optuna stored best params
OPTUNA_DIR = TSLIB_DIR / "checkpoints" / "optuna"

# Where final training outputs will go
FINAL_DIR = TSLIB_DIR / "checkpoints" / "final_best"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ===== USER CONFIG =====
MODELS = ["DLinear", "TimeXer", "TimeMixer", "AMy_M_Linear_Regression", "TimesNet", "FEDformer", "PatchTST", "iTransformer","Nonstationary_Transformer"]  
N_FEATURES = 90

ROOT_PATH = "/home/ubuntu/Time-Series-Library/data_"
DATA_PATH = "data_karmoy_2024_L42_processed.csv"

TASK_NAME = "long_term_forecast"
FEATURES_MODE = "M"
TARGET = "OT"
FREQ = "t"

SEQ_LEN = 48
LABEL_LEN = 24
PRED_LEN = 1

# Final training budget
FINAL_EPOCHS = 50
PATIENCE = 8
BATCH_BEATS_QUIET = True  # keep output minimal

# CPU-only
FORCE_CPU = True

# Save test predictions and plots/files
SAVE_TEST_OUTPUTS = True

# If you want inverse scaling for metrics/plots in test:
INVERSE = False  # set True if you want inverse_transform behavior in test()

# Common defaults; best_params will override lr/bs/dropout/wd + model-specific params
COMMON = dict(
    task_name=TASK_NAME,
    is_training=1,
    data="MYDATA",
    root_path=ROOT_PATH,
    data_path=DATA_PATH,
    features=FEATURES_MODE,
    target=TARGET,
    freq=FREQ,
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    enc_in=N_FEATURES,
    dec_in=N_FEATURES,
    c_out=N_FEATURES,

    itr=1,
    num_workers=0,
    train_epochs=FINAL_EPOCHS,
    patience=PATIENCE,

    # this script DOES test after training
    run_test=1,

    quiet=BATCH_BEATS_QUIET,
    inverse=INVERSE,
)


def load_best_params(model: str) -> dict:
    path = OPTUNA_DIR / model / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run tuning first so best_params.json exists."
        )
    obj = json.loads(path.read_text(encoding="utf-8"))
    # file format: {"best_value": ..., "best_params": {...}}
    return obj["best_params"]


def build_args_list(model: str, model_id: str, exp_dir: Path, params: dict) -> list[str]:
    args_list: list[str] = []

    # Common args
    for k, v in COMMON.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            # include flag if True; skip if False
            if v:
                args_list.append(flag)
        else:
            args_list += [flag, str(v)]

    # Force CPU if requested
    if FORCE_CPU:
        args_list.append("--no_use_gpu")

    # Save outputs from test
    if SAVE_TEST_OUTPUTS:
        args_list.append("--save_test_outputs")

    # Required per-run identifiers + output paths
    args_list += ["--model", model, "--model_id", model_id]
    args_list += ["--exp_dir", str(exp_dir)]
    args_list += ["--metrics_path", str(exp_dir / "metrics.jsonl")]
    args_list += ["--des", "final_best"]

    # Hyperparams from Optuna best_params
    # These keys must exist in your parser; they do in run.py.
    if "learning_rate" in params:
        args_list += ["--learning_rate", str(params["learning_rate"])]
    if "batch_size" in params:
        args_list += ["--batch_size", str(params["batch_size"])]
    if "dropout" in params:
        args_list += ["--dropout", str(params["dropout"])]
    if "weight_decay" in params:
        args_list += ["--weight_decay", str(params["weight_decay"])]

    # Model-specific params
    if model == "AMy_lstm":
        # if these are not in best_params (should be), fallback to safe defaults
        args_list += ["--d_mark", str(params.get("d_mark", 5))]
        args_list += ["--lstm_hidden", str(params.get("lstm_hidden", 128))]
        args_list += ["--lstm_layers", str(params.get("lstm_layers", 1))]

    if model == "TimeMixer":
        args_list += ["--down_sampling_method", str(params.get("down_sampling_method", "avg"))]
        args_list += ["--down_sampling_layers", str(params.get("down_sampling_layers", 1))]
        args_list += ["--down_sampling_window", str(params.get("down_sampling_window", 2))]
        args_list += ["--channel_independence", str(params.get("channel_independence", 0))]

    return args_list


def run_one_model(model: str):
    params = load_best_params(model)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{model}_{ts}"
    exp_dir = FINAL_DIR / model / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save the config we are about to run (super useful for thesis reproducibility)
    (exp_dir / "best_params_used.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    args_list = build_args_list(model=model, model_id=run_name, exp_dir=exp_dir, params=params)
    args = run.parse_args(args_list)

    # Run training + test in-process
    exp = Exp_Long_Term_Forecast(args)

    setting = f"{args.model}_{args.model_id}"
    print(f"\n==== FINAL TRAIN START: {setting} ====")
    print(f"exp_dir: {exp_dir}")
    print(f"epochs: {args.train_epochs} | patience: {args.patience} | bs: {args.batch_size} | lr: {args.learning_rate}")

    exp.train(setting=setting)

    if args.run_test:
        print(f"\n==== FINAL TEST START: {setting} ====")
        exp.test(setting=setting, test=1)

    print(f"\n==== DONE: {setting} ====")
    print(f"All outputs in: {exp_dir}\n")


def main():
    for m in MODELS:
        run_one_model(m)


if __name__ == "__main__":
    main()

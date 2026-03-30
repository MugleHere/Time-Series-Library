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

# ===== USER CONFIG ===== First list already run
#MODELS = [
#    "DLinear", "TimeXer", "TimeMixer", "AMy_M_Linear_Regression",
#    "TimesNet", "FEDformer", "PatchTST", "iTransformer", "Nonstationary_Transformer",
#    "AMy_lstm","Autoformer", "Reformer", "Informer",
#]
MODELS = [
"Transformer",
"Crossformer",
]
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
QUIET = False  # for final runs, you may want tqdm output

# CPU-only
FORCE_CPU = True

# Save test predictions and plots/files
SAVE_TEST_OUTPUTS = True

# If you want inverse scaling for metrics/plots in test:
INVERSE = False  # set True if you want inverse_transform behavior in test()

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

    # do test after training
    run_test=1,

    quiet=QUIET,
    inverse=INVERSE,
)

# -------------------------
# Architecture caps (match your sweep)
# Apply them here too so final training uses same “CPU-feasible” variants.
# -------------------------
ARCH_CAPS = {
    "PatchTST": dict(d_model=64, n_heads=4, e_layers=2, d_ff=256, patch_len=16),
    "iTransformer": dict(d_model=64, n_heads=4, e_layers=2, d_ff=256, factor=1),
    "TimeXer": dict(d_model=64, n_heads=4, e_layers=2, d_ff=256, patch_len=16),
    "TimeMixer": dict(d_model=64, e_layers=2, d_ff=256),
    "TimesNet": dict(d_model=64, n_heads=4, e_layers=2, d_ff=128, top_k=3, num_kernels=3),
    "FEDformer": dict(d_model=64, n_heads=4, e_layers=1, d_layers=1, d_ff=128, factor=1, moving_avg=13),
    # Nonstationary_Transformer: keep as-is unless you added caps in sweep
}

ARCH_KEYS = [
    "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "factor",
    "patch_len", "top_k", "num_kernels", "moving_avg"
]


def load_best_params(model: str) -> dict:
    path = OPTUNA_DIR / model / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run tuning first so best_params.json exists.")
    obj = json.loads(path.read_text(encoding="utf-8"))
    # expected format: {"best_value": ..., "best_params": {...}}
    bp = obj.get("best_params", None)
    if not isinstance(bp, dict):
        raise ValueError(f"{path} does not contain best_params dict.")
    return bp


def build_args_list(model: str, model_id: str, exp_dir: Path, best_params: dict) -> list[str]:
    args_list: list[str] = []

    # Common args
    for k, v in COMMON.items():
        flag = f"--{k}"
        if isinstance(v, bool):
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

    # Apply best hyperparams from Optuna
    for k in ["learning_rate", "batch_size", "dropout", "weight_decay"]:
        if k in best_params:
            args_list += [f"--{k}", str(best_params[k])]

    # Apply model-specific “structural” params from best_params (if present)
    # (Most sweeps won’t include them, but harmless if they do.)
    for k in ARCH_KEYS:
        if k in best_params:
            args_list += [f"--{k}", str(best_params[k])]

    # Apply required fixed model-specific args (TimeMixer, LSTM baselines etc.)
    if model == "AMy_lstm":
        args_list += ["--d_mark", str(best_params.get("d_mark", 5))]
        args_list += ["--lstm_hidden", str(best_params.get("lstm_hidden", 128))]
        args_list += ["--lstm_layers", str(best_params.get("lstm_layers", 1))]

    if model == "TimeMixer":
        # Ensure these exist (your model expects them)
        args_list += ["--down_sampling_method", str(best_params.get("down_sampling_method", "avg"))]
        args_list += ["--down_sampling_layers", str(best_params.get("down_sampling_layers", 1))]
        args_list += ["--down_sampling_window", str(best_params.get("down_sampling_window", 2))]
        args_list += ["--channel_independence", str(best_params.get("channel_independence", 0))]

    # IMPORTANT: enforce the same caps as sweep (final training should match sweep variant)
    caps = ARCH_CAPS.get(model)
    if caps:
        for k, v in caps.items():
            args_list += [f"--{k}", str(v)]

    # NOTE: do NOT pass --no_save_ckpt here; final training should save checkpoints
    return args_list


def run_one_model(model: str):
    try:
        best_params = load_best_params(model)
    except Exception as e:
        print(f"[skip] {model}: {e}")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{model}_{ts}"
    exp_dir = FINAL_DIR / model / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Record exactly what we used
    payload = {
        "model": model,
        "timestamp": ts,
        "best_params_from_optuna": best_params,
        "arch_caps_applied": ARCH_CAPS.get(model, {}),
        "common": COMMON,
    }
    (exp_dir / "final_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args_list = build_args_list(model=model, model_id=run_name, exp_dir=exp_dir, best_params=best_params)
    args = run.parse_args(args_list)

    exp = Exp_Long_Term_Forecast(args)
    setting = f"{args.model}_{args.model_id}"

    print(f"\n==== FINAL TRAIN START: {setting} ====")
    print(f"exp_dir: {exp_dir}")
    print(f"epochs: {args.train_epochs} | patience: {args.patience} | bs: {args.batch_size} | lr: {args.learning_rate}")
    # Show whether checkpoint saving is enabled (should be)
    print(f"no_save_ckpt: {getattr(args, 'no_save_ckpt', False)}")

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
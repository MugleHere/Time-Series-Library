import json
import time
from pathlib import Path

import run
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


TSLIB_DIR = Path(__file__).resolve().parent

# CHANGE THESE
OPTUNA_DIR_CANDIDATES = [
    TSLIB_DIR / "checkpoints_horizon1_2" / "optuna",   # CURRENT SWEEP DATA
]
FINAL_DIR = TSLIB_DIR / "final_best_horizon1" # FINAL OUTPUTS HERE
PRED_LEN = 1 # PREDICITON LENGTH



FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ===== USER CONFIG =====
MODELS = ["DLinear"]
#    , "TimeXer", "TimeMixer", "AMy_M_Linear_Regression",
#    "TimesNet", "PatchTST", "AMy_lstm", "Nonstationary_Transformer",
#    "iTransformer", "Autoformer"
#]
#
N_FEATURES = 89

ROOT_PATH = "/home/ubuntu/Time-Series-Library/data_"
DATA_PATH = "data_karmoy_to_2024_L42_processed.csv"

TASK_NAME = "long_term_forecast"
FEATURES_MODE = "M"
TARGET = "OT"
FREQ = "t"

SEQ_LEN = 48
LABEL_LEN = 24


# Final training budget
FINAL_EPOCHS = 50
PATIENCE = 8
QUIET = False

# CPU-only
FORCE_CPU = True

# Save test predictions and plots/files
SAVE_TEST_OUTPUTS = True

# Inverse scaling in test if wanted
INVERSE = False

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
    itr=3, #######################
    num_workers=0,
    train_epochs=FINAL_EPOCHS,
    patience=PATIENCE,
    run_test=1,
    quiet=QUIET,
    inverse=INVERSE,
)

MODEL_DEFAULTS = {
    "PatchTST": dict(
        d_model=64,
        n_heads=4,
        e_layers=1,
        d_ff=128,
        patch_len=16,
    ),
    "iTransformer": dict(
        d_model=64,
        n_heads=4,
        e_layers=1,
        d_ff=128,
        factor=1,
    ),
    "TimeXer": dict(
        d_model=64,
        n_heads=4,
        e_layers=1,
        d_ff=128,
        patch_len=16,
    ),
    "TimeMixer": dict(
        d_model=64,
        e_layers=1,
        d_ff=128,
        down_sampling_method="avg",
        down_sampling_layers=1,
        down_sampling_window=2,
        channel_independence=0,
    ),
    "TimesNet": dict(
        d_model=64,
        n_heads=4,
        e_layers=1,
        d_ff=128,
        top_k=3,
        num_kernels=3,
    ),
    "Nonstationary_Transformer": dict(
        d_model=64,
        n_heads=4,
        e_layers=1,
        d_ff=128,
    ),
    "Autoformer": dict(
        d_model=64,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=128,
        factor=1,
    ),
    "AMy_lstm": dict(
        d_mark=5,
        lstm_layers=1,
    ),
}

OPTIM_KEYS = [
    "learning_rate",
    "batch_size",
    "dropout",
    "weight_decay",
]

ARCH_KEYS = [
    "d_model",
    "n_heads",
    "e_layers",
    "d_layers",
    "d_ff",
    "factor",
    "patch_len",
    "top_k",
    "num_kernels",
    "moving_avg",
    "seg_len",
]

EXTRA_MODEL_KEYS = [
    "down_sampling_method",
    "down_sampling_layers",
    "down_sampling_window",
    "channel_independence",
    "lstm_hidden",
    "lstm_layers",
    "d_mark",
]


def find_best_params_path(model: str) -> Path:
    for optuna_dir in OPTUNA_DIR_CANDIDATES:
        path = optuna_dir / model / "best_params.json"
        if path.exists():
            return path
    searched = [str(p / model / "best_params.json") for p in OPTUNA_DIR_CANDIDATES]
    raise FileNotFoundError(
        f"Missing best_params.json for model '{model}'. Looked in:\n" + "\n".join(searched)
    )


def load_best_params(model: str) -> tuple[dict, Path]:
    path = find_best_params_path(model)
    obj = json.loads(path.read_text(encoding="utf-8"))
    bp = obj.get("best_params")
    if not isinstance(bp, dict):
        raise ValueError(f"{path} does not contain a valid best_params dict.")
    return bp, path


def add_arg(args_list: list[str], key: str, value):
    if isinstance(value, bool):
        if value:
            args_list.append(f"--{key}")
    else:
        args_list += [f"--{key}", str(value)]


def build_args_list(model: str, model_id: str, exp_dir: Path, best_params: dict) -> list[str]:
    args_list: list[str] = []

    for k, v in COMMON.items():
        add_arg(args_list, k, v)

    if FORCE_CPU:
        args_list.append("--no_use_gpu")

    if SAVE_TEST_OUTPUTS:
        args_list.append("--save_test_outputs")

    args_list += ["--model", model, "--model_id", model_id]
    args_list += ["--exp_dir", str(exp_dir)]
    args_list += ["--metrics_path", str(exp_dir / "metrics.jsonl")]
    args_list += ["--des", "final_best"]

    for k in OPTIM_KEYS:
        if k in best_params:
            add_arg(args_list, k, best_params[k])

    for k in ARCH_KEYS + EXTRA_MODEL_KEYS:
        if k in best_params:
            add_arg(args_list, k, best_params[k])

    defaults = MODEL_DEFAULTS.get(model, {})
    for k, v in defaults.items():
        if k not in best_params:
            add_arg(args_list, k, v)

    return args_list


def run_one_model(model: str):
    try:
        best_params, best_params_path = load_best_params(model)
    except Exception as e:
        print(f"[skip] {model}: {e}")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{model}_{ts}"
    exp_dir = FINAL_DIR / model / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "timestamp": ts,
        "best_params_path": str(best_params_path),
        "best_params_from_optuna": best_params,
        "defaults_applied_if_missing": MODEL_DEFAULTS.get(model, {}),
        "common": COMMON,
        "force_cpu": FORCE_CPU,
        "save_test_outputs": SAVE_TEST_OUTPUTS,
    }
    (exp_dir / "final_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args_list = build_args_list(
        model=model,
        model_id=run_name,
        exp_dir=exp_dir,
        best_params=best_params,
    )
    args = run.parse_args(args_list)

    exp = Exp_Long_Term_Forecast(args)
    setting = f"{args.model}_{args.model_id}"

    print(f"\n==== FINAL TRAIN START: {setting} ====")
    print(f"best params from: {best_params_path}")
    print(f"exp_dir: {exp_dir}")
    print(
        f"epochs: {args.train_epochs} | patience: {args.patience} | "
        f"bs: {getattr(args, 'batch_size', None)} | lr: {getattr(args, 'learning_rate', None)}"
    )
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
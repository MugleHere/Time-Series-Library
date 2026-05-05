"""
iterative_eval.py
-----------------
Evaluates all models autoregressively (iterative multi-step forecast) by:
  1. Reading best hyperparameters from the same optuna dir as final_eval_optuna.py
  2. Loading the checkpoint produced by final_eval_optuna.py for the trained horizon
  3. Rolling the model forward `iterative_horizon` steps one step at a time,
     feeding each prediction back as input
  4. Saving per-model metrics + a final ranking JSON

Key config (top of file):
    TRAINED_HORIZON    : pred_len the models were trained with  (e.g. 1)
    ITERATIVE_HORIZON  : how many steps to forecast iteratively (e.g. 36)

Output layout mirrors final_eval_optuna.py:
    iterative_results_H{TRAINED_HORIZON}_iter{ITERATIVE_HORIZON}/
        {model}/
            run/
                preds.npy
                trues.npy
                metrics.jsonl
        final_results.json    ← ranking across all models
"""

import os
import sys
import json
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ============================================================
#  USER CONFIG  — edit these to match your setup
# ============================================================

TSLIB_DIR = Path(__file__).resolve().parent

# ── which horizon were the models trained on? ───────────────
TRAINED_HORIZON = 1          # pred_len used during training / optuna sweep

# ── how many steps to forecast iteratively? ─────────────────
ITERATIVE_HORIZON = 36       # autoregressively roll this many steps

# ── where are the optuna best_params.json files? ────────────
#    (same list as in final_eval_optuna.py)
OPTUNA_DIR_CANDIDATES = [
    TSLIB_DIR / f"checkpoints_horizon{TRAINED_HORIZON}_2" / "optuna",
    TSLIB_DIR / f"checkpoints_horizon{TRAINED_HORIZON}" / "optuna",
]

# ── where did final_eval_optuna.py save the trained checkpoints? ──
#    Script will look for:
#      FINAL_TRAIN_DIR / {model} / <any run folder> / checkpoint.pth  (or best.pth)
FINAL_TRAIN_DIR = TSLIB_DIR / f"final_best_horizon{TRAINED_HORIZON}"

# ── models to evaluate (same list as sweep + final_eval) ────
MODELS = []

# ── baseline models to evaluate iteratively (no checkpoint, no optuna) ──
# These are handled separately in run_one_model via BASELINE_MODELS set.
BASELINE_MODELS = ["AMy_Average_baseline"]

# ── data ────────────────────────────────────────────────────
ROOT_PATH  = "/home/ubuntu/Time-Series-Library/data_"
DATA_PATH  = "data_karmoy_to_2024_L42_processed.csv"
N_FEATURES = 89
SEQ_LEN    = 48
LABEL_LEN  = 24
FREQ       = "t"
TARGET     = "OT"
FEATURES   = "M"

# ── inference ───────────────────────────────────────────────
FORCE_CPU   = True
STRIDE      = 1      # evaluate every Nth test window (1 = all windows)
MAX_WINDOWS = None   # set an int to cap for quick testing, None = all

# ── fixed architecture caps (must match what was used in sweep) ──
MODEL_DEFAULTS = {
    "PatchTST": dict(d_model=64, n_heads=4, e_layers=1, d_ff=128, patch_len=16),
    "iTransformer": dict(d_model=64, n_heads=4, e_layers=1, d_ff=128, factor=1),
    "TimeXer": dict(d_model=64, n_heads=4, e_layers=1, d_ff=128, patch_len=16),
    "TimeMixer": dict(
        d_model=64, e_layers=1, d_ff=128,
        down_sampling_method="avg", down_sampling_layers=1,
        down_sampling_window=2, channel_independence=0,
    ),
    "TimesNet": dict(d_model=64, n_heads=4, e_layers=1, d_ff=128, top_k=3, num_kernels=3),
    "Nonstationary_Transformer": dict(d_model=64, n_heads=4, e_layers=1, d_ff=128, factor=1),
    "Autoformer": dict(d_model=64, n_heads=4, e_layers=1, d_layers=1, d_ff=128, factor=1),
    "AMy_lstm": dict(d_mark=5, lstm_hidden=64, lstm_layers=1),
}

# ── output ──────────────────────────────────────────────────
OUT_DIR = TSLIB_DIR / f"iterative_results_avg_baseline_H{TRAINED_HORIZON}_iter{ITERATIVE_HORIZON}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================


# ── thin namespace so model __init__ signatures are satisfied ─
class ArgsNamespace:
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)


# ── helpers ──────────────────────────────────────────────────

def find_best_params(model: str) -> dict:
    """Read best_params.json from the first optuna candidate that has it."""
    for base in OPTUNA_DIR_CANDIDATES:
        p = base / model / "best_params.json"
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            bp = obj.get("best_params")
            if isinstance(bp, dict):
                return bp
    searched = [str(b / model / "best_params.json") for b in OPTUNA_DIR_CANDIDATES]
    raise FileNotFoundError(
        f"No best_params.json for '{model}'. Looked in:\n" + "\n".join(searched)
    )


def find_checkpoint(model: str) -> Path:
    """
    Look inside FINAL_TRAIN_DIR / model / <any subdir> for checkpoint.pth or best.pth.
    Returns the path to the first one found.
    """
    model_dir = FINAL_TRAIN_DIR / model
    if not model_dir.exists():
        raise FileNotFoundError(f"No final-train directory for '{model}': {model_dir}")

    # walk one level of subdirectories (run folders named by timestamp)
    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for fname in ("best.pth", "checkpoint.pth", "last.pth"):
            ckpt = run_dir / fname
            if ckpt.exists():
                return ckpt

    raise FileNotFoundError(
        f"No .pth checkpoint found under {model_dir}. "
        "Run final_eval_optuna.py first."
    )


def infer_lstm_arch_from_ckpt(ckpt_path: Path, device: torch.device, n_features: int) -> dict:
    """
    Infer all AMy_lstm architecture hyperparameters directly from checkpoint
    weight shapes, so we never have to guess or hardcode them.

    Keys read:
      lstm.weight_ih_l0  shape (4*hidden, input_size)
          -> lstm_hidden = shape[0] // 4
          -> d_mark      = input_size - n_features   (input_size = enc_in + d_mark)
      lstm.weight_ih_lN  (one key per layer)
          -> lstm_layers = number of such keys found

    Returns a dict with the inferred values, or empty dict if no LSTM keys found.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt

    if "lstm.weight_ih_l0" not in state:
        return {}

    w = state["lstm.weight_ih_l0"]          # (4*hidden, input_size)
    lstm_hidden = w.shape[0] // 4
    input_size  = w.shape[1]
    d_mark      = input_size - n_features    # input_size = enc_in + d_mark

    # count how many lstm layers exist by counting weight_ih_lN keys
    lstm_layers = sum(
        1 for k in state if k.startswith("lstm.weight_ih_l")
    )

    return {
        "lstm_hidden" : lstm_hidden,
        "lstm_layers" : lstm_layers,
        "d_mark"      : d_mark,
    }


def build_model_args(model: str, best_params: dict, trained_horizon: int,
                     ckpt_path: Path | None = None,
                     device: torch.device = torch.device("cpu")) -> ArgsNamespace:
    """
    Construct the args namespace the model's __init__ expects.

    Priority for each architecture param:
      1. best_params (from optuna best_params.json)
      2. MODEL_DEFAULTS[model]  (the fixed caps used during the sweep)
      3. safe global fallback

    pred_len is always set to trained_horizon so the model's projection
    layers are built with the same shape as the saved checkpoint.
    """
    # Merge: global fallback < MODEL_DEFAULTS < best_params (non-optim keys)
    optim_keys = {"learning_rate", "batch_size", "dropout", "weight_decay"}
    arch = dict(MODEL_DEFAULTS.get(model, {}))
    for k, v in best_params.items():
        if k not in optim_keys:
            arch[k] = v

    # For AMy_lstm the architecture params (lstm_hidden, lstm_layers, d_mark)
    # were hardcoded in suggest_params and never written to best_params.json.
    # Read them directly from the checkpoint to be certain.
    if model == "AMy_lstm" and ckpt_path is not None:
        inferred = infer_lstm_arch_from_ckpt(ckpt_path, device, n_features=N_FEATURES)
        if inferred:
            arch.update(inferred)
            print(
                f"  [AMy_lstm] inferred from checkpoint: "
                f"lstm_hidden={inferred.get('lstm_hidden')}  "
                f"lstm_layers={inferred.get('lstm_layers')}  "
                f"d_mark={inferred.get('d_mark')}"
            )

    d = dict(
        # ── critical: must match the pred_len the checkpoint was trained with ──
        pred_len    = trained_horizon,
        seq_len     = SEQ_LEN,
        label_len   = LABEL_LEN,
        enc_in      = N_FEATURES,
        dec_in      = N_FEATURES,
        c_out       = N_FEATURES,
        task_name   = "long_term_forecast",
        is_training = 0,
        model       = model,
        model_id    = "iterative_eval",
        features    = FEATURES,
        target      = TARGET,
        freq        = FREQ,
        # ── safe global fallbacks (overwritten by arch dict below) ──────────
        d_model               = 64,
        n_heads               = 4,
        e_layers              = 1,
        d_layers              = 1,
        d_ff                  = 128,
        factor                = 1,
        dropout               = best_params.get("dropout", 0.0),
        patch_len             = 16,
        top_k                 = 3,
        num_kernels           = 3,
        moving_avg            = 25,
        down_sampling_layers  = 1,
        down_sampling_window  = 2,
        down_sampling_method  = "avg",
        channel_independence  = 0,
        lstm_hidden           = 64,
        lstm_layers           = 1,
        d_mark                = 5,
        # ── misc fields every model may check ───────────────────────────────
        individual        = False,
        embed             = "timeF",
        activation        = "gelu",
        use_norm          = 1,
        decomp_method     = "moving_avg",   # TimeMixer / Autoformer
        expand            = 2,
        d_conv            = 4,
        output_attention  = False,
        distil            = True,
        p_hidden_dims     = [128, 128],
        p_hidden_layers   = 2,
        use_amp           = False,
        use_gpu           = False,
        use_multi_gpu     = False,
    )
    # Architecture from optuna / MODEL_DEFAULTS wins over global fallbacks
    d.update(arch)
    return ArgsNamespace(d)


def load_model(model_name: str, args: ArgsNamespace, ckpt_path: Path,
               device: torch.device) -> nn.Module:
    from models import (
        Autoformer, Transformer, TimesNet, Nonstationary_Transformer,
        DLinear, FEDformer, Informer, LightTS, Reformer, ETSformer,
        PatchTST, Pyraformer, MICN, Crossformer, FiLM, iTransformer,
        Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple,
        TemporalFusionTransformer, TimeXer, WPMixer,
    )
    model_dict = {
        "Autoformer": Autoformer, "Transformer": Transformer,
        "TimesNet": TimesNet, "Nonstationary_Transformer": Nonstationary_Transformer,
        "DLinear": DLinear, "FEDformer": FEDformer, "Informer": Informer,
        "LightTS": LightTS, "Reformer": Reformer, "ETSformer": ETSformer,
        "PatchTST": PatchTST, "Pyraformer": Pyraformer, "MICN": MICN,
        "Crossformer": Crossformer, "FiLM": FiLM, "iTransformer": iTransformer,
        "Koopa": Koopa, "TiDE": TiDE, "FreTS": FreTS, "TimeMixer": TimeMixer,
        "TSMixer": TSMixer, "SegRNN": SegRNN, "MambaSimple": MambaSimple,
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "TimeXer": TimeXer, "WPMixer": WPMixer,
    }
    # custom models
    for cname in ["AMy_M_Linear_Regression", "AMy_lstm"]:
        try:
            import importlib
            mod = importlib.import_module(f"models.{cname}")
            model_dict[cname] = mod
        except ImportError:
            pass

    if model_name not in model_dict:
        raise ValueError(f"Model '{model_name}' not in model_dict.")

    model = model_dict[model_name].Model(args).float().to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_test_dataset(iterative_horizon: int):
    """
    Load Dataset_MYDATA with flag='test'.
    pred_len is set to iterative_horizon so that index_map guarantees
    at least that many ground-truth rows after each window start.
    """
    from data_provider.data_loader import Dataset_MYDATA

    class _A:
        pass
    a = _A()
    a.seq_len     = SEQ_LEN
    a.label_len   = LABEL_LEN
    a.pred_len    = iterative_horizon
    a.train_ratio = 0.7
    a.val_ratio   = 0.1

    return Dataset_MYDATA(
        args        = a,
        root_path   = ROOT_PATH,
        flag        = "test",
        features    = FEATURES,
        data_path   = DATA_PATH,
        target      = TARGET,
        scale       = True,
        timeenc     = 0,
        freq        = FREQ,
    )


# ── single model-step forward pass ───────────────────────────

@torch.no_grad()
def model_step(model: nn.Module,
               x:       torch.Tensor,   # (1, seq_len, C)
               x_mark:  torch.Tensor,   # (1, seq_len, D)
               y_hist:  torch.Tensor,   # (1, label_len, C)
               y_mark:  torch.Tensor,   # (1, label_len + trained_horizon, D)
               trained_horizon: int,
               device: torch.device) -> np.ndarray:
    """One forward pass; returns all predicted steps as (trained_horizon, C)."""
    C = x.shape[-1]
    dec_inp = torch.cat(
        [y_hist, torch.zeros(1, trained_horizon, C, device=device)], dim=1
    )

    out = model(x, x_mark, dec_inp, y_mark)   # (1, trained_horizon or more, C)
    return out[:, :trained_horizon, :].cpu().numpy()[0]   # (trained_horizon, C)


# ── iterative rollout for one model ──────────────────────────
def run_iterative(model: nn.Module, ds, device: torch.device,
                  trained_horizon: int, iterative_horizon: int) -> tuple:
    """
    Returns preds (N, iterative_horizon, C) and trues (N, iterative_horizon, C).

    Chunked iterative rollout:
      - each model call predicts `trained_horizon` steps
      - all predicted steps are fed back into the rolling context
      - rollout continues until `iterative_horizon` total steps are produced
    """
    data_x     = ds.data_x
    data_y     = ds.data_y
    data_stamp = ds.data_stamp
    index_map  = ds.index_map

    n_windows = len(index_map)
    if MAX_WINDOWS is not None:
        n_windows = min(n_windows, MAX_WINDOWS)

    all_preds, all_trues = [], []

    for wi in range(0, n_windows, STRIDE):
        s_begin = int(index_map[wi])

        # initial context
        x_np  = data_x[s_begin : s_begin + SEQ_LEN]               # (seq_len, C)
        sx_np = data_stamp[s_begin : s_begin + SEQ_LEN]           # (seq_len, D)

        lbl_s   = s_begin + SEQ_LEN - LABEL_LEN
        yh_np   = data_y[lbl_s : lbl_s + LABEL_LEN]               # (label_len, C)
        syl_np  = data_stamp[lbl_s : lbl_s + LABEL_LEN]           # (label_len, D)

        fut_s   = s_begin + SEQ_LEN
        fut_e   = fut_s + iterative_horizon
        if fut_e > len(data_y):
            break

        true_fut = data_y[fut_s : fut_e]                          # (iterative_horizon, C)

        roll_x   = x_np.copy()
        roll_sx  = sx_np.copy()
        roll_yh  = yh_np.copy()
        roll_syl = syl_np.copy()

        produced_steps = 0
        step_preds = []

        while produced_steps < iterative_horizon:
            steps_this_call = min(trained_horizon, iterative_horizon - produced_steps)
            abs_row = fut_s + produced_steps

            # build future timestamps for a full trained_horizon call
            fut_stamp_rows = []
            for k in range(trained_horizon):
                r = abs_row + k
                fut_stamp_rows.append(
                    data_stamp[r] if r < len(data_stamp) else data_stamp[-1]
                )
            fut_stamp = np.stack(fut_stamp_rows, axis=0)   # (trained_horizon, D)

            y_mark_np = np.concatenate([roll_syl, fut_stamp], axis=0)

            x_t  = torch.tensor(roll_x[None], dtype=torch.float32, device=device)
            sx_t = torch.tensor(roll_sx[None], dtype=torch.float32, device=device)
            yh_t = torch.tensor(roll_yh[None], dtype=torch.float32, device=device)
            ym_t = torch.tensor(y_mark_np[None], dtype=torch.float32, device=device)

            pred_block = model_step(
                model, x_t, sx_t, yh_t, ym_t, trained_horizon, device
            )   # (trained_horizon, C)

            # keep only what we still need for the requested iterative horizon
            pred_block = pred_block[:steps_this_call]   # (steps_this_call, C)
            step_preds.append(pred_block)

            # feed all predicted steps back in, one by one
            for k in range(steps_this_call):
                pred_k = pred_block[k]                  # (C,)
                stamp_k_row = abs_row + k
                next_stamp = (
                    data_stamp[stamp_k_row]
                    if stamp_k_row < len(data_stamp)
                    else data_stamp[-1]
                )

                roll_x   = np.concatenate([roll_x[1:],   pred_k[None, :]],       axis=0)
                roll_sx  = np.concatenate([roll_sx[1:],  next_stamp[None, :]],   axis=0)
                roll_yh  = np.concatenate([roll_yh[1:],  pred_k[None, :]],       axis=0)
                roll_syl = np.concatenate([roll_syl[1:], next_stamp[None, :]],   axis=0)

            produced_steps += steps_this_call

        preds_one = np.concatenate(step_preds, axis=0)   # (iterative_horizon, C)

        all_preds.append(preds_one)
        all_trues.append(true_fut)

        if (wi // STRIDE + 1) % 200 == 0:
            n_done = wi // STRIDE + 1
            print(f"    {n_done} windows done", flush=True)

    preds = np.stack(all_preds, axis=0)
    trues = np.stack(all_trues, axis=0)
    return preds, trues


# ── metrics ──────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    """
    Compute all metrics using the exact same formulas as utils/metrics.py.
    All values are dimensionless ratios (not percentages) to match that file.

    preds / trues shape: (N_windows, n_steps, n_channels)

    Overall scalars: reduce over ALL elements (windows, steps, channels).
    Per-horizon:     reduce over windows and channels only -> one value per step.
    """
    errors = preds - trues   # (N, H, C)

    # ── MAE: mean absolute error ─────────────────────────────────────────
    # metrics.py: np.mean(np.abs(true - pred))
    overall_mae = float(np.mean(np.abs(errors)))
    horizon_mae = np.mean(np.abs(errors), axis=(0, 2)).tolist()   # (H,)

    # ── MSE: mean squared error ──────────────────────────────────────────
    # metrics.py: np.mean((true - pred) ** 2)
    overall_mse = float(np.mean(errors ** 2))
    horizon_mse = np.mean(errors ** 2, axis=(0, 2)).tolist()      # (H,)

    # ── RMSE: root mean squared error ────────────────────────────────────
    # metrics.py: np.sqrt(MSE(pred, true))
    # Note: sqrt(mean(errors^2)) over ALL elements, NOT mean of per-step RMSEs.
    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
    horizon_rmse = np.sqrt(np.mean(errors ** 2, axis=(0, 2))).tolist()

    # ── MAPE: mean absolute percentage error (ratio, not %) ─────────────
    # metrics.py: np.mean(np.abs((true - pred) / true))
    # No epsilon guard in the original — we match that exactly.
    # Warning: will be large / inf wherever true ≈ 0 (standardised data).
    overall_mape = float(np.mean(np.abs(errors / trues)))
    horizon_mape = np.mean(np.abs(errors / trues), axis=(0, 2)).tolist()

    # ── MSPE: mean squared percentage error (ratio, not %) ───────────────
    # metrics.py: np.mean(np.square((true - pred) / true))
    overall_mspe = float(np.mean(np.square(errors / trues)))
    horizon_mspe = np.mean(np.square(errors / trues), axis=(0, 2)).tolist()

    # ── SMAPE: symmetric MAPE (ratio, not %) ─────────────────────────────
    # metrics.py: mean( 2*|pred-true| / (|true|+|pred|) ) with mask denom > 1e-8
    denom = np.abs(trues) + np.abs(preds)    # (N, H, C)
    mask  = denom > 1e-8

    # overall
    if np.any(mask):
        overall_smape = float(
            np.mean(2.0 * np.abs(errors[mask]) / (denom[mask] + 1e-8))
        )
    else:
        overall_smape = float("nan")

    # per horizon step: apply mask per step slice
    horizon_smape = []
    for h in range(errors.shape[1]):
        e_h = errors[:, h, :]          # (N, C)
        d_h = denom[:, h, :]
        m_h = mask[:, h, :]
        if np.any(m_h):
            val = float(np.mean(2.0 * np.abs(e_h[m_h]) / (d_h[m_h] + 1e-8)))
        else:
            val = float("nan")
        horizon_smape.append(val)

    return {
        "overall_mse"   : overall_mse,
        "overall_mae"   : overall_mae,
        "overall_rmse"  : overall_rmse,
        "overall_mape"  : overall_mape,
        "overall_mspe"  : overall_mspe,
        "overall_smape" : overall_smape,
        "horizon_mse"   : horizon_mse,
        "horizon_mae"   : horizon_mae,
        "horizon_rmse"  : horizon_rmse,
        "horizon_mape"  : horizon_mape,
        "horizon_mspe"  : horizon_mspe,
        "horizon_smape" : horizon_smape,
        "n_windows"     : int(preds.shape[0]),
        "n_steps"       : int(preds.shape[1]),
        "n_channels"    : int(preds.shape[2]),
    }


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


# ── per-model runner ─────────────────────────────────────────


def run_one_baseline(model_name: str, ds, device: torch.device) -> dict:
    result = {
        "model"            : model_name,
        "trained_horizon"  : TRAINED_HORIZON,
        "iterative_horizon": ITERATIVE_HORIZON,
        "is_baseline"      : True,
        "status"           : "pending",
        "overall_mse"      : None,
        "overall_mae"      : None,
        "overall_rmse"     : None,
        "overall_mape"     : None,
        "overall_mspe"     : None,
        "overall_smape"    : None,
        "horizon_mse"      : None,
        "horizon_mae"      : None,
        "horizon_rmse"     : None,
        "horizon_mape"     : None,
        "horizon_mspe"     : None,
        "horizon_smape"    : None,
        "n_windows"        : None,
        "ckpt_path"        : None,
        "best_params_used" : None,
        "seconds"          : None,
    }

    run_dir = OUT_DIR / model_name / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    t0 = time.time()

    try:
        args = ArgsNamespace(dict(
            pred_len    = TRAINED_HORIZON,
            seq_len     = SEQ_LEN,
            label_len   = LABEL_LEN,
            enc_in      = N_FEATURES,
            dec_in      = N_FEATURES,
            c_out       = N_FEATURES,
            task_name   = "long_term_forecast",
            is_training = 0,
            model       = model_name,
            model_id    = "iterative_baseline",
            features    = FEATURES,
            target      = TARGET,
            freq        = FREQ,
        ))

        import importlib
        mod   = importlib.import_module(f"models.{model_name}")
        model = mod.Model(args).float().to(device)
        model.eval()
        print(f"  [baseline] instantiated {model_name} (no checkpoint)")

        print(f"  rolling out {ITERATIVE_HORIZON} steps over {len(ds.index_map)} windows ...")
        preds, trues = run_iterative(model, ds, device,
                                     trained_horizon=TRAINED_HORIZON,
                                     iterative_horizon=ITERATIVE_HORIZON)

        m = compute_metrics(preds, trues)
        result.update(m)
        result["status"] = "ok"

        np.save(run_dir / "preds.npy", preds)
        np.save(run_dir / "trues.npy", trues)

        append_jsonl(metrics_path, {"event": "result", **m,
                                    "model": model_name,
                                    "trained_horizon": TRAINED_HORIZON,
                                    "iterative_horizon": ITERATIVE_HORIZON,
                                    "is_baseline": True})

    except Exception as e:
        result["status"] = "failed"
        result["error"]  = str(e)
        tb = traceback.format_exc()
        (run_dir / "exception.txt").write_text(tb, encoding="utf-8")
        print(f"  FAILED: {e}")
        append_jsonl(metrics_path, {"event": "error", "model": model_name, "error": str(e)})

    result["seconds"] = round(time.time() - t0, 1)
    return result

def run_one_model(model_name: str, ds, device: torch.device) -> dict:
    result = {
        "model"            : model_name,
        "trained_horizon"  : TRAINED_HORIZON,
        "iterative_horizon": ITERATIVE_HORIZON,
        "status"           : "pending",
        "overall_mse"      : None,
        "overall_mae"      : None,
        "overall_rmse"     : None,
        "overall_mape"     : None,
        "overall_mspe"     : None,
        "overall_smape"    : None,
        "horizon_mse"      : None,
        "horizon_mae"      : None,
        "horizon_rmse"     : None,
        "horizon_mape"     : None,
        "horizon_mspe"     : None,
        "horizon_smape"    : None,
        "n_windows"        : None,
        "ckpt_path"        : None,
        "best_params_used" : None,
        "seconds"          : None,
    }

    run_dir = OUT_DIR / model_name / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    t0 = time.time()

    try:
        # 1. load hyperparameters
        best_params = find_best_params(model_name)
        result["best_params_used"] = best_params
        print(f"  params: {best_params}")

        # 2. find checkpoint
        ckpt_path = find_checkpoint(model_name)
        result["ckpt_path"] = str(ckpt_path)
        print(f"  ckpt  : {ckpt_path}")

        # 3. build model — pass ckpt_path so lstm_hidden can be inferred from weights
        args  = build_model_args(model_name, best_params, TRAINED_HORIZON,
                                 ckpt_path=ckpt_path, device=device)
        model = load_model(model_name, args, ckpt_path, device)

        # 4. iterative rollout
        print(f"  rolling out {ITERATIVE_HORIZON} steps over {len(ds.index_map)} windows ...")
        preds, trues = run_iterative(model, ds, device, TRAINED_HORIZON, ITERATIVE_HORIZON)

        # 5. metrics
        m = compute_metrics(preds, trues)
        result.update(m)
        result["status"] = "ok"

        # 6. save arrays
        np.save(run_dir / "preds.npy", preds)
        np.save(run_dir / "trues.npy", trues)

        append_jsonl(metrics_path, {"event": "result", **m,
                                    "model": model_name,
                                    "trained_horizon": TRAINED_HORIZON,
                                    "iterative_horizon": ITERATIVE_HORIZON})

    except Exception as e:
        result["status"] = "failed"
        result["error"]  = str(e)
        tb = traceback.format_exc()
        (run_dir / "exception.txt").write_text(tb, encoding="utf-8")
        print(f"  FAILED: {e}")
        append_jsonl(metrics_path, {"event": "error", "model": model_name, "error": str(e)})

    result["seconds"] = round(time.time() - t0, 1)
    return result


# ── main ─────────────────────────────────────────────────────

def main():
    print(f"{'='*60}")
    print(f"  Iterative eval")
    print(f"  Trained horizon  : {TRAINED_HORIZON}")
    print(f"  Iterative horizon: {ITERATIVE_HORIZON}")
    print(f"  Models           : {MODELS}")
    print(f"  Baselines        : {BASELINE_MODELS}")
    print(f"  Output dir       : {OUT_DIR}")
    print(f"{'='*60}\n")

    device = torch.device("cpu") if FORCE_CPU else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}\n")

    # Load test dataset once — shared across all models
    print("Loading test dataset...")
    ds = load_test_dataset(ITERATIVE_HORIZON)
    print(f"Test windows available: {len(ds.index_map)}\n")

    all_results = []

    total = len(MODELS) + len(BASELINE_MODELS)

    for i, model_name in enumerate(MODELS, 1):
        print(f"[{i}/{total}] {model_name}")
        r = run_one_model(model_name, ds, device)
        all_results.append(r)

        status_str = (
            f"  mse={r['overall_mse']:.6f}  mae={r['overall_mae']:.6f}"
            f"  mape={r['overall_mape']:.6f}  smape={r['overall_smape']:.6f}"
            if r["status"] == "ok"
            else f"  status={r['status']}"
        )
        print(f"  done in {r['seconds']}s{status_str}\n")

    for j, model_name in enumerate(BASELINE_MODELS, len(MODELS) + 1):
        print(f"[{j}/{total}] {model_name}  [baseline]")
        r = run_one_baseline(model_name, ds, device)
        all_results.append(r)

        status_str = (
            f"  mse={r['overall_mse']:.6f}  mae={r['overall_mae']:.6f}"
            f"  mape={r['overall_mape']:.6f}  smape={r['overall_smape']:.6f}"
            if r["status"] == "ok"
            else f"  status={r['status']}"
        )
        print(f"  done in {r['seconds']}s{status_str}\n")

    # ── save final ranking ───────────────────────────────────
    def sort_key(r):
        v = r.get("overall_mse")
        return v if v is not None else float("inf")

    ranked = sorted(all_results, key=sort_key)
    (OUT_DIR / "final_results.json").write_text(
        json.dumps(ranked, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"Final ranking (overall_mse, iterative horizon={ITERATIVE_HORIZON}):")
    for r in ranked:
        mse_str   = f"{r['overall_mse']:.6f}"   if r["overall_mse"]   is not None else "N/A"
        mae_str   = f"{r['overall_mae']:.6f}"   if r["overall_mae"]   is not None else "N/A"
        mape_str  = f"{r['overall_mape']:.6f}"  if r["overall_mape"]  is not None else "N/A"
        mspe_str  = f"{r['overall_mspe']:.6f}"  if r["overall_mspe"]  is not None else "N/A"
        smape_str = f"{r['overall_smape']:.6f}" if r["overall_smape"] is not None else "N/A"
        print(
            f"  {r['model']:30s}  mse={mse_str}  mae={mae_str}"
            f"  mape={mape_str}  mspe={mspe_str}  smape={smape_str}  [{r['status']}]"
        )

    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
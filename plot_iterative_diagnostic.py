import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def default_base_dir(root: Path, mode: str, trained_horizon: int, forecast_horizon: int) -> Path:
    if mode == "iterative":
        return root / f"iterative_results_H{trained_horizon}_iter{forecast_horizon}"
    if mode == "direct":
        return root / f"final_best_horizon{forecast_horizon}"
    raise ValueError("mode must be 'iterative' or 'direct'")


def load_arrays(base_dir: Path, model: str, mode: str):
    if mode == "iterative":
        run_dir = base_dir / model / "run"
        pred_path = run_dir / "preds.npy"
        true_path = run_dir / "trues.npy"

    elif mode == "direct":
        model_dir = base_dir / model

        if not model_dir.exists():
            raise FileNotFoundError(f"Model folder does not exist: {model_dir}")

        run_dirs = [p for p in model_dir.iterdir() if p.is_dir()]

        if not run_dirs:
            raise FileNotFoundError(f"No run folder found inside {model_dir}")

        run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
        results_dir = run_dir / "results"

        pred_path = results_dir / "pred.npy"
        true_path = results_dir / "true.npy"

    else:
        raise ValueError("mode must be 'iterative' or 'direct'")

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    if not true_path.exists():
        raise FileNotFoundError(f"True-value file not found: {true_path}")

    preds = np.load(pred_path)
    trues = np.load(true_path)

    print(f"Loaded {model}")
    print(f"  pred: {pred_path}")
    print(f"  true: {true_path}")

    return preds, trues


def get_feature_name(csv_path: Path, feature_idx: int, n_features: int):
    df = pd.read_csv(csv_path, nrows=1)

    # Assumes first column is timestamp/date and last column is segment_id
    feature_names = list(df.columns[1:-1])

    if len(feature_names) != n_features:
        print(
            f"Warning: CSV has {len(feature_names)} feature names, "
            f"but arrays have {n_features} features."
        )
        return f"feature_{feature_idx}"

    return feature_names[feature_idx]


def select_window(preds, trues, feature_idx, mode):
    if mode == "worst":
        mse_per_window = np.mean(
            (preds[:, :, feature_idx] - trues[:, :, feature_idx]) ** 2,
            axis=1,
        )
        return int(np.argmax(mse_per_window)), float(np.max(mse_per_window))

    return int(mode), None


def load_inputs(input_path: Path | None, window_idx: int, feature_idx: int):
    if input_path is None:
        return None, None

    if not input_path.exists():
        print(f"Warning: input_path does not exist: {input_path}")
        return None, None

    inputs = np.load(input_path)
    print("inputs shape:", inputs.shape)

    input_window = inputs[window_idx, :, feature_idx]
    input_steps = np.arange(-input_window.shape[0], 0)

    return input_window, input_steps


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="/home/ubuntu/Time-Series-Library")
    parser.add_argument("--mode", type=str, default="iterative", choices=["iterative", "direct"])

    parser.add_argument("--trained_horizon", type=int, default=1)
    parser.add_argument("--forecast_horizon", type=int, default=36)

    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)

    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/ubuntu/Time-Series-Library/data_/data_karmoy_to_2024_L42_processed.csv",
    )

    parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--feature", type=int, default=0)

    parser.add_argument(
        "--window",
        type=str,
        default="worst",
        help="Use 'worst' or a specific window index, e.g. 25",
    )

    parser.add_argument("--out", type=str, default="single_model_diagnostic_forecast_plot.png")

    args = parser.parse_args()

    root = Path(args.root)

    if args.base_dir is None:
        base_dir = default_base_dir(
            root,
            args.mode,
            args.trained_horizon,
            args.forecast_horizon,
        )
    else:
        base_dir = Path(args.base_dir)

    if args.input_path is None:
        possible_input = (
            root
            / f"iterative_results_H{args.trained_horizon}_iter{args.forecast_horizon}_with_inputs"
            / "inputs.npy"
        )
        input_path = possible_input if possible_input.exists() else None
    else:
        input_path = Path(args.input_path)

    print(f"mode: {args.mode}")
    print(f"base_dir: {base_dir}")
    print(f"input_path: {input_path}")

    preds, trues = load_arrays(base_dir, args.model, args.mode)

    print(f"{args.model} preds shape:", preds.shape)
    print(f"{args.model} trues shape:", trues.shape)

    n_windows, horizon, n_features = preds.shape

    feature_idx = args.feature
    if feature_idx < 0:
        feature_idx = n_features + feature_idx

    if feature_idx < 0 or feature_idx >= n_features:
        raise ValueError(f"Feature index must be between 0 and {n_features - 1}")

    feature_name = get_feature_name(Path(args.csv_path), feature_idx, n_features)

    window_idx, worst_mse = select_window(
        preds,
        trues,
        feature_idx,
        args.window,
    )

    if window_idx < 0 or window_idx >= n_windows:
        raise ValueError(f"Window index must be between 0 and {n_windows - 1}")

    forecast_steps = np.arange(1, horizon + 1)

    true = trues[window_idx, :, feature_idx]
    pred = preds[window_idx, :, feature_idx]

    squared_error = (pred - true) ** 2
    mse = np.mean(squared_error)

    input_window, input_steps = load_inputs(input_path, window_idx, feature_idx)

    print(f"Selected window: {window_idx}")
    print(f"Feature index: {feature_idx}")
    print(f"Feature name: {feature_name}")
    print(f"{args.model} MSE: {mse:.4f}")

    if worst_mse is not None:
        print(f"Worst-window MSE: {worst_mse:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if input_window is not None:
        axes[0].plot(input_steps, input_window, label="Input window", linewidth=2)
        axes[0].axvline(0, linestyle=":", linewidth=1.5)

    axes[0].plot(forecast_steps, true, label="True future", linewidth=2.5)
    axes[0].plot(
        forecast_steps,
        pred,
        label=args.model,
        linestyle="--",
        linewidth=2,
    )

    axes[0].set_ylabel("Value", fontsize=20)
    axes[0].set_title(
        f"{args.mode.capitalize()} forecast for {args.model} on {feature_name} "
        f"(Horizon = {args.forecast_horizon})",
        fontsize=20,
    )
    axes[0].legend(fontsize=20)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        forecast_steps,
        squared_error,
        label=f"{args.model} squared error",
        linewidth=2,
    )

    axes[1].set_xlabel("Time step", fontsize=20)
    axes[1].set_ylabel("Squared error per step", fontsize=20)
    axes[1].set_title(
        f"Per-step squared error: {args.model} MSE={mse:.4f}",
        fontsize=20,
    )
    axes[1].legend(fontsize=20)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()

    out_path = Path(args.out)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
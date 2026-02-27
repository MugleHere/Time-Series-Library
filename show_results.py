# show_results.py
# Put this next to final_eval.py
#
# Usage:
#   python show_results.py --run lstm_MLinearRegression_20260225_110456
#   python show_results.py --latest

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
print("RUNNING:", __file__)


# -----------------------------
# Helpers
# -----------------------------
def find_latest_run(final_eval_root: Path) -> Optional[Path]:
    if not final_eval_root.exists():
        return None
    runs = [p for p in final_eval_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics_jsonl(metrics_path: Path) -> Dict[str, List]:
    curves = {"epoch": [], "train_loss": [], "val_loss": []}
    if not metrics_path.exists():
        return curves

    for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("event") == "epoch_end":
            curves["epoch"].append(int(obj.get("epoch", len(curves["epoch"]) + 1)))
            curves["train_loss"].append(obj.get("train_loss"))
            curves["val_loss"].append(obj.get("val_loss"))

    return curves

def plot_pred_true_examples(seed_rows: List[Dict[str, Any]], agg_rows: List[Dict[str, Any]], out_dir: Path,
                            n_examples: int = 5, channel: int = -1) -> None:
    """
    For each model/config, plot a few (pred_len) forecasts vs truth from the representative seed.
    Assumes pred.npy and true.npy are saved under exp_dir/results/.
    Shapes: (N, pred_len, C)
    """
    for agg in agg_rows:
        model = agg["model"]
        tag = agg["config_tag"]

        rep = pick_representative_seed(seed_rows, model=model, config_tag=tag)
        if rep is None:
            print(f"[plot] No representative seed found for {model} tag={tag}")
            continue

        exp_dir = Path(rep["exp_dir"])
        pred_path = exp_dir / "results" / "pred.npy"
        true_path = exp_dir / "results" / "true.npy"

        if not pred_path.exists() or not true_path.exists():
            print(f"[plot] Missing pred/true arrays for {model} tag={tag}. "
                  f"Expected:\n  {pred_path}\n  {true_path}\n"
                  f"Enable saving via --save_test_outputs (and ensure results dir is created).")
            continue

        preds = np.load(pred_path)
        trues = np.load(true_path)

        if preds.ndim != 3 or trues.ndim != 3:
            print(f"[plot] Unexpected shapes for {model}: pred={preds.shape}, true={trues.shape}")
            continue

        N = min(preds.shape[0], trues.shape[0])
        if N == 0:
            print(f"[plot] Empty pred/true arrays for {model}")
            continue

        # choose a few evenly spaced examples
        idxs = np.linspace(0, N - 1, num=min(n_examples, N), dtype=int)

        H = preds.shape[1]
        x = np.arange(H)

        for j, idx in enumerate(idxs, start=1):
            y_pred = preds[idx, :, channel]
            y_true = trues[idx, :, channel]

            plt.figure()
            plt.plot(x, y_true, label="true")
            plt.plot(x, y_pred, label="pred")
            plt.xlabel("horizon step")
            plt.ylabel("value")
            plt.title(f"{model} pred vs true (rep seed={rep.get('seed')}, sample={idx}, ch={channel})")
            plt.legend()
            plt.tight_layout()

            out_path = out_dir / f"pred_true_{model}_sample{j}.png"
            plt.savefig(out_path, dpi=200)
            print(f"[plot] Saved {out_path}")


def plot_pred_true_mean_profile(seed_rows: List[Dict[str, Any]], agg_rows: List[Dict[str, Any]], out_dir: Path,
                                channel: int = -1) -> None:
    """
    Plot mean prediction vs mean truth over the whole test set for representative seed.
    Useful when pred_len=1 too (just one point).
    """
    for agg in agg_rows:
        model = agg["model"]
        tag = agg["config_tag"]

        rep = pick_representative_seed(seed_rows, model=model, config_tag=tag)
        if rep is None:
            continue

        exp_dir = Path(rep["exp_dir"])
        pred_path = exp_dir / "results" / "pred.npy"
        true_path = exp_dir / "results" / "true.npy"

        if not pred_path.exists() or not true_path.exists():
            continue

        preds = np.load(pred_path)
        trues = np.load(true_path)

        y_pred = preds[:, :, channel]
        y_true = trues[:, :, channel]

        pred_mean = np.mean(y_pred, axis=0)
        true_mean = np.mean(y_true, axis=0)

        x = np.arange(pred_mean.shape[0])

        plt.figure()
        plt.plot(x, true_mean, label="true_mean")
        plt.plot(x, pred_mean, label="pred_mean")
        plt.xlabel("horizon step")
        plt.ylabel("value")
        plt.title(f"{model} mean pred vs true (rep seed={rep.get('seed')}, ch={channel})")
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"pred_true_mean_{model}.png"
        plt.savefig(out_path, dpi=200)
        print(f"[plot] Saved {out_path}")


def pick_representative_seed(seed_rows: List[Dict[str, Any]], model: str, config_tag: str) -> Optional[Dict[str, Any]]:
    """
    Choose the seed run whose test_mse is closest to the median test_mse for this config.
    """
    rows = [r for r in seed_rows if r.get("model") == model and r.get("config_tag") == config_tag and r.get("returncode") == 0]
    if not rows:
        return None
    mses = np.array([r["test_mse"] for r in rows if r.get("test_mse") is not None], dtype=float)
    if mses.size == 0:
        return None
    med = float(np.median(mses))
    best_idx = int(np.argmin(np.abs(mses - med)))
    return rows[best_idx]


def print_agg_table(agg_rows: List[Dict[str, Any]]) -> None:
    def f(x):
        return "None" if x is None else f"{x:.6g}"

    print("\nAggregated results per model/config (ranked by median test_mse):")
    print("-" * 160)
    print(f"{'model':24s} {'mse_med':>10s} {'mse_mean±std':>18s} {'rmse_mean±std':>18s} {'mae_mean±std':>18s} {'n_ok':>6s} {'tag':>10s}")
    print("-" * 160)

    rows_sorted = sorted(agg_rows, key=lambda r: r.get("test_mse_median") if r.get("test_mse_median") is not None else float("inf"))
    for r in rows_sorted:
        mse_mean = r.get("test_mse_mean")
        mse_std = r.get("test_mse_std")
        rmse_mean = r.get("test_rmse_mean")
        rmse_std = r.get("test_rmse_std")
        mae_mean = r.get("test_mae_mean")
        mae_std = r.get("test_mae_std")

        print(
            f"{str(r.get('model','')):24s} "
            f"{f(r.get('test_mse_median')):>10s} "
            f"{(f(mse_mean) + '±' + f(mse_std)):>18s} "
            f"{(f(rmse_mean) + '±' + f(rmse_std)):>18s} "
            f"{(f(mae_mean) + '±' + f(mae_std)):>18s} "
            f"{str(r.get('n_success',0)) + '/' + str(r.get('n_seeds',0)):>6s} "
            f"{str(r.get('config_tag','')):>10s}"
        )
    print("-" * 160)


def print_seed_breakdown(seed_rows: List[Dict[str, Any]], agg_rows: List[Dict[str, Any]]) -> None:
    def f(x):
        return "None" if x is None else f"{x:.6g}"

    print("\nPer-seed breakdown (for each model/config):")
    print("-" * 120)
    for agg in agg_rows:
        model = agg["model"]
        tag = agg["config_tag"]
        rows = [r for r in seed_rows if r.get("model") == model and r.get("config_tag") == tag]
        rows = sorted(rows, key=lambda r: r.get("seed", 0))
        print(f"\n{model}  tag={tag}  (median test_mse={agg.get('test_mse_median')})")
        for r in rows:
            print(f"  seed={r.get('seed')}  rc={r.get('returncode')}  test_mse={f(r.get('test_mse'))}  best_val={f(r.get('best_val_loss'))}  secs={f(r.get('seconds'))}")
    print("-" * 120)


def plot_two_bar_mse(agg_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    2 bars: one per model. Uses test_mse_median, error bar uses std of test_mse if available.
    """
    # one row per model (you likely have 1 config per model if BEST_PER_MODEL=True)
    # if multiple per model, take the best by median
    by_model: Dict[str, Dict[str, Any]] = {}
    for r in agg_rows:
        m = r["model"]
        if m not in by_model or (r.get("test_mse_median") or float("inf")) < (by_model[m].get("test_mse_median") or float("inf")):
            by_model[m] = r

    models = list(sorted(by_model.keys()))
    vals = [by_model[m].get("test_mse_median") for m in models]
    errs = [by_model[m].get("test_mse_std") for m in models]  # std across seeds (optional)

    plt.figure()
    x = np.arange(len(models))
    plt.bar(x, vals, yerr=errs, capsize=6)
    plt.xticks(x, models, rotation=0)
    plt.ylabel("test_mse (median across seeds)")
    plt.title("Model comparison (median test_mse; error=std across seeds)")
    plt.tight_layout()

    out_path = out_dir / "compare_test_mse.png"
    plt.savefig(out_path, dpi=200)
    print(f"[plot] Saved {out_path}")


def plot_learning_curves_representative(seed_rows: List[Dict[str, Any]], agg_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    For each model, plot ONE learning curve: the seed run closest to median test_mse.
    """
    for agg in agg_rows:
        model = agg["model"]
        tag = agg["config_tag"]

        rep = pick_representative_seed(seed_rows, model=model, config_tag=tag)
        if rep is None:
            print(f"[plot] No representative seed found for {model} tag={tag}")
            continue

        metrics_path = Path(rep["metrics_path"])
        curves = read_metrics_jsonl(metrics_path)
        if not curves["epoch"]:
            print(f"[plot] No curves found at {metrics_path}")
            continue

        epochs = curves["epoch"]

        def to_float(xs):
            out = []
            for x in xs:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(np.nan)
            return out

        train_loss = to_float(curves["train_loss"])
        val_loss = to_float(curves["val_loss"])

        plt.figure()
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"{model} learning curve (rep seed={rep.get('seed')}, test_mse={rep.get('test_mse')})")
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"curve_rep_{model}.png"
        plt.savefig(out_path, dpi=200)
        print(f"[plot] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="", help="Folder name under checkpoints/final_eval/")
    parser.add_argument("--latest", action="store_true", help="Use most recently modified run folder")
    args = parser.parse_args()

    tslib_dir = Path(__file__).resolve().parent
    final_eval_root = tslib_dir / "checkpoints" / "final_eval"

    if args.run:
        run_dir = final_eval_root / args.run
    else:
        run_dir = find_latest_run(final_eval_root) if args.latest else None

        if run_dir is None:
            raise FileNotFoundError(f"No runs found under {final_eval_root}")

    seed_path = run_dir / "seed_results.json"
    agg_path = run_dir / "final_results.json"

    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path}")

    agg_rows = load_json(agg_path)

    # NEW format if seed_results.json exists and agg looks aggregated
    is_new = seed_path.exists() and isinstance(agg_rows, list) and agg_rows and ("config_tag" in agg_rows[0] or "test_mse_median" in agg_rows[0])

    if is_new:
        seed_rows = load_json(seed_path)
        print(f"Loaded seed results: {seed_path} ({len(seed_rows)} rows)")
        print(f"Loaded aggregated results: {agg_path} ({len(agg_rows)} rows)")

        print_agg_table(agg_rows)
        print_seed_breakdown(seed_rows, agg_rows)

        out_dir = run_dir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_two_bar_mse(agg_rows, out_dir)
        plot_learning_curves_representative(seed_rows, agg_rows, out_dir)


        print(f"\nDone. Outputs under:\n  {out_dir}")

    else:
        # OLD format fallback: final_results.json is just per-model rows
        rows = agg_rows if isinstance(agg_rows, list) else []
        print(f"Loaded legacy final results: {agg_path} ({len(rows)} rows)")
        print("This looks like the OLD format (no per-seed aggregation).")
        print("Re-run final_eval.py with the 3-seed version to enable aggregation + representative curves.")




if __name__ == "__main__":
    main()

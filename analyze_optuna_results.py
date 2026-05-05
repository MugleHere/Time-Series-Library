import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd


# =========================
# CONFIG
# =========================
OPTUNA_ROOT = Path("checkpoints_horizon36/optuna")
OUT_ROOT = OPTUNA_ROOT / "_analysis"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TOP_N_TRIALS = 10
SAVE_PER_MODEL_TRIALS = True
MAKE_PARAM_SCATTERS = True
MAKE_IMPORTANCE_PLOTS = True
MAKE_HISTORY_PLOTS = True


# =========================
# HELPERS
# =========================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_study_for_model(model_dir: Path):
    db_path = model_dir / "optuna.db"
    if not db_path.exists():
        return None, f"Missing database: {db_path}"

    storage = f"sqlite:///{db_path.as_posix()}"

    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
    except Exception as e:
        return None, f"Could not read study summaries: {e}"

    if len(summaries) == 0:
        return None, f"No studies found in {db_path}"

    preferred_name = f"{model_dir.name}_tune"
    study_name = None

    for s in summaries:
        if s.study_name == preferred_name:
            study_name = s.study_name
            break

    if study_name is None:
        study_name = summaries[0].study_name

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        return study, None
    except Exception as e:
        return None, f"Could not load study '{study_name}': {e}"


def get_trial_dataframe(study: optuna.study.Study) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "state", "params"))
    if df.empty:
        return df

    # Normalize duration to seconds if possible
    if "duration" in df.columns:
        df["duration_seconds"] = df["duration"].apply(lambda x: x.total_seconds() if pd.notnull(x) else np.nan)

    # Add rank among completed trials
    if "value" in df.columns:
        df["value_numeric"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        df["value_numeric"] = np.nan

    complete_mask = df["state"].astype(str).str.contains("COMPLETE", na=False)
    df["rank_complete"] = np.nan
    if complete_mask.any():
        ranked = df.loc[complete_mask, "value_numeric"].rank(method="min", ascending=True)
        df.loc[complete_mask, "rank_complete"] = ranked

    return df


def summarize_study(model_name: str, study: optuna.study.Study, df: pd.DataFrame) -> dict:
    state_counts = df["state"].astype(str).value_counts().to_dict() if not df.empty else {}

    complete_df = df[df["state"].astype(str).str.contains("COMPLETE", na=False)].copy()
    best_value = np.nan
    median_value = np.nan
    mean_value = np.nan
    std_value = np.nan
    best_trial = None

    if not complete_df.empty:
        best_idx = complete_df["value_numeric"].idxmin()
        best_value = safe_float(complete_df.loc[best_idx, "value_numeric"])
        best_trial = int(complete_df.loc[best_idx, "number"])
        median_value = safe_float(complete_df["value_numeric"].median())
        mean_value = safe_float(complete_df["value_numeric"].mean())
        std_value = safe_float(complete_df["value_numeric"].std())

    summary = {
        "model": model_name,
        "study_name": study.study_name,
        "n_trials_total": len(df),
        "n_complete": int(state_counts.get("TrialState.COMPLETE", 0)),
        "n_pruned": int(state_counts.get("TrialState.PRUNED", 0)),
        "n_fail": int(state_counts.get("TrialState.FAIL", 0)),
        "n_running": int(state_counts.get("TrialState.RUNNING", 0)),
        "best_trial": best_trial,
        "best_value": best_value,
        "median_complete_value": median_value,
        "mean_complete_value": mean_value,
        "std_complete_value": std_value,
    }

    # Include best params if available
    try:
        summary["best_params"] = json.dumps(study.best_params, ensure_ascii=False)
    except Exception:
        summary["best_params"] = "{}"

    return summary


def plot_importances(model_name: str, importances: dict, out_dir: Path):
    if not importances:
        return

    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_title(f"{model_name}: parameter importance")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Parameter")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "param_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_optimization_history(model_name: str, df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return

    d = df[df["state"].astype(str).str.contains("COMPLETE", na=False)].copy()
    if d.empty:
        return

    d = d.sort_values("number")
    d["best_so_far"] = d["value_numeric"].cummin()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d["number"], d["value_numeric"], marker="o", linestyle="", alpha=0.8, label="trial value")
    ax.plot(d["number"], d["best_so_far"], linewidth=2, label="best so far")
    ax.set_title(f"{model_name}: optimization history")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective value")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "optimization_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_param_scatters(model_name: str, df: pd.DataFrame, out_dir: Path):
    d = df[df["state"].astype(str).str.contains("COMPLETE", na=False)].copy()
    if d.empty:
        return

    param_cols = [c for c in d.columns if c.startswith("params_")]
    for col in param_cols:
        series = d[col]
        val = d["value_numeric"]

        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Numeric parameters
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() >= max(3, len(series) // 2):
            ax.scatter(numeric_series, val, alpha=0.8)
            ax.set_xlabel(col.replace("params_", ""))
            ax.set_ylabel("Objective value")
            ax.set_title(f"{model_name}: {col.replace('params_', '')} vs objective")
        else:
            # Categorical parameters
            cats = series.astype(str).fillna("NA")
            cat_order = sorted(cats.unique())
            mapping = {c: i for i, c in enumerate(cat_order)}
            x = cats.map(mapping)
            ax.scatter(x, val, alpha=0.8)
            ax.set_xticks(range(len(cat_order)))
            ax.set_xticklabels(cat_order, rotation=30, ha="right")
            ax.set_xlabel(col.replace("params_", ""))
            ax.set_ylabel("Objective value")
            ax.set_title(f"{model_name}: {col.replace('params_', '')} vs objective")

            # plot category means
            means = d.groupby(cats)["value_numeric"].mean()
            mean_x = [mapping[c] for c in means.index]
            ax.plot(mean_x, means.values, marker="o", linestyle="-")

        plt.tight_layout()
        fname = f"{col.replace('params_', '')}_vs_objective.png"
        fig.savefig(out_dir / fname, dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_text_report(model_name: str, summary: dict, importances: dict, df: pd.DataFrame, out_dir: Path):
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append(f"Study: {summary['study_name']}")
    lines.append(f"Trials total: {summary['n_trials_total']}")
    lines.append(f"Complete: {summary['n_complete']}")
    lines.append(f"Pruned: {summary['n_pruned']}")
    lines.append(f"Failed: {summary['n_fail']}")
    lines.append(f"Best trial: {summary['best_trial']}")
    lines.append(f"Best value: {summary['best_value']}")
    lines.append(f"Median complete value: {summary['median_complete_value']}")
    lines.append("")

    # Parameter importance intel
    if importances:
        lines.append("Parameter importance:")
        for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {k}: {v:.4f}")
        lines.append("")

        top_param = max(importances, key=importances.get)
        lines.append(f"Most influential parameter: {top_param}")
        lines.append("")

    # Basic performance intel by parameter
    d = df[df["state"].astype(str).str.contains("COMPLETE", na=False)].copy()
    if not d.empty:
        for col in [c for c in d.columns if c.startswith("params_")]:
            pname = col.replace("params_", "")
            s = d[col]

            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().sum() >= max(3, len(s) // 2):
                corr = pd.DataFrame({"x": numeric, "y": d["value_numeric"]}).corr().iloc[0, 1]
                if pd.notnull(corr):
                    lines.append(f"Correlation({pname}, objective): {corr:.4f}")
            else:
                tmp = d[[col, "value_numeric"]].copy()
                tmp[col] = tmp[col].astype(str)
                grp = tmp.groupby(col)["value_numeric"].mean().sort_values()
                if len(grp) > 0:
                    lines.append(f"Best average category for {pname}: {grp.index[0]} ({grp.iloc[0]:.6f})")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


# =========================
# MAIN
# =========================
all_summaries = []
all_importances = []

model_dirs = sorted([d for d in OPTUNA_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")])

for model_dir in model_dirs:
    model_name = model_dir.name
    print(f"Analyzing {model_name}...")

    study, err = load_study_for_model(model_dir)
    if err is not None:
        print(f"  Skipped: {err}")
        continue

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = get_trial_dataframe(study)
    if SAVE_PER_MODEL_TRIALS:
        df.to_csv(out_dir / "trials.csv", index=False)

    summary = summarize_study(model_name, study, df)
    all_summaries.append(summary)

    # Save top trials
    complete_df = df[df["state"].astype(str).str.contains("COMPLETE", na=False)].copy()
    if not complete_df.empty:
        top_df = complete_df.sort_values("value_numeric").head(TOP_N_TRIALS)
        top_df.to_csv(out_dir / f"top_{TOP_N_TRIALS}_trials.csv", index=False)

    # Parameter importance
    importances = {}
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"  Importance failed for {model_name}: {e}")

    with (out_dir / "param_importance.json").open("w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2)

    imp_df = pd.DataFrame(
        [{"model": model_name, "parameter": k, "importance": v} for k, v in importances.items()]
    )
    imp_df.to_csv(out_dir / "param_importance.csv", index=False)

    all_importances.extend(
        [{"model": model_name, "parameter": k, "importance": v} for k, v in importances.items()]
    )

    if MAKE_IMPORTANCE_PLOTS:
        plot_importances(model_name, importances, out_dir)

    if MAKE_HISTORY_PLOTS:
        plot_optimization_history(model_name, df, out_dir)

    if MAKE_PARAM_SCATTERS:
        plot_param_scatters(model_name, df, out_dir)

    write_text_report(model_name, summary, importances, df, out_dir)

# Save cross-model summary
summary_df = pd.DataFrame(all_summaries)
if not summary_df.empty:
    summary_df = summary_df.sort_values("best_value", ascending=True)
    summary_df.to_csv(OUT_ROOT / "all_models_summary.csv", index=False)

imp_all_df = pd.DataFrame(all_importances)
if not imp_all_df.empty:
    imp_all_df.to_csv(OUT_ROOT / "all_models_param_importance.csv", index=False)

    # Mean importance across models
    imp_mean = (
        imp_all_df.groupby("parameter", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    imp_mean.to_csv(OUT_ROOT / "mean_param_importance_across_models.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(imp_mean["parameter"], imp_mean["importance"])
    ax.set_title("Mean parameter importance across models")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Mean importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_ROOT / "mean_param_importance_across_models.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

print(f"\nDone. Results saved to: {OUT_ROOT.resolve()}")
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(".")  # or Path("path/to/your/project")


def extract_state_dict(ckpt):
    """
    Return the model state dict from different possible checkpoint formats.
    """
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "net", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # if the checkpoint itself looks like a state_dict
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Could not identify a model state_dict in checkpoint.")


def count_parameters_from_state_dict(state_dict):
    """
    Count tensors stored in the state_dict.
    Note: this is usually close to the parameter count, but may also include buffers.
    """
    total_params = 0
    total_bytes = 0

    for tensor in state_dict.values():
        if torch.is_tensor(tensor):
            total_params += tensor.numel()
            total_bytes += tensor.numel() * tensor.element_size()

    return total_params, total_bytes


def analyze_checkpoints(root: Path):
    rows = []

    for best_path in root.glob("final_best_horizon*/**/best.pth"):
        try:
            ckpt = torch.load(best_path, map_location="cpu")
            state_dict = extract_state_dict(ckpt)
            total_params, total_bytes = count_parameters_from_state_dict(state_dict)

            parts = best_path.parts

            # try to extract horizon/model/run folder from path
            horizon_folder = next((p for p in parts if p.startswith("final_best_horizon")), None)

            # Expected structure:
            # final_best_horizonX / ModelName / RunFolder / best.pth
            run_folder = best_path.parent.name
            model_folder = best_path.parent.parent.name if best_path.parent.parent != best_path.parent else None

            # optional config
            config_path = best_path.parent / "final_config.json"
            config_data = {}
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                except Exception:
                    config_data = {}

            rows.append({
                "horizon_folder": horizon_folder,
                "model": model_folder,
                "run_folder": run_folder,
                "best_pth": str(best_path),
                "parameters": total_params,
                "approx_param_size_mb": total_bytes / (1024 ** 2),
                "best_pth_size_mb": best_path.stat().st_size / (1024 ** 2),
                "config_model": config_data.get("model"),
            })
            print(f"{model_folder} has {total_params} params")

        except Exception as e:
            rows.append({
                "horizon_folder": None,
                "model": None,
                "run_folder": None,
                "best_pth": str(best_path),
                "parameters": None,
                "approx_param_size_mb": None,
                "best_pth_size_mb": None,
                "config_model": None,
                "error": str(e),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["horizon_folder", "model"]).reset_index(drop=True)

    return df


df_params = analyze_checkpoints(ROOT)

print(df_params)
df_params.to_csv("model_parameter_summary.csv", index=False)
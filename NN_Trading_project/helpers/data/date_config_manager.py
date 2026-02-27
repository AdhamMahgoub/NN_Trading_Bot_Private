"""
date_config_manager.py
----------------------
Persists the active date/interval hyperparameters to date_config.txt.
If any value has changed since the last run, all cached stock CSVs are
deleted so the download step re-fetches fresh data automatically.
"""

import shutil
from pathlib import Path


def _read_config(path: Path) -> dict:
    config = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()
    return config


def _write_config(path: Path, config: dict) -> None:
    lines = ["# Auto-generated — do not edit manually", ""]
    for k, v in config.items():
        lines.append(f"{k} = {v}")
    path.write_text("\n".join(lines) + "\n")


def check_and_refresh_date_config(
    current_config: dict,
    config_path: Path,
    stocks_dir: Path,
) -> bool:
    """
    Compare *current_config* against the stored date_config.txt.

    If the config has changed:
      - Prints which keys changed.
      - Deletes *stocks_dir* so old CSVs are removed.
      - Writes the new config to *config_path*.
      - Returns True  (caller should re-download data).

    If the config is unchanged:
      - Prints a confirmation message.
      - Returns False (caller can use cached CSVs).

    Parameters
    ----------
    current_config : dict
        Mapping of config-key -> string value (e.g. {"START_DATE": "2024-01-01", ...}).
    config_path : Path
        Path to the persisted date_config.txt file.
    stocks_dir : Path
        Directory that holds per-ticker CSV files.
    """
    stored_config = _read_config(config_path)

    if stored_config != current_config:
        print("Date config changed — clearing cached CSVs for a fresh download.")
        if stored_config:
            print("  Previous config:")
            for k, v in stored_config.items():
                new_v = current_config.get(k, "<removed>")
                marker = " <-- CHANGED" if new_v != v else ""
                print(f"    {k}: {v}{marker}")

        if stocks_dir.exists():
            shutil.rmtree(stocks_dir)
            print(f"  Deleted: {stocks_dir}")
        stocks_dir.mkdir(parents=True, exist_ok=True)

        _write_config(config_path, current_config)
        print(f"  Updated: {config_path}")
        return True

    print("Date config unchanged — using cached CSVs where available.")
    print(f"  Config file: {config_path}")
    return False

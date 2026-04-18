import shutil
from pathlib import Path


def check_and_refresh_date_config(current_config, config_path, stocks_dir):
    stored = _read_config(config_path)

    if stored == current_config:
        print(f"Config unchanged — using cached CSVs ({config_path})")
        return False

    print("Config changed — clearing cached CSVs for fresh download.")
    for k, v in stored.items():
        new_v = current_config.get(k, "<removed>")
        if new_v != v:
            print(f"  {k}: {v!r} → {new_v!r}")

    if stocks_dir.exists():
        shutil.rmtree(stocks_dir)
    stocks_dir.mkdir(parents=True, exist_ok=True)

    _write_config(config_path, current_config)
    return True


def _read_config(path):
    if not path.exists():
        return {}
    return {
        k.strip(): v.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
        for k, v in [line.split("=", 1)]
    }


def _write_config(path, config):
    lines = ["# Auto-generated - do not edit manually", ""]
    lines += [f"{k} = {v}" for k, v in config.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

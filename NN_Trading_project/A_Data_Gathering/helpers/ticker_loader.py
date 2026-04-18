import random
from pathlib import Path


def load_tickers(path, subset=None, seed=42):
    all_tickers = sorted({
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    })

    if subset is None:
        return all_tickers

    if isinstance(subset, list):
        subset = [t.upper() for t in subset]
        missing = [t for t in subset if t not in all_tickers]
        if missing:
            print(f"WARNING: unknown tickers: {missing}")
        return sorted(t for t in subset if t in all_tickers)

    if isinstance(subset, int):
        random.seed(seed)
        return sorted(random.sample(all_tickers, min(subset, len(all_tickers))))

    raise ValueError(f"subset must be None, list, or int — got {type(subset)}")

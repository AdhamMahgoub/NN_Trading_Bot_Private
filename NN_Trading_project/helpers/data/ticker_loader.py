"""
ticker_loader.py
----------------
Loads tickers from a .txt file and optionally returns a subset.
"""

import random
from pathlib import Path


def load_tickers(
    path: Path,
    subset=None,
    seed: int = 42,
) -> list[str]:
    """Read tickers from *path* and apply optional subset selection.

    Parameters
    ----------
    path : Path
        Text file with one ticker per line. Lines starting with '#' are ignored.
    subset : None | list[str] | int
        None        → return all tickers
        list[str]   → return only those tickers (warns about unknowns)
        int         → return that many tickers chosen at random (seeded)
    seed : int
        Random seed used when subset is an int.

    Returns
    -------
    list[str]  – sorted, de-duplicated, upper-cased
    """
    all_tickers = sorted({
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    })

    if subset is None:
        return all_tickers

    if isinstance(subset, list):
        subset_upper = [t.upper() for t in subset]
        missing = [t for t in subset_upper if t not in all_tickers]
        if missing:
            print(f"WARNING: tickers not in {path.name}: {missing}")
        return sorted(t for t in subset_upper if t in all_tickers)

    if isinstance(subset, int):
        random.seed(seed)
        return sorted(random.sample(all_tickers, min(subset, len(all_tickers))))

    raise ValueError(f"subset must be None, a list, or an int — got {type(subset)}")

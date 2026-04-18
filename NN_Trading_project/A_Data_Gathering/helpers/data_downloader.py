from pathlib import Path
import pandas as pd
import yfinance as yf


def download_tickers(tickers, start, end, interval, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = pd.to_datetime(start), pd.to_datetime(end)

    print(f"Downloading {interval} data  {start.date()} → {end.date()}  [{len(tickers)} tickers]")

    failed, skipped, n_dl, new_rows = [], [], 0, 0

    for ticker in tickers:
        path = out_dir / f"{ticker}.csv"
        if path.exists():
            try:
                if not pd.read_csv(path).empty:
                    skipped.append(ticker)
                    continue
            except Exception:
                pass

        try:
            df = yf.download(ticker, start=start, end=end, interval=interval,
                             auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            print(f"  WARN  {ticker}: {e}")
            failed.append(ticker)
            continue

        if df.empty:
            print(f"  WARN  {ticker}: no data in range")
            failed.append(ticker)
            continue

        df = df.reset_index()
        if "Datetime" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        df.to_csv(path, index=False)
        n_dl += 1
        new_rows += len(df)
        print(f"  OK    {ticker}  rows={len(df)}  ({n_dl}/{len(tickers)})")

    total_rows = sum(
        len(pd.read_csv(f)) for f in out_dir.glob("*.csv")
        if not _safe_empty(f)
    )

    print(f"\nDone — {n_dl} downloaded, {len(skipped)} cached, {len(failed)} failed")
    print(f"Rows: {new_rows} new / {total_rows} total on disk")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(sorted(set(failed)))}")

    return {"failed": failed, "skipped": skipped,
            "downloaded_count": n_dl, "new_rows": new_rows, "total_rows": total_rows}


def _safe_empty(path):
    try:
        return pd.read_csv(path).empty
    except Exception:
        return True

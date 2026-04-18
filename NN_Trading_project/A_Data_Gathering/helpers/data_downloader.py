"""
data_downloader.py
------------------
Utility for fetching price CSVs via yfinance.

The core function is :
    download_tickers(tickers, start, end, interval, out_dir)

It will create the output directory if necessary, skip existing nonempty
CSVs, and report counts of downloaded/failed/skipped rows.

Returns a dict with keys:
  - failed  : list of tickers that had errors or no data
  - skipped : list of tickers where a valid CSV already existed
  - downloaded_count : number of tickers actually downloaded
  - new_rows : total rows added this run
  - total_rows : total rows on disk after the run

The function prints progress info to stdout, matching the original
notebook behaviour.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf


def download_tickers(
    tickers,
    start,
    end,
    interval,
    out_dir,
):
    """Fetch CSVs for the given tickers and date range.

    Parameters
    ----------
    tickers : Sequence[str]
    start : pandas.Timestamp or str
    end : pandas.Timestamp or str
    interval : str
    out_dir : Path or str
        destination directory for per-ticker CSV files.

    Returns
    -------
    dict
        A summary dict containing failed/skipped/downloaded counts and
        row totals.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(start)
    end   = pd.to_datetime(end)

    print(f"Downloading {interval} data for {len(tickers)} tickers from >>> {start.date()} to {end.date()} <<< into {out_dir} ...")

    failed_tickers = []
    skipped_tickers = []
    downloaded_tickers = 0
    total_new_rows = 0

    for ticker in tickers:
        out_path = out_dir / f"{ticker}.csv"

        # If file already exists and is non-empty, skip downloading again
        if out_path.exists() and out_path.is_file():
            try:
                existing_df = pd.read_csv(out_path)
                if not existing_df.empty:
                    print(f" SKIP :: {ticker} (cached CSV found at {out_path})")
                    skipped_tickers.append(ticker)
                    continue
            except Exception:
                print(f" INFO :: Existing file for {ticker} could not be read; re-downloading.")

        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception as e:
            print(f" WARNING :: download failed for {ticker}: {e}")
            failed_tickers.append(ticker)
            continue

        if df.empty:
            print(f" WARNING :: (no {interval} data for {ticker} in this range)")
            failed_tickers.append(ticker)
            continue

        df = df.reset_index()
        if "Datetime" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        df.to_csv(out_path, index=False)
        downloaded_tickers += 1
        total_new_rows += len(df)
        print(f" OK   :: {ticker} rows={len(df)} | downloaded {downloaded_tickers}/{len(tickers)}")

    # post-run statistics
    total_rows = 0
    for csv_file in out_dir.glob("*.csv"):
        try:
            total_rows += len(pd.read_csv(csv_file))
        except Exception:
            pass

    print("Path to dataset files:", str(out_dir.parent))
    print("Individual ticker CSVs are in:", out_dir)
    print(f"Total number of rows across all CSVs on disk: {total_rows}")
    print(f"Newly downloaded rows in this run: {total_new_rows}")
    print(f"Downloaded tickers in this run: {downloaded_tickers}/{len(tickers)}")

    if failed_tickers:
        print("\nTickers with download errors or no data ({}):".format(len(failed_tickers)))
        print(", ".join(sorted(set(failed_tickers))))
    else:
        print("\nAll requested tickers that were downloaded in this run succeeded.")

    if skipped_tickers:
        print("\nTickers skipped because CSV already exists ({}):".format(len(skipped_tickers)))
        print(", ".join(sorted(set(skipped_tickers))))

    return {
        "failed": failed_tickers,
        "skipped": skipped_tickers,
        "downloaded_count": downloaded_tickers,
        "new_rows": total_new_rows,
        "total_rows": total_rows,
    }

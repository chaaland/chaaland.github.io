"""
Print the stock universe used by covariance.py as a Python dict, to paste into its universe_selection cell. Needs
internet access, so it is only run when the universe is rebuilt, never by the notebook.

    python build_universe.py

The universe is the current S&P 500 ranked by average daily dollar volume (Close x Volume) over the first quarter of
2020, which is before the sample period of the notebook starts. To keep small sectors from being left with one or two
names, the top MIN_PER_SECTOR stocks of every GICS sector are taken first, and the rest are filled in by overall rank.
"""

import io

import pandas as pd
import requests
import yfinance as yf

UNIVERSE_SIZE = 75
MIN_PER_SECTOR = 4
RETRIES = 2
RANKING_START, RANKING_END = "2020-01-01", "2020-04-01"
# second share classes of companies that are already in the index, which move together with the first
DUPLICATE_SHARE_CLASSES = ["GOOGL", "FOX", "NWS"]


def fetch_sp500_candidates() -> pd.DataFrame:
    """Current S&P 500 constituents with their GICS sector and average daily dollar volume over the ranking window."""
    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"}, timeout=30
    ).text
    table = pd.read_html(io.StringIO(html))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False)  # Yahoo writes BRK.B as BRK-B

    prices = yf.download(
        tickers.tolist(), start=RANKING_START, end=RANKING_END, auto_adjust=False, progress=False, group_by="column"
    )
    dollar_volume = (prices["Close"] * prices["Volume"]).mean()

    # the batch download randomly drops tickers, so retry the missing ones one at a time before treating them as absent
    for attempt in range(RETRIES):
        for ticker in dollar_volume.index[dollar_volume.isna()]:
            history = yf.Ticker(ticker).history(start=RANKING_START, end=RANKING_END, auto_adjust=False)
            if len(history) > 0:
                dollar_volume[ticker] = (history["Close"] * history["Volume"]).mean()

    candidates = pd.DataFrame({"Ticker": tickers, "Sector": table["GICS Sector"], "DollarVolume": tickers.map(dollar_volume)})
    absent = candidates[candidates["DollarVolume"].isna()]["Ticker"].tolist()
    print(f"no trading data in the ranking window for {len(absent)} stocks (mostly recent listings): {', '.join(absent)}")
    return candidates.dropna()


def pick_universe(candidates: pd.DataFrame) -> pd.DataFrame:
    """The top stocks by dollar volume, after first taking the top MIN_PER_SECTOR of every sector."""
    ranked = candidates[~candidates["Ticker"].isin(DUPLICATE_SHARE_CLASSES)].sort_values("DollarVolume", ascending=False)
    floor = ranked.groupby("Sector").head(MIN_PER_SECTOR)
    fill = ranked.drop(floor.index).head(UNIVERSE_SIZE - len(floor))
    return ranked.loc[ranked.index.isin(floor.index.union(fill.index))]  # keeps the dollar volume ranking


if __name__ == "__main__":
    chosen = pick_universe(fetch_sp500_candidates())
    print("    sector_tickers = {")
    for sector, group in chosen.groupby("Sector", sort=False):
        print(f'        "{sector}": [{", ".join(repr(t).replace(chr(39), chr(34)) for t in group["Ticker"])}],')
    print("    }  # fmt: skip")

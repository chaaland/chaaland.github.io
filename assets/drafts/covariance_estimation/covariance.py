import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Covariance Estimation")


@app.cell
def imports():
    import datetime
    import yfinance as yf
    import polars as pl
    import polars.selectors as cs
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    from pathlib import Path

    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    return Path, cs, end_date, mo, np, pl, plt, start_date, yf


@app.function
def make_cartesian_plane(ax):
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")


@app.cell
def chart_style_helpers():
    import matplotlib.dates as _mdates
    import matplotlib.ticker as _mticker

    _GRID = "#e5e7eb"
    _SPINE = "#d1d5db"
    _DIM = "#6b7280"

    def style_ax(ax, fig=None, date_interval=6, labelsize=8.5, tick_length=5, pad=6, y_fmt=None):
        if fig is not None:
            fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        ax.yaxis.grid(True, color=_GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(_SPINE)

        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(_mdates.MonthLocator(interval=date_interval))
        ax.tick_params(axis="x", colors=_DIM, labelsize=labelsize, length=tick_length, pad=pad, rotation=30)
        ax.tick_params(axis="y", colors=_DIM, labelsize=labelsize, length=0, pad=pad)

        if y_fmt is not None:
            ax.yaxis.set_major_formatter(_mticker.FuncFormatter(y_fmt))

    return (style_ax,)


@app.cell
def plot_helpers(np):
    _INK = "#111827"
    _DIM = "#6b7280"
    _GRID = "#e5e7eb"

    def new_grid(plt, nrows: int, ncols: int, figsize: tuple[float, float], **subplot_kwargs):
        """A white-background figure with an nrows x ncols grid of axes."""
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kwargs)
        fig.patch.set_facecolor("#ffffff")
        return fig, axes

    def finish_grid(fig, title: str, top: float = 0.94):
        """Left-aligned figure title, with the layout leaving room for it above the panels."""
        title_figure(fig, title)
        fig.tight_layout(rect=[0, 0, 1, top])

    def title_figure(fig, text: str, fontsize: float = 11):
        fig.suptitle(text, color=_INK, fontsize=fontsize, fontweight="bold", x=0.0, ha="left")

    def title_axes(ax, text: str, fontsize: float = 11, pad: float = 10):
        ax.set_title(text, color=_INK, fontsize=fontsize, fontweight="bold", loc="left", pad=pad)

    def plot_line_with_end_marker(ax, dates, values, color, label=None):
        """A line with a dot on its last point."""
        ax.plot(dates, values, color=color, linewidth=1.6, alpha=0.9, zorder=2, label=label)
        ax.scatter([dates[-1]], [values[-1]], color=color, s=22, zorder=5, linewidths=0)

    def reference_line(ax, y: float = 0):
        ax.axhline(y, color="#9ca3af", linewidth=0.8, zorder=1)

    def boxed_legend(ax, loc: str = "upper left", **kwargs):
        ax.legend(
            frameon=True,
            loc=loc,
            facecolor="#f9fafb",
            edgecolor="#d1d5db",
            labelcolor=_INK,
            fontsize=8.5,
            handlelength=1.3,
            borderpad=0.9,
            borderaxespad=0.8,
            **kwargs,
        )

    def legend_above_axes(ax, ncol: int):
        """Frameless legend above the axes, level with the title, so it never covers data."""
        ax.legend(
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=ncol,
            labelcolor=_INK,
            fontsize=8.5,
            handlelength=1.3,
        )

    def style_plain_ax(ax, labelsize: float = 7.5, x_fmt=None, y_fmt=None):
        """Light styling for axes that are not time axes (see style_ax for those)."""
        ax.set_facecolor("#ffffff")
        ax.grid(True, color=_GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_DIM, labelsize=labelsize, length=0)
        if x_fmt is not None:
            ax.xaxis.set_major_formatter(x_fmt)
        if y_fmt is not None:
            ax.yaxis.set_major_formatter(y_fmt)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def draw_correlation_matrix(
        ax, corr: np.ndarray, order: np.ndarray, labels: list[str], name: str, highlight=(), max_labels: int = 50, vmax: float = 1.0
    ):
        """
        Heatmap of corr with rows and columns taken in `order`, on a symmetric scale from -vmax to vmax.

        With more than max_labels stocks the tick labels would overlap, so only the `highlight` tickers are labeled.
        """
        image = ax.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ordered_labels = [labels[i] for i in order]
        if len(order) > max_labels:
            ordered_labels = [label if label in highlight else "" for label in ordered_labels]
        ax.set_xticks(range(len(order)), ordered_labels, rotation=90, fontsize=6, color=_DIM)
        ax.set_yticks(range(len(order)), ordered_labels, fontsize=6, color=_DIM)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(name, color=_INK, fontsize=10, fontweight="bold", loc="left")
        return image

    def mark_sector_blocks(ax, block_ends: np.ndarray):
        """Draw a line between consecutive sector blocks of a matrix that is ordered by sector."""
        for end_of_block in block_ends[:-1]:
            ax.axhline(end_of_block - 0.5, color=_INK, linewidth=0.8)
            ax.axvline(end_of_block - 0.5, color=_INK, linewidth=0.8)

    def label_sector_blocks(ax, sectors, block_starts: np.ndarray, block_ends: np.ndarray, n: int):
        """Write each sector's name to the right of its block."""
        for sector, start, stop in zip(sectors, block_starts, block_ends):
            ax.text(n - 0.5 + 0.8, (start + stop - 1) / 2, sector, ha="left", va="center", fontsize=8, color=_INK)

    def robust_ylim(*series: np.ndarray, quantile: float = 0.995, margin: float = 1.1) -> tuple[float, int]:
        """
        Symmetric y-limit covering the central `quantile` of |values| with some margin, and the number of points outside it.

        series: arrays of any shape, whose values are pooled together.
        quantile: share of the |values| that the limit has to cover.
        margin: factor that the quantile is scaled by.

        Returns (ylim, n_outside). A handful of extreme days would otherwise squash the rest of the plot.
        """
        values = np.abs(np.concatenate([np.ravel(x) for x in series]))
        ylim = margin * np.quantile(values, quantile)
        return ylim, int((values > ylim).sum())

    def diagonal_std_devs(cov: np.ndarray) -> np.ndarray:
        """Standard deviations from the diagonal of cov, shape (N, N, T), as an (N, T) array."""
        return np.stack([cov[i, i, :] ** 0.5 for i in range(cov.shape[0])])

    return (
        boxed_legend,
        diagonal_std_devs,
        draw_correlation_matrix,
        finish_grid,
        label_sector_blocks,
        legend_above_axes,
        mark_sector_blocks,
        new_grid,
        plot_line_with_end_marker,
        reference_line,
        robust_ylim,
        style_plain_ax,
        title_axes,
        title_figure,
    )


@app.cell(hide_code=True)
def section_load_stock_returns(mo):
    mo.md(r"""
    # Load Stock Returns
    """)
    return


@app.cell
def plotted_symbols():
    # The first five are plotted throughout the notebook; the rest of the universe is added for the mixture experiments
    symbols = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
    colors = ["tab:blue", "tab:gray", "tab:purple", "tab:red", "tab:green"]
    symbol_to_color = dict(zip(symbols, colors))
    return symbol_to_color, symbols


@app.cell
def universe_selection(symbols):
    # 75 stocks from the S&P 500: the 4 most liquid of every GICS sector (average daily dollar volume over Q1 2020,
    # before the sample starts), then the most liquid of the rest. build_universe.py regenerates this list.
    sector_tickers = {
        "Information Technology": [
            "AAPL", "MSFT", "AMD", "NVDA", "INTC", "MU", "CSCO", "ADBE", "CRM", "QCOM", "AVGO", "IBM", "TXN", "NOW"
        ],
        "Consumer Discretionary": ["TSLA", "AMZN", "HD", "BKNG", "MCD", "SBUX", "NKE"],
        "Communication Services": ["META", "GOOG", "NFLX", "DIS", "T", "VZ", "CMCSA", "CHTR"],
        "Industrials": ["BA", "UBER", "RTX", "GE", "DAL", "MMM"],
        "Financials": ["V", "JPM", "BAC", "MA", "BRK-B", "C", "WFC", "PYPL", "GS", "XYZ", "MS"],
        "Health Care": ["UNH", "JNJ", "GILD", "PFE", "MRK", "ABBV", "BMY", "MCK"],
        "Energy": ["XOM", "CVX", "OXY", "SLB"],
        "Consumer Staples": ["PG", "WMT", "COST", "KO", "PEP"],
        "Utilities": ["NEE", "DUK", "SO", "AEP"],
        "Real Estate": ["AMT", "DLR", "PLD", "SPG"],
        "Materials": ["LIN", "NEM", "SHW", "APD"],
    }  # fmt: skip
    sector_of = {ticker: sector for sector, tickers in sector_tickers.items() for ticker in tickers}

    assert set(symbols) <= set(sector_of), f"plotted symbols missing from the universe: {set(symbols) - set(sector_of)}"
    universe = symbols + [ticker for ticker in sector_of if ticker not in symbols]  # plotted symbols come first
    return sector_of, universe


@app.cell
def load_stock_data(Path, cs, end_date, pl, start_date, symbols, universe, yf):
    data_file = Path("stocks.parquet")

    # Only download tickers that are not already in the cache, then write everything back
    cached_df = pl.read_parquet(data_file) if data_file.exists() else None
    cached_tickers = set() if cached_df is None else set(cached_df["Ticker"].unique())
    missing_symbols = [symbol for symbol in universe if symbol not in cached_tickers]

    if missing_symbols:
        frames = [] if cached_df is None else [cached_df]

        for symbol in missing_symbols:
            history = yf.Ticker(symbol).history(start=start_date, end=end_date)
            assert len(history) > 0, (
                f"{symbol} is not in stocks.parquet and yfinance returned no data for it. "
                "Offline, stocks.parquet must already contain every ticker in the universe."
            )
            frames.append(pl.from_pandas(history.reset_index()).with_columns(pl.lit(symbol).alias("Ticker")))

        cached_df = pl.concat(frames, how="diagonal_relaxed").sort("Ticker", "Date")
        cached_df.write_parquet(data_file)

    universe_df = cached_df.filter(pl.col("Ticker").is_in(universe))
    universe_df = universe_df.select(cs.by_name("Ticker", "Date"), ~cs.by_name("Ticker", "Date"))

    # the returns matrix needs every stock on the same dates
    assert universe_df.group_by("Ticker").len()["len"].n_unique() == 1, "tickers have different numbers of days"

    # the plots below only use the five tickers in `symbols`
    df = universe_df.filter(pl.col("Ticker").is_in(symbols))
    df
    return df, universe_df


@app.cell
def plot_close_prices(
    boxed_legend,
    df,
    plot_line_with_end_marker,
    plt,
    style_ax,
    symbol_to_color,
    title_axes,
):
    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    for (_ticker,), _sub_df in df.partition_by("Ticker", as_dict=True, maintain_order=True).items():
        plot_line_with_end_marker(
            _ax, _sub_df["Date"].to_numpy(), _sub_df["Close"].to_numpy(), symbol_to_color[_ticker]
        )

    style_ax(_ax, fig=_fig, date_interval=6, y_fmt=lambda v, _: f"${v:,.0f}")
    _ax.set_ylim(0, 1000)
    _handles = [
        plt.Line2D([0], [0], color=symbol_to_color[t], linewidth=2, label=t)
        for t in ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
    ]
    boxed_legend(_ax, handles=_handles)
    title_axes(_ax, "FAANG  ·  Daily Close Price  (2020 – 2024)")
    _fig.tight_layout()
    plt.gcf()
    return


@app.cell
def returns_functions(pl):
    def daily_returns(prices_df):
        """
        Simple daily returns per ticker from the Close column, as a long (Ticker, Date, Return) frame.

        prices_df: long frame with a Ticker, Date, and Close column.
        """
        return prices_df.select("Ticker", "Date", Return=pl.col("Close").pct_change().over("Ticker")).filter(
            pl.col("Return").is_not_null()  # the first day of each ticker has no return
        )

    def returns_matrix(returns_df, tickers):
        """
        Pivot long returns into (dates, matrix), where matrix is (T, n) with columns in the order of tickers.

        returns_df: long (Ticker, Date, Return) frame, from daily_returns.
        tickers: tickers to keep, in the order of the columns of the matrix.
        """
        wide_df = returns_df.pivot(on="Ticker", index="Date")
        return wide_df["Date"].to_numpy(), wide_df.select(tickers).to_numpy()

    return daily_returns, returns_matrix


@app.cell
def compute_returns(daily_returns, df):
    returns_df = daily_returns(df)

    returns_df
    return (returns_df,)


@app.cell
def check_mean_returns(pl, returns_df):
    returns_df.group_by("Ticker").agg(
        pl.mean("Return").alias("mean"),
        pl.std("Return").alias("std"),
        (pl.mean("Return") / pl.std("Return")).alias("mean/std"),
    ).sort("Ticker")
    return


@app.cell
def plot_return_timeseries(
    finish_grid,
    new_grid,
    plt,
    returns_df,
    style_ax,
    symbol_to_color,
    title_axes,
):
    _fig, _axes = new_grid(plt, 2, 3, figsize=(12, 5))
    for _i, ((_ticker,), _sub_df) in enumerate(
        returns_df.partition_by("Ticker", as_dict=True, maintain_order=True).items()
    ):
        _ax = _axes.flat[_i]
        _ax.plot(
            _sub_df["Date"].to_numpy(),
            _sub_df["Return"].to_numpy(),
            color=symbol_to_color[_ticker],
            linewidth=0.9,
            alpha=0.75,
            zorder=2,
        )
        style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4, y_fmt=lambda v, _: f"{v:.1%}")
        title_axes(_ax, _ticker, fontsize=9, pad=5)

    _axes.flat[-1].set_visible(False)
    finish_grid(_fig, "FAANG  ·  Daily Close Return  (2020 – 2024)")
    plt.gcf()
    return


@app.cell
def plot_cumulative_returns(
    boxed_legend,
    pl,
    plot_line_with_end_marker,
    plt,
    reference_line,
    returns_df,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    _cumulative_df = returns_df.with_columns(
        CumReturn=(1 + pl.col("Return")).cum_prod().over("Ticker") - 1,
    )

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _by_ticker = _cumulative_df.partition_by("Ticker", as_dict=True)
    for _ticker in symbols:  # same order as the close price legend
        _sub_df = _by_ticker[(_ticker,)]
        plot_line_with_end_marker(
            _ax, _sub_df["Date"].to_numpy(), _sub_df["CumReturn"].to_numpy(), symbol_to_color[_ticker], label=_ticker
        )

    reference_line(_ax)
    style_ax(_ax, fig=_fig, date_interval=6, y_fmt=lambda v, _: f"{v:.0%}")
    boxed_legend(_ax)
    title_axes(_ax, "FAANG  ·  Cumulative Return  (2020 – 2024)")
    _fig.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def section_fixed_window_ma(mo):
    mo.md(r"""
    # Fixed Window Moving Average
    """)
    return


@app.cell(hide_code=True)
def ma_returns_window_selector(mo):
    ma_returns_window_radio = mo.ui.radio(
        options={"5 days (1 week)": 5, "20 days (1 month)": 20, "60 days (1 quarter)": 60},
        value="20 days (1 month)",
        label="MA window",
        inline=True,
    )
    return (ma_returns_window_radio,)


@app.cell
def plot_ma_returns(
    legend_above_axes,
    ma_returns_window_radio,
    mo,
    pl,
    plt,
    reference_line,
    returns_df,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    _w = ma_returns_window_radio.value

    _returns_ma = returns_df.select(
        "Ticker",
        "Date",
        pl.col("Return").rolling_mean(window_size=_w, min_samples=1).over("Ticker"),
        pl.row_index().over("Ticker").alias("index"),
    ).filter(pl.col("index").ge(_w))
    _by_ticker = _returns_ma.partition_by("Ticker", as_dict=True)

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    for _ticker in symbols:  # same order as the close price legend
        _sub_df = _by_ticker[(_ticker,)]
        _ax.plot(
            _sub_df["Date"].to_numpy(),
            _sub_df["Return"].to_numpy(),
            color=symbol_to_color[_ticker],
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
            label=_ticker,
        )

    reference_line(_ax)
    style_ax(_ax, fig=_fig, date_interval=6, y_fmt=lambda v, _: f"{v:.1%}")
    legend_above_axes(_ax, ncol=5)
    title_axes(_ax, f"FAANG  ·  Close Return MA  (window = {_w}d)")
    _fig.tight_layout()
    mo.vstack([ma_returns_window_radio, _fig])
    return


@app.cell(hide_code=True)
def section_moving_average_formula(mo):
    mo.md(r"""
    ## Moving Average

    $$\hat{\Sigma}_{t+1} = {\alpha_{t+1} \over \alpha_t}\hat{\Sigma}_t + \alpha_{t+1} \left(r_tr_t^T - r_{t-M}r_{t-M}^T\right)$$
    """)
    return


@app.cell
def rolling_window_covariance(np):
    def rolling_window_covariance_estimate_naive(arr: np.ndarray, window: int, demean: bool = False) -> np.ndarray:
        """
        Covariance over a trailing window, recomputed from scratch on every day.

        arr: (T, N) returns, one row per day.
        window: number of days in the window (shorter at the start).
        demean: subtract the window's mean before taking the covariance. Defaults to the zero-mean second moment.

        Returns (N, N, T): entry [..., t] uses days up to and including t.
        """
        T, N = arr.shape
        covariances = []

        for t in range(T):
            samples = arr[max(0, t - window + 1) : t + 1]  # t x N
            r_mean = np.mean(samples, axis=0)  # (N,)
            if demean:
                samples = samples - r_mean

            n_samples, n_stocks = samples.shape
            cov_t = 1 / n_samples * samples.T @ samples  # N x N
            covariances.append(cov_t)

        return np.stack(covariances, axis=-1)

    def rolling_window_covariance_estimate(arr: np.ndarray, window: int) -> np.ndarray:
        """
        Zero-mean rolling window second moment, updated recursively as a day enters and leaves the window.

        arr: (T, N) returns, one row per day.
        window: number of days in the window (shorter at the start).

        Returns (N, N, T): entry [..., t] uses days up to and including t.
        """
        T, N = arr.shape

        r_last = np.zeros((N,))
        r_t = arr[0]
        cov_t = np.outer(r_t, r_t)
        covariances = [cov_t]

        for t in range(1, T):
            if t >= window:
                r_last = arr[t - window]

            r_t = arr[t]
            alpha_curr = 1 / min(t + 1, window)
            alpha_prev = 1 / min(t, window)  # inf to start, but we don't use it
            cov_t = alpha_curr / alpha_prev * cov_t + alpha_curr * (np.outer(r_t, r_t) - np.outer(r_last, r_last))

            covariances.append(cov_t)

        return np.stack(covariances, axis=-1)

    return (
        rolling_window_covariance_estimate,
        rolling_window_covariance_estimate_naive,
    )


@app.cell
def prepare_returns_array(returns_df, returns_matrix, symbols):
    returns_dates, returns_arr = returns_matrix(returns_df, symbols)  # (T,), (T, n)
    return returns_arr, returns_dates


@app.cell(hide_code=True)
def ma_window_selector(mo):
    ma_window_radio = mo.ui.radio(
        options={"5 days (1 week)": 5, "20 days (1 month)": 20, "60 days (1 quarter)": 60},
        value="20 days (1 month)",
        label="MA window",
        inline=True,
    )
    return (ma_window_radio,)


@app.cell
def plot_ma_std_dev_comparison(
    diagonal_std_devs,
    finish_grid,
    ma_window_radio,
    mo,
    new_grid,
    plt,
    returns_arr,
    returns_dates,
    rolling_window_covariance_estimate_naive,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    _DIM = "#6b7280"
    _ma_window = ma_window_radio.value

    def _std_devs(demean):
        _cov = rolling_window_covariance_estimate_naive(returns_arr, window=_ma_window, demean=demean)
        return diagonal_std_devs(_cov)  # (stocks, time)

    _std_raw = _std_devs(demean=False)
    _std_dm = _std_devs(demean=True)
    _ymax = 1.05 * max(_std_raw.max(), _std_dm.max())  # shared across panels, so nothing is cut off

    _fig, _axes = new_grid(plt, 2, 3, figsize=(12, 5))
    for _i, _symbol in enumerate(symbols):
        _ax = _axes.flat[_i]
        _c = symbol_to_color[_symbol]
        _ax.plot(returns_dates, _std_raw[_i], color=_c, linewidth=1.0, alpha=0.9, zorder=2, label="raw")
        _ax.plot(
            returns_dates,
            _std_dm[_i],
            color=_c,
            linewidth=1.0,
            alpha=0.55,
            linestyle="--",
            zorder=2,
            label="demeaned",
        )
        style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4, y_fmt=lambda v, _: f"{v:.1%}")
        _ax.set_ylim(0, _ymax)
        title_axes(_ax, _symbol, fontsize=9, pad=5)
        if _i == 0:
            _ax.legend(frameon=False, fontsize=7, labelcolor=_DIM)

    _axes.flat[-1].set_visible(False)
    finish_grid(_fig, f"FAANG  ·  Rolling Std Dev  (window = {_ma_window}d)")
    mo.vstack([ma_window_radio, _fig])
    return


@app.cell
def section_ewma_formula(mo):
    mo.md(r"""
    ## Exponentially Weighted Moving Average

    $$\hat{\Sigma}_{t+1} = {\beta - \beta^t \over 1-\beta^t}\hat{\Sigma}_t + {1-\beta\over 1- \beta^t} r_tr_t^T $$
    """)
    return


@app.cell
def ewma_functions(np):
    def _regularize_covariance(cov: np.ndarray, shrinkage_alpha: float = 0.01):
        cov = (1 - shrinkage_alpha) * cov + shrinkage_alpha * np.diag(np.diag(cov))  # ensure full rank
        return cov

    def ewma_cov(samples, beta):
        """
        Zero-mean EWMA second moment of one set of samples, as in the paper (matches ewma_covariance_estimate).

        samples: (T, N) returns, oldest row first.
        beta: forgetting factor in (0, 1).

        Returns the (N, N) estimate at the last row.
        """
        T, N = samples.shape
        weights = (beta ** np.arange(T))[::-1]
        factor = (1 - beta) / (1 - beta**T)
        cov_t = factor * (weights[:, None] * samples).T @ samples  # (N, N)
        return cov_t

    def ewma_covariance_estimate_naive(arr, beta: float, shrinkage_alpha: float = 0.01):
        """
        EWMA covariance recomputed from scratch on every day. O(T^3), so it is only a check on the fast version.

        arr: (T, N) returns, one row per day.
        beta: forgetting factor in (0, 1).
        shrinkage_alpha: weight on the diagonal of the estimate, which keeps it full rank.

        Returns (N, N, T): entry [..., t] uses days up to and including t.
        """
        T, N = arr.shape
        covariances = []

        for t in range(1, T + 1):
            samples = arr[:t]  # (t, N)
            cov_t = ewma_cov(samples, beta)
            cov_t = _regularize_covariance(cov_t, shrinkage_alpha)
            covariances.append(cov_t)

        return np.stack(covariances, axis=-1)

    def ewma_covariance_estimate(arr: np.ndarray, beta: float, shrinkage_alpha: float = 0.01):
        """
        EWMA covariance, updated recursively.

        arr: (T, N) returns, one row per day.
        beta: forgetting factor in (0, 1).
        shrinkage_alpha: weight on the diagonal of the estimate, which keeps it full rank.

        Returns (N, N, T): entry [..., t] uses days up to and including t.
        """
        T, N = arr.shape

        r_t = arr[0]
        cov_t = np.outer(r_t, r_t)  # raw recursion state, never shrunk

        # shrink only the stored output, so every step is regularized like the naive version
        covariances = [_regularize_covariance(cov_t, shrinkage_alpha)]

        for t in range(1, T):
            r_t = arr[t]

            cov_t = (beta - beta ** (t + 1)) / (1 - beta ** (t + 1)) * cov_t
            cov_t += (1 - beta) / (1 - beta ** (t + 1)) * np.outer(r_t, r_t)  # (N, N)

            covariances.append(_regularize_covariance(cov_t, shrinkage_alpha))

        return np.stack(covariances, axis=-1)

    def equivalent_window(half_life: float) -> int:
        """Rolling-window length with the same effective sample size as an EWMA of this half-life.

        Normalized EWMA weights have 1 / sum(w^2) = (1 + beta) / (1 - beta) effective samples,
        and a rolling window of length M has M, so this makes the two estimators comparable.
        """
        beta = 0.5 ** (1 / half_life)
        return round((1 + beta) / (1 - beta))

    return (
        equivalent_window,
        ewma_covariance_estimate,
        ewma_covariance_estimate_naive,
    )


@app.cell
def check_ewma_naive_matches_fast(
    ewma_covariance_estimate,
    ewma_covariance_estimate_naive,
    np,
    returns_arr,
):
    # the naive O(T^3) version is only a correctness check, so run it on a short slice
    _slice = returns_arr[:200]
    for _half_life in [5, 20, 60]:
        _beta = 0.5 ** (1 / _half_life)
        np.testing.assert_allclose(
            ewma_covariance_estimate_naive(_slice, _beta),
            ewma_covariance_estimate(_slice, _beta),
            atol=1e-12,
        )
    return


@app.cell(hide_code=True)
def section_ewma_vs_ma(mo):
    mo.md(r"""
    ### Comparing EWMA and a rolling window

    An EWMA is parameterized by a half-life $H$ and a rolling window by a length $M$. These are not
    the same kind of "memory", so matching $M = H$ would be misleading: the EWMA still puts weight on
    data far older than $H$ days, while the window drops it entirely.

    A fairer comparison matches how *noisy* the two estimators are. Any estimator of the form
    $\hat{\Sigma} = \sum_i w_i r_i r_i^T$ with weights summing to one has variance proportional to
    the sum of squared weights (for i.i.d. returns),

    $$\mathbf{var}(\hat{\Sigma}) \propto \sum_i w_i^2 \quad\Longleftrightarrow\quad N_{\text{eff}} = {1 \over \sum_i w_i^2},$$

    where $N_{\text{eff}}$ is the *effective sample size*. For a rolling window, $w_i = 1/M$ on $M$
    samples, so $N_{\text{eff}} = M$. For the EWMA, the normalized weights are
    $w_i = (1-\beta)\beta^i$ (with $i$ the age of the sample), so

    $$\sum_i w_i^2 = (1-\beta)^2 \sum_{i \ge 0} \beta^{2i} = {(1-\beta)^2 \over 1-\beta^2} = {1-\beta \over 1+\beta}
    \quad\Longrightarrow\quad N_{\text{eff}} = {1+\beta \over 1-\beta}, \qquad \beta = 2^{-1/H}.$$

    Setting $M = N_{\text{eff}}$ gives a rolling window whose estimates are about as noisy as the EWMA's.
    For the half-lives below this gives $H = 5, 20, 60 \Rightarrow M = 14, 58, 173$ days. Any remaining
    difference in the plots is then about *shape* (smooth decay vs. a hard cutoff), not noise level:
    the window shows abrupt jumps when a large return enters and, $M$ days later, leaves.
    """)
    return


@app.cell(hide_code=True)
def ewma_half_life_selector(mo):
    ewma_half_life_radio = mo.ui.radio(
        options={"5 days": 5, "20 days": 20, "60 days": 60},
        value="20 days",
        label="EWMA half-life",
        inline=True,
    )
    return (ewma_half_life_radio,)


@app.cell
def plot_ewma_vs_ma_std_dev(
    diagonal_std_devs,
    equivalent_window,
    ewma_covariance_estimate,
    ewma_half_life_radio,
    finish_grid,
    mo,
    new_grid,
    plt,
    returns_arr,
    returns_dates,
    rolling_window_covariance_estimate,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    _DIM = "#6b7280"
    _half_life = ewma_half_life_radio.value

    _window = equivalent_window(_half_life)  # same effective sample size as the EWMA
    _cov_ewma = ewma_covariance_estimate(returns_arr, beta=0.5 ** (1 / _half_life))
    _cov_ma = rolling_window_covariance_estimate(returns_arr, window=_window)
    _std_ewma = diagonal_std_devs(_cov_ewma)  # (stocks, time)
    _std_ma = diagonal_std_devs(_cov_ma)
    _ymax = 1.05 * max(_std_ewma.max(), _std_ma.max())  # shared across panels, so nothing is cut off

    _fig, _axes = new_grid(plt, 2, 3, figsize=(12, 5))
    for _i, _symbol in enumerate(symbols):
        _ax = _axes.flat[_i]
        _c = symbol_to_color[_symbol]
        _ax.plot(returns_dates, _std_ewma[_i], color=_c, linewidth=1.0, alpha=0.9, zorder=2, label="ewma")
        _ax.plot(
            returns_dates,
            _std_ma[_i],
            color=_c,
            linewidth=1.0,
            alpha=0.55,
            linestyle="--",
            zorder=2,
            label=f"ma ({_window}d)",
        )
        style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4, y_fmt=lambda v, _: f"{v:.1%}")
        _ax.set_ylim(0, _ymax)
        title_axes(_ax, _symbol, fontsize=9, pad=5)
        if _i == 0:
            _ax.legend(frameon=False, fontsize=7, labelcolor=_DIM)

    _axes.flat[-1].set_visible(False)
    finish_grid(_fig, rf"FAANG  ·  EWMA vs MA Std Dev  ($H^{{vol}}$ = {_half_life}d, MA window = {_window}d)")
    mo.vstack([ewma_half_life_radio, _fig])
    return


@app.cell
def plot_full_covariance_grid(
    ewma_covariance_estimate,
    finish_grid,
    new_grid,
    plt,
    returns_arr,
    returns_dates,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    import itertools

    _INK = "#111827"

    _half_life = 20
    _cov_ewma = ewma_covariance_estimate(returns_arr, beta=0.5 ** (1 / _half_life))

    _fig, _axes = new_grid(plt, 5, 5, figsize=(14, 12))

    for _i, (_symbol_1, _symbol_2) in enumerate(itertools.product(symbols, symbols)):
        x, y = divmod(_i, 5)
        _ax = _axes[x, y]
        _c = _INK if _symbol_1 != _symbol_2 else symbol_to_color[_symbol_1]

        _ax.plot(returns_dates, _cov_ewma[x, y, :], color=_c, linewidth=0.9, alpha=0.85, zorder=2)
        style_ax(_ax, date_interval=24, labelsize=6.5, tick_length=3, pad=3)
        _ax.set_ylim(0, 0.005)

        if x == 0:
            title_axes(_ax, _symbol_2, fontsize=8, pad=4)
        if y == 0:
            _ax.set_ylabel(_symbol_1, color=_INK, fontsize=8, fontweight="bold")

    finish_grid(_fig, f"FAANG  ·  EWMA Covariance  (H = {_half_life}d)", top=0.97)
    plt.gcf()
    return (itertools,)


@app.cell
def define_iterated_ewma(ewma_covariance_estimate, np):
    def _one_step_ahead(x: np.ndarray) -> np.ndarray:
        """Shift the last axis right by one so x[..., t] only uses data through t-1.

        The EWMA recursion at index t has already absorbed r_t (a nowcast), but the paper's
        predictor for day t may only see r_1, ..., r_{t-1}. Day 0 has no history, so it
        falls back to the first estimate.
        """
        return np.concatenate([x[..., :1], x[..., :-1]], axis=-1)

    def iterated_ewma(returns: np.ndarray, h_vol: float, h_corr: float, clip_at: float | None = 4.2) -> np.ndarray:
        """
        IEWMA predictor. Returns covariance predictions of shape (N, N, T), where
        entry [..., t] is the prediction for day t made from returns[:t] only.

        returns: (T, N) returns, one row per day.
        h_vol: half-life in days of the EWMA that estimates each stock's volatility.
        h_corr: half-life in days of the EWMA that estimates the correlations of the standardized returns.
        clip_at winsorizes the standardized returns at +-clip_at before the correlation EWMA
        (the paper's rule of thumb is 4.2). Pass None to disable; the paper's reference code
        (cvxgrp/cov_pred_finance) does not winsorize in its stock experiment.
        """
        covs = ewma_covariance_estimate(returns, beta=0.5 ** (1 / h_vol))  # (N, N, T)
        std_devs = _one_step_ahead(np.diagonal(covs, axis1=0, axis2=1).T ** 0.5)  # (N, T)

        returns_standardized = returns / std_devs.T  # (T, N), sigma_t excludes r_t
        winsorized_returns = (
            returns_standardized if clip_at is None else np.clip(returns_standardized, a_min=-clip_at, a_max=clip_at)
        )

        R = ewma_covariance_estimate(winsorized_returns, 0.5 ** (1 / h_corr))  # (N, N, T)
        std_devs_R = np.diagonal(R, axis1=0, axis2=1).T ** 0.5  # (N, T)
        R_corr = R / std_devs_R[None, :, :] / std_devs_R[:, None, :]  # (N, N, T)
        R_corr = _one_step_ahead(R_corr)

        return std_devs[None, :, :] * R_corr * std_devs[:, None, :]

    return (iterated_ewma,)


@app.cell
def define_plot_covariance_matrices(
    finish_grid,
    itertools,
    new_grid,
    np,
    plt,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    _INK = "#111827"
    _DIM = "#6b7280"

    def plot_covariance_matrices(cov_mat: np.ndarray, dates: np.ndarray, h_vol: float, h_corr: float):
        _fig, _axes = new_grid(plt, 5, 5, figsize=(14, 12))

        for _i, (_symbol_1, _symbol_2) in enumerate(itertools.product(symbols, symbols)):
            x, y = divmod(_i, 5)
            _ax = _axes[x, y]

            if x > y:
                _ax.set_visible(False)
                continue

            is_off_diag = _symbol_1 != _symbol_2
            _c = _INK if is_off_diag else symbol_to_color[_symbol_1]

            std_dev_1 = cov_mat[x, x, :] ** 0.5
            std_dev_2 = cov_mat[y, y, :] ** 0.5
            _ts = cov_mat[x, y, :] / (std_dev_1 * std_dev_2) if is_off_diag else cov_mat[x, y, :] ** 0.5

            _ax.plot(dates, _ts, color=_c, linewidth=0.9, alpha=0.85, zorder=2)
            _y_fmt = None if is_off_diag else lambda v, _: f"{v:.1%}"
            style_ax(_ax, date_interval=24, labelsize=6.5, tick_length=3, pad=3, y_fmt=_y_fmt)
            _ax.set_ylim(0, 1 if is_off_diag else 0.06)

            if y == x:
                _ax.set_ylabel(_symbol_1, color=_INK, fontsize=8, fontweight="bold")
            if x == 0:
                title_axes(_ax, _symbol_2, fontsize=8, pad=4)

        finish_grid(_fig, rf"FAANG  ·  IEWMA Correlation & Std Dev  ($H^{{vol}}$ = {h_vol}d, $H^{{cor}}$ = {h_corr}d)", top=0.97)
        return _fig

    return (plot_covariance_matrices,)


@app.cell
def run_iterated_ewma(
    iterated_ewma,
    plot_covariance_matrices,
    returns_arr,
    returns_dates,
):
    _h_vol, _h_corr = 20, 60
    _cov_mat = iterated_ewma(returns_arr, h_vol=_h_vol, h_corr=_h_corr)

    plot_covariance_matrices(_cov_mat, returns_dates, _h_vol, _h_corr)
    return


@app.cell(hide_code=True)
def section_hierarchical_residuals(mo):
    mo.md(r"""
    # Hierarchical Residuals (Market, Sector, Idiosyncratic)

    A different way to build a covariance matrix is to peel off layers of common variation one at a time,
    in a fixed economic order, much like Gram-Schmidt. Let $r_t \in \mathbf{R}^n$ be the returns of $n$ stocks.

    ## Layer 1: market

    Let $R \in \mathbf{R}^{T \times n}$ be the matrix of returns and $m = \tfrac{1}{n} R\mathbf{1} \in \mathbf{R}^T$ the
    equal-weighted market return. Regress every stock on the market, with no intercept (second moments, as in the
    rest of this notebook):

    $$\underset{\beta}{\text{minimize}}\quad \|R - m\beta^T\|_F^2, \qquad \beta \in \mathbf{R}^n .$$

    Column $i$ of $m\beta^T$ is $\beta_i m$, so this is $n$ separate scalar regressions, one per stock. The solution is

    $$\beta = {R^T m \over m^T m}, \qquad \text{that is,} \qquad \beta_i = {\mathbf{cov}(r_i, m) \over \mathbf{var}(m)} = {\sum_t r_{i,t}\, m_t \over \sum_t m_t^2},$$

    and the residual matrix, with entries $e^{(1)}_{i,t} = r_{i,t} - \beta_i m_t$, is

    $$E = R - m\beta^T = \left(I - {m m^T \over m^T m}\right) R \in \mathbf{R}^{T \times n}.$$

    Each column of $E$ is exactly orthogonal to $m$, that is $m^T E = 0$. This is one Gram-Schmidt step with $m$ as
    the first vector, and the residual is the part of each stock's return that the market does not explain. The means
    are not subtracted (the zero-mean convention used in the rest of this notebook), which moves the betas by less
    than 0.01 on this data.
    """)
    return


@app.cell
def plot_market_residual_series(
    market_factor,
    market_residual_returns,
    new_grid,
    np,
    plt,
    robust_ylim,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
    title_figure,
    universe,
    universe_dates,
):
    _DIM = "#6b7280"

    # first the market return itself, then the part of each FAANG stock's return that the market does not explain
    _panels = [("Market", market_factor, "#374151")] + [
        (_ticker, market_residual_returns[:, universe.index(_ticker)], symbol_to_color[_ticker]) for _ticker in symbols
    ]
    _all_series = np.stack([_series for _, _series, _ in _panels])

    # shared y-limits cover the central 99.5% of points, so a handful of extreme days do not squash the rest
    _ylim, _n_clipped = robust_ylim(_all_series)

    _fig, _axes = new_grid(plt, 2, 3, figsize=(12, 5), sharey=True)
    for _ax, (_name, _series, _color) in zip(_axes.flat, _panels):
        _ax.plot(universe_dates, _series, color=_color, linewidth=0.9, alpha=0.75, zorder=2)
        style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4, y_fmt=lambda v, _: f"{v:.0%}")
        title_axes(_ax, f"{_name}   std = {_series.std():.1%}", fontsize=9, pad=5)
    _axes.flat[0].set_ylim(-_ylim, _ylim)

    title_figure(_fig, r"Daily market return and FAANG market residuals $r_i - \beta_i m$")
    _fig.text(0.0, 0.005, f"{_n_clipped} extreme points are outside the y-limits.", color=_DIM, fontsize=8, ha="left")
    _fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.gcf()
    return


@app.cell
def plot_market_fit(
    market_betas,
    market_factor,
    market_residual_returns,
    new_grid,
    np,
    plt,
    robust_ylim,
    second_moment_r_squared,
    style_plain_ax,
    symbol_to_color,
    symbols,
    title_axes,
    title_figure,
    universe,
    universe_returns_arr,
):
    _INK = "#111827"
    _DIM = "#6b7280"
    _pct = lambda v, _: f"{v:.0%}"

    _fig, _axes = new_grid(plt, 1, len(symbols), figsize=(14, 3.6), sharex=True, sharey=True)
    _x_line = np.array([market_factor.min(), market_factor.max()])

    # y-limits cover the central 99.5% of points, so a handful of extreme days do not squash the rest
    _idx = [universe.index(_t) for _t in symbols]
    _ylim, _n_clipped = robust_ylim(universe_returns_arr[:, _idx])

    for _ax, _ticker in zip(_axes, symbols):
        _i = universe.index(_ticker)
        _r = universe_returns_arr[:, _i]
        _e = market_residual_returns[:, _i]
        _r_squared = second_moment_r_squared(_e, _r)

        # the stock's return against the market, with the fitted line r = beta * m
        _ax.scatter(market_factor, _r, s=6, color=symbol_to_color[_ticker], alpha=0.4, linewidths=0, zorder=2)
        _ax.plot(_x_line, market_betas[_i] * _x_line, color=_INK, linewidth=1.2, zorder=3)
        title_axes(_ax, rf"{_ticker}   $\beta$ = {market_betas[_i]:.2f},  $R^2$ = {_r_squared:.0%}", fontsize=9, pad=6)
        _ax.set_xlabel("market return", color=_DIM, fontsize=8)
        style_plain_ax(_ax, x_fmt=_pct, y_fmt=_pct)

    _axes[0].set_ylim(-_ylim, _ylim)
    _axes[0].set_ylabel("stock return", color=_DIM, fontsize=8)

    title_figure(
        _fig, r"FAANG  ·  How well does $r_i = \beta_i m$ fit?  ($m$ is the equal-weighted market return)", fontsize=10.5
    )
    _fig.text(0.0, 0.005, f"{_n_clipped} extreme points are outside the y-limits.", color=_DIM, fontsize=8, ha="left")
    _fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.gcf()
    return


@app.cell
def hierarchical_residual_functions(np, pl):
    def project_out(returns: np.ndarray, factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project every column of returns onto factor (no intercept) and remove it, one Gram-Schmidt step.

        returns: (T, n) matrix with one column per stock.
        factor: (T,) series to project out, for example the market return.

        Returns (betas, residuals), with betas of shape (n,) and residuals of shape (T, n):
            betas[i] = <returns[:, i], factor> / <factor, factor>,   residuals[:, i] = returns[:, i] - betas[i] * factor,
        and every residual column is orthogonal to factor.
        """
        betas = factor @ returns / (factor @ factor)
        return betas, returns - np.outer(factor, betas)

    def rolling_betas(returns: np.ndarray, factor: np.ndarray, window: int) -> np.ndarray:
        """
        Look-ahead-free version of project_out's betas, as a time series of shape (T, n).

        returns: (T, n) matrix with one column per stock.
        factor: (T,) series the stocks are regressed on, for example the market return.
        window: number of past days each beta is estimated from.

        betas[t, i] uses only days t - window, ..., t - 1, so it is known before day t. Until a full window
        of history exists (t < window) the beta is set to 1.
        """
        # the sum over days t - window, ..., t - 1 is the rolling sum ending at day t - 1, shifted down one row;
        # rows without a full window are null
        cross = pl.DataFrame(returns * factor[:, None], orient="row").select(pl.all().rolling_sum(window).shift(1))
        square = pl.Series(factor**2).rolling_sum(window).shift(1)

        betas = cross.to_numpy() / square.to_numpy()[:, None]
        betas[:window] = 1.0
        return betas

    def second_moment_r_squared(residual: np.ndarray, response: np.ndarray) -> float:
        """
        Share of the response's second moment explained by a no-intercept fit, 1 - ||residual||^2 / ||response||^2.

        residual, response: arrays of the same shape, the series that was fit and what is left of it after the fit.
        """
        return 1 - (residual**2).sum() / (response**2).sum()

    def sector_indices(universe: list[str], sector_of: dict[str, str]) -> dict[str, np.ndarray]:
        """
        Map each sector to the indices of its stocks in universe, in the order the sectors are listed.

        universe: tickers, in the order of the columns of the returns matrix.
        sector_of: sector of each ticker.
        """
        return {
            sector: np.array([i for i, ticker in enumerate(universe) if sector_of[ticker] == sector])
            for sector in dict.fromkeys(sector_of.values())
        }

    def sectors_by_size(sector_rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        The same mapping with the largest sector first, so that sector blocks shrink down the diagonal of a heatmap.

        sector_rows: sector name -> indices of that sector's stocks, from sector_indices. Ties keep their order.
        """
        return dict(sorted(sector_rows.items(), key=lambda item: -len(item[1])))

    def project_out_sector_factors(
        residuals: np.ndarray,
        sector_rows: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Second layer: within each sector, the factor is the average of the stocks' residuals, and each stock's
        residual is projected onto it.

        residuals: (T, n) returns after the market layer.
        sector_rows: sector name -> indices of that sector's stocks (columns of residuals), from sector_indices.

        Returns (sector_factors, betas, residuals): a (T,) factor for every sector, the (n,) slope of every stock on
        its own sector's factor, and the (T, n) residuals after removing the sector factors.
        """
        betas = np.zeros(residuals.shape[1])
        new_residuals = residuals.copy()
        sector_factors = {}
        for sector, rows in sector_rows.items():
            factor = residuals[:, rows].mean(axis=1)
            betas[rows], new_residuals[:, rows] = project_out(residuals[:, rows], factor)
            sector_factors[sector] = factor
        return sector_factors, betas, new_residuals

    def variance_shares(
        returns: np.ndarray,
        market_residuals: np.ndarray,
        sector_residuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Share of each stock's second-moment variance explained by the market, sector, and idiosyncratic layers.

        returns: (T, n) returns.
        market_residuals: (T, n) returns after the market layer.
        sector_residuals: (T, n) returns after the market and sector layers.

        Returns three (n,) arrays: the market, sector, and idiosyncratic shares. The three parts
        (returns - market_residuals, market_residuals - sector_residuals, sector_residuals) are mutually
        orthogonal, so the shares add up to exactly 1 for every stock.
        """
        total = (returns**2).mean(axis=0)
        parts = (returns - market_residuals, market_residuals - sector_residuals, sector_residuals)
        market_share, sector_share, idiosyncratic_share = ((part**2).mean(axis=0) / total for part in parts)
        return market_share, sector_share, idiosyncratic_share

    def sector_blocks(sector_rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Order that groups stocks by sector, with the start and end position of each sector's block in that order.

        sector_rows: sector name -> indices of that sector's stocks, from sector_indices.
        """
        order = np.concatenate(list(sector_rows.values()))
        block_ends = np.cumsum([len(rows) for rows in sector_rows.values()])
        return order, np.concatenate([[0], block_ends[:-1]]), block_ends

    def same_sector_mask(universe: list[str], sector_of: dict[str, str]) -> np.ndarray:
        """
        (n, n) boolean matrix that is True for pairs of stocks in the same sector (including i = j).

        universe: tickers, in the order of the rows and columns of the matrix.
        sector_of: sector of each ticker.
        """
        sectors = np.array([sector_of[ticker] for ticker in universe])
        return sectors[:, None] == sectors[None, :]

    def within_and_across(corr: np.ndarray, same_sector: np.ndarray) -> tuple[float, float]:
        """
        Average correlation of distinct pairs in the same sector, and of pairs in different sectors.

        corr: (n, n) correlation matrix.
        same_sector: (n, n) boolean matrix from same_sector_mask.
        """
        off_diagonal = ~np.eye(len(corr), dtype=bool)
        return corr[same_sector & off_diagonal].mean(), corr[~same_sector].mean()

    def first_principal_component(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        First principal component of the uncentered returns, matching the second-moment convention above.

        returns: (T, n) matrix with one column per stock.

        Returns (loadings, scores, explained variance ratio), with unit loadings of shape (n,) whose sign is
        fixed so that they sum to a positive number, and scores of shape (T,).
        """
        _, singular_values, Vt = np.linalg.svd(returns, full_matrices=False)
        loadings = Vt[0] * np.sign(Vt[0].sum())
        return loadings, returns @ loadings, singular_values[0] ** 2 / np.sum(singular_values**2)

    def rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """
        Correlation of x and y over each trailing window of the given length, starting at day window - 1.

        x, y: (T,) series.
        window: number of days in each window.
        """
        rolling = pl.DataFrame({"x": x, "y": y}).select(pl.rolling_corr("x", "y", window_size=window))
        return rolling.to_numpy().ravel()[window - 1 :]

    return (
        first_principal_component,
        project_out,
        project_out_sector_factors,
        rolling_betas,
        rolling_correlation,
        same_sector_mask,
        second_moment_r_squared,
        sector_blocks,
        sector_indices,
        sectors_by_size,
        variance_shares,
        within_and_across,
    )


@app.cell
def market_layer(project_out, universe_returns_arr):
    market_factor = universe_returns_arr.mean(axis=1)  # equal-weighted market return, (T,)
    market_betas, market_residual_returns = project_out(universe_returns_arr, market_factor)
    return market_betas, market_factor, market_residual_returns


@app.cell
def plot_market_vs_pc1(
    first_principal_component,
    market_factor,
    market_residual_returns,
    new_grid,
    np,
    plt,
    rolling_correlation,
    style_ax,
    style_plain_ax,
    title_axes,
    title_figure,
    universe_dates,
    universe_returns_arr,
):
    _DIM = "#6b7280"
    _R = universe_returns_arr
    _window = 63

    _loadings, _pc1, _explained = first_principal_component(_R)
    _cos_equal_weights = _loadings.sum() / np.sqrt(len(_loadings))  # cosine between PC1 loadings and the equal-weight market
    _corr = np.corrcoef(market_factor, _pc1)[0, 1]
    _residual_corr = np.array([np.corrcoef(_e, _pc1)[0, 1] for _e in market_residual_returns.T])
    _rolling = rolling_correlation(market_factor, _pc1, _window)

    _fig, (_ax_scatter, _ax_roll) = new_grid(plt, 1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2]})

    _ax_scatter.scatter(market_factor, _pc1, s=6, color="tab:blue", alpha=0.35, linewidths=0, zorder=2)
    _ax_scatter.set_xlabel("market factor (equal-weighted return)", color=_DIM, fontsize=8.5)
    _ax_scatter.set_ylabel("PC1 score", color=_DIM, fontsize=8.5)
    style_plain_ax(_ax_scatter, labelsize=8, x_fmt=lambda v, _: f"{v:.0%}")
    title_axes(_ax_scatter, f"corr = {_corr:.3f}", fontsize=9, pad=6)

    _ax_roll.plot(universe_dates[_window - 1 :], _rolling, color="tab:blue", linewidth=1.2, zorder=2)
    style_ax(_ax_roll, date_interval=6, labelsize=8, y_fmt=lambda v, _: f"{v:.2f}")
    _ax_roll.set_ylim(min(0.9, _rolling.min() - 0.01), 1.0)
    title_axes(_ax_roll, f"{_window}-day rolling correlation", fontsize=9, pad=6)

    title_figure(_fig, "Market return vs first principal component of the returns")
    _fig.text(
        0.0,
        0.005,
        f"PC1 explains {_explained:.0%} of the variance. Cosine between its loadings and equal weights: "
        f"{_cos_equal_weights:.2f}. Mean |corr(market residual, PC1)|: {np.abs(_residual_corr).mean():.2f}.",
        color=_DIM,
        fontsize=8,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.gcf()
    return


@app.cell(hide_code=True)
def section_lookahead_free_beta(mo):
    mo.md(r"""
    ### A look-ahead-free version

    The fit above uses one $\beta_i$ per stock, estimated on the whole sample, so the residual on day $t$ uses
    information from days after $t$. A predictor cannot do that. Instead, let $\beta$ be a time series that is
    estimated from the previous $W$ days only ($W = 40$ unless you change it in the plot below):

    $$\beta_{i,t} = {\mathbf{cov}(r_i, m) \over \mathbf{var}(m)} = {\sum_{s=t-W}^{t-1} r_{i,s}\, m_s \over \sum_{s=t-W}^{t-1} m_s^2}, \qquad e^{(1)}_{i,t} = r_{i,t} - \beta_{i,t}\, m_t ,$$

    where $\mathbf{cov}$ and $\mathbf{var}$ are second moments over the window. The window stops at day $t-1$, so
    $\beta_{i,t}$ is known before day $t$'s returns arrive. The market return $m_t$ is still the one observed on day $t$,
    as in any factor decomposition of that day's returns.

    For the first $W$ days there is no full window, so we set $\beta_{i,t} = 1$. This assumes the stock moves
    one-for-one with the market, so its residual is $r_{i,t} - m_t$. It is a natural default, because $m$ is the
    equal-weighted average of the stocks and so the cross-sectional average of the betas is exactly $1$ in every window.
    Setting $\beta = 1$ imputes the average stock.
    """)
    return


@app.cell(hide_code=True)
def beta_window_selector(mo):
    beta_window_radio = mo.ui.radio(
        options={"20 days": 20, "40 days": 40, "63 days (1 quarter)": 63, "126 days (6 months)": 126},
        value="40 days",
        label="Beta window",
        inline=True,
    )
    return (beta_window_radio,)


@app.cell
def rolling_market_beta(
    beta_window_radio,
    market_factor,
    np,
    rolling_betas,
    universe_returns_arr,
):
    beta_window_days = beta_window_radio.value

    rolling_market_betas = rolling_betas(universe_returns_arr, market_factor, beta_window_days)  # (T, n)
    rolling_market_residual_returns = universe_returns_arr - rolling_market_betas * market_factor[:, None]

    # m is the average of the stocks, so the betas average to exactly 1 in every window, which is why beta = 1 is the default
    np.testing.assert_allclose(rolling_market_betas.mean(axis=1), 1.0)
    return beta_window_days, rolling_market_betas


@app.cell
def plot_rolling_market_betas(
    beta_window_days,
    beta_window_radio,
    legend_above_axes,
    market_betas,
    mo,
    plt,
    reference_line,
    rolling_market_betas,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
    universe,
    universe_dates,
):
    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _ax.axvspan(universe_dates[0], universe_dates[beta_window_days - 1], color="#e5e7eb", alpha=0.7, linewidth=0, zorder=0)
    reference_line(_ax, 1)
    for _ticker in symbols:  # same order as the close price legend
        _i = universe.index(_ticker)
        _c = symbol_to_color[_ticker]
        _ax.plot(universe_dates, rolling_market_betas[:, _i], color=_c, linewidth=1.2, alpha=0.9, zorder=2, label=_ticker)
        _ax.axhline(market_betas[_i], color=_c, linewidth=0.9, linestyle=":", alpha=0.8, zorder=1)

    style_ax(_ax, fig=_fig, date_interval=6, y_fmt=lambda v, _: f"{v:.1f}")
    legend_above_axes(_ax, ncol=5)
    title_axes(_ax, rf"FAANG  ·  Rolling Market Beta  ($\beta_{{i,t}}$, {beta_window_days}-day window)")
    _fig.text(
        0.0,
        0.005,
        rf"Dotted lines: full-sample $\beta_i$. Shaded: the first {beta_window_days} days, where $\beta_{{i,t}} = 1$.",
        color="#6b7280",
        fontsize=8,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0.03, 1, 1])
    mo.vstack([beta_window_radio, _fig])
    return


@app.cell(hide_code=True)
def section_correlation_structure(mo):
    mo.md(r"""
    ### Is there structure beyond the market?

    The market layer removes one common factor. Is there more structure in the correlations? Below are correlation
    matrices of the daily returns of all the stocks over a 50-day window, and the slider moves the window. On the left
    the stocks are in an arbitrary (random) order. In the middle the same matrix is reordered to group stocks of the
    same sector together, with the sector boundaries marked. If stocks in the same sector move together more than
    stocks in different sectors, the reordered matrix shows blocks along the diagonal, which is the kind of structure
    that a sector factor can capture.

    On the right is the same sector ordering for the market residuals $e^{(1)}$ from Layer 1, on the same color scale.
    Removing the market takes away the common level that every stock shares, and what remains is how stocks move
    relative to the market. Because the residuals sum to zero across stocks, their average pairwise correlation is
    slightly negative. Fifty days is short for 75 stocks (there are fewer days than stocks), so these estimates are noisy, and the residuals use the
    full-sample betas from Layer 1.
    """)
    return


@app.cell(hide_code=True)
def correlation_window_selector(mo, universe_returns_arr):
    correlation_window_days = 50
    correlation_end_slider = mo.ui.slider(
        start=correlation_window_days,
        stop=universe_returns_arr.shape[0],
        step=1,
        value=universe_returns_arr.shape[0],
        label="Window ends at day",
        show_value=True,
        full_width=True,
    )
    return correlation_end_slider, correlation_window_days


@app.cell
def plot_correlation_heatmaps(
    correlation_end_slider,
    correlation_window_days,
    draw_correlation_matrix,
    label_sector_blocks,
    mark_sector_blocks,
    market_residual_returns,
    mo,
    np,
    plt,
    same_sector_mask,
    sector_blocks,
    sector_indices,
    sector_of,
    sectors_by_size,
    symbols,
    title_figure,
    universe,
    universe_dates,
    universe_returns_arr,
    within_and_across,
):
    _DIM = "#6b7280"
    _n = len(universe)

    _end = correlation_end_slider.value
    _window = slice(_end - correlation_window_days, _end)
    _corr = np.corrcoef(universe_returns_arr[_window], rowvar=False)  # (n, n)
    _corr_residual = np.corrcoef(market_residual_returns[_window], rowvar=False)  # same window, market removed

    _random_order = np.random.default_rng(0).permutation(_n)
    _sector_rows = sectors_by_size(sector_indices(universe, sector_of))
    _sector_order, _block_starts, _block_ends = sector_blocks(_sector_rows)

    _fig, _axes = plt.subplots(1, 3, figsize=(22, 9))
    _fig.patch.set_facecolor("#ffffff")
    _panels = (
        (_axes[0], _corr, _random_order, "Arbitrary order"),
        (_axes[1], _corr, _sector_order, "Grouped by sector"),
        (_axes[2], _corr_residual, _sector_order, "Grouped by sector, market removed"),
    )
    for _ax, _matrix, _order, _name in _panels:
        _image = draw_correlation_matrix(_ax, _matrix, _order, universe, _name, highlight=symbols)

    # mark the sector blocks on the reordered matrices, and name them to the right of the last one
    for _ax in _axes[1:]:
        mark_sector_blocks(_ax, _block_ends)
    label_sector_blocks(_axes[2], _sector_rows, _block_starts, _block_ends, _n)

    # average correlation of pairs in the same sector against pairs in different sectors
    _same_sector = same_sector_mask(universe, sector_of)
    _within, _across = within_and_across(_corr, _same_sector)
    _within_residual, _across_residual = within_and_across(_corr_residual, _same_sector)

    _fig.subplots_adjust(left=0.03, right=0.91, bottom=0.2, top=0.9, wspace=0.08)
    _fig.colorbar(_image, cax=_fig.add_axes([0.37, 0.085, 0.2, 0.015]), orientation="horizontal", label="correlation")
    _end_date = str(universe_dates[_end - 1])[:10]
    title_figure(_fig, f"Correlation of daily returns over {correlation_window_days} days ending {_end_date}")
    _fig.text(
        0.0,
        0.005,
        "Average correlation of pairs in the same sector / in different sectors: "
        f"{_within:.2f} / {_across:.2f} before removing the market, {_within_residual:.2f} / {_across_residual:.2f} after.",
        color=_DIM,
        fontsize=8,
        ha="left",
    )
    mo.vstack([correlation_end_slider, _fig])
    return


@app.cell(hide_code=True)
def section_sector_layer(mo):
    mo.md(r"""
    ## Layer 2: sector

    Within each sector $S$, average the market residuals to get a sector factor
    $s_{S,t} = \tfrac{1}{\lvert S \rvert}\sum_{i \in S} e^{(1)}_{i,t}$ and project again, this time each market
    residual series onto its sector factor:

    $$\gamma_i = {\mathbf{cov}(e^{(1)}_i, s_{S(i)}) \over \mathbf{var}(s_{S(i)})} = {\sum_t e^{(1)}_{i,t}\, s_{S(i),t} \over \sum_t s_{S(i),t}^2}, \qquad e^{(2)}_{i,t} = e^{(1)}_{i,t} - \gamma_i s_{S(i),t} .$$

    What is left, $e^{(2)}$, is the idiosyncratic return. Each factor only gets the variance that the earlier ones
    left over, which is the Gram-Schmidt step. Putting the two layers together, every return is exactly

    $$r_t = \beta\, m_t + \Gamma s_t + e^{(2)}_t, \qquad \text{that is,} \qquad r_{i,t} = \beta_i\, m_t + \gamma_i\, s_{S(i),t} + e^{(2)}_{i,t},$$

    where $s_t \in \mathbf{R}^K$ stacks the $K$ sector factors and $\Gamma \in \mathbf{R}^{n \times K}$ has a single
    nonzero entry in each row, $\Gamma_{i,S(i)} = \gamma_i$. This is an identity that follows from the definitions of
    $e^{(1)}$ and $e^{(2)}$, not a modeling assumption.

    Taking the covariance of the three terms $\beta m$, $\Gamma s$ and $e^{(2)}$ gives a covariance for each plus cross
    terms. Let $\sigma_m^2 = \mathbf{var}(m)$ and $\Sigma^s = \mathbf{cov}(s) \in \mathbf{R}^{K \times K}$ be the
    covariance of the sector factors. Some cross terms vanish exactly: $m$ is uncorrelated with every sector factor
    and every $e^{(2)}_i$, and each $e^{(2)}_i$ is uncorrelated with its own sector's factor. Others do not:

    - the residuals $e^{(2)}_i$ of different stocks can be correlated, and within a sector they sum to exactly zero
      because $\sum_{i \in S} \gamma_i = \lvert S \rvert$,
    - a sector factor can be correlated with the residuals of stocks in other sectors.

    Dropping those terms gives the model covariance, with a market, a sector and an idiosyncratic component,

    $$\hat{\Sigma} = \underbrace{\sigma_m^2\, \beta\beta^T}_{\text{market}} + \underbrace{\Gamma \Sigma^s \Gamma^T}_{\text{sector}} + \underbrace{D}_{\text{idiosyncratic}}, \qquad D = \mathbf{diag}\left(\mathbf{var}(e^{(2)}_1), \ldots, \mathbf{var}(e^{(2)}_n)\right).$$

    The market term has rank one and the sector term has rank at most $K$, so $\hat{\Sigma}$ is low rank plus diagonal,
    the same form as the factor model of Johansson et al. (2023). It is an approximation to the covariance of $r_t$,
    not an identity, and it is full rank because $D \succ 0$. Counting parameters, $\beta$ and $\Gamma$ have $2n$ free
    entries, $\sigma_m^2$ and $\Sigma^s$ have $1 + K(K+1)/2$ and $D$ has $n$, far fewer than the $n(n+1)/2$ of an
    unrestricted covariance matrix.

    Caveats. The sectors are the GICS sectors from the S&P 500 constituents list. The betas in both layers
    are estimated on the full sample, so the residuals are in-sample. A predictor would need betas from past data
    only, for example from EWMAs. The smallest sectors have only four stocks, so their sector factors are noisy and
    contain a large share of each stock itself.
    """)
    return


@app.cell
def sector_layer(
    market_residual_returns,
    project_out_sector_factors,
    sector_indices,
    sector_of,
    universe,
):
    sector_rows = sector_indices(universe, sector_of)
    # sector factors are built from what the market layer left behind
    sector_factor_returns, sector_betas, sector_residual_returns = project_out_sector_factors(
        market_residual_returns, sector_rows
    )
    return (
        sector_betas,
        sector_factor_returns,
        sector_residual_returns,
        sector_rows,
    )


@app.cell
def plot_sector_fit(
    market_residual_returns,
    new_grid,
    np,
    plt,
    robust_ylim,
    second_moment_r_squared,
    sector_betas,
    sector_factor_returns,
    sector_residual_returns,
    sector_rows,
    style_plain_ax,
    title_axes,
    title_figure,
    universe,
):
    _INK = "#111827"
    _DIM = "#6b7280"
    _pct = lambda v, _: f"{v:.0%}"

    _sectors = [("Information Technology", "tab:blue"), ("Financials", "tab:green")]  # (sector, color) for each row
    _n_cols = 5

    _fig, _axes = new_grid(plt, len(_sectors), _n_cols, figsize=(14, 3.4 * len(_sectors)), sharex="row", sharey="row")
    _n_clipped = 0

    for _row, (_sector, _color) in enumerate(_sectors):
        _stocks = sector_rows[_sector][:_n_cols]
        _factor = sector_factor_returns[_sector]
        _x_line = np.array([_factor.min(), _factor.max()])

        # y-limits cover the central 99.5% of points, so a handful of extreme days do not squash the rest
        _ylim, _row_clipped = robust_ylim(market_residual_returns[:, _stocks])
        _n_clipped += _row_clipped

        for _col, _i in enumerate(_stocks):
            _ax = _axes[_row, _col]
            _market_residual = market_residual_returns[:, _i]
            # share of the market residual's second moment that the sector factor explains
            _r_squared = second_moment_r_squared(sector_residual_returns[:, _i], _market_residual)

            # the stock's market residual against its sector factor, with the fitted line e = gamma * s
            _ax.scatter(_factor, _market_residual, s=6, color=_color, alpha=0.4, linewidths=0, zorder=2)
            _ax.plot(_x_line, sector_betas[_i] * _x_line, color=_INK, linewidth=1.2, zorder=3)
            title_axes(
                _ax,
                rf"{universe[_i]}   $\gamma$ = {sector_betas[_i]:.2f},  $R^2$ = {_r_squared:.0%}",
                fontsize=9,
                pad=6,
            )
            style_plain_ax(_ax, x_fmt=_pct, y_fmt=_pct)
            _ax.set_xlabel("sector factor", color=_DIM, fontsize=8)

        for _col in range(len(_stocks), _n_cols):
            _axes[_row, _col].set_visible(False)
        _axes[_row, 0].set_ylim(-_ylim, _ylim)
        _axes[_row, 0].set_ylabel(f"{_sector}\nmarket residual", color=_DIM, fontsize=8)

    title_figure(_fig, r"Market residual vs sector factor, with fitted slope $\gamma_i$", fontsize=10.5)
    _fig.text(
        0.0,
        0.005,
        f"{_n_clipped} extreme points are outside the y-limits. "
        "The sector factor is the average of the sector's residuals, so it includes the stock itself.",
        color=_DIM,
        fontsize=8,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.gcf()
    return


@app.cell
def plot_sector_residual_series(
    new_grid,
    np,
    plt,
    robust_ylim,
    sector_factor_returns,
    sector_of,
    sector_residual_returns,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
    title_figure,
    universe,
    universe_dates,
):
    _INK = "#111827"
    _DIM = "#6b7280"

    # for each FAANG stock: what is left after the market and the sector, and the sector factor that was removed
    _panels = []
    for _ticker in symbols:
        _sector = sector_of[_ticker]
        _panels.append(
            (_ticker, _sector, sector_residual_returns[:, universe.index(_ticker)], sector_factor_returns[_sector])
        )
    _all_series = np.stack([_series for _, _, _residual, _factor in _panels for _series in (_residual, _factor)])

    # shared y-limits cover the central 99.5% of points, so a handful of extreme days do not squash the rest
    _ylim, _n_clipped = robust_ylim(_all_series)

    _fig, _axes = new_grid(plt, 2, 3, figsize=(12, 5), sharey=True)
    for _ax, (_ticker, _sector, _residual, _factor) in zip(_axes.flat, _panels):
        _ax.plot(universe_dates, _residual, color=symbol_to_color[_ticker], linewidth=0.9, alpha=0.75, zorder=2)
        _ax.plot(universe_dates, _factor, color=_INK, linewidth=1.0, alpha=0.9, zorder=3)
        style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4, y_fmt=lambda v, _: f"{v:.0%}")
        title_axes(_ax, f"{_ticker}  ·  {_sector}", fontsize=9, pad=5)
        _ax.set_xlabel(
            rf"$e^{{(2)}}$ std {_residual.std():.1%}   ·   sector factor std {_factor.std():.1%}", color=_DIM, fontsize=8
        )
    _axes.flat[0].set_ylim(-_ylim, _ylim)
    _axes.flat[-1].set_visible(False)

    title_figure(_fig, r"FAANG idiosyncratic residuals $e^{(2)}$ and their sector factors")
    _fig.text(
        0.0,
        0.005,
        rf"Colored line: $e^{{(2)}}$. Dark line: the sector factor. {_n_clipped} extreme points are outside the y-limits.",
        color=_DIM,
        fontsize=8,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.gcf()
    return


@app.cell(hide_code=True)
def section_correlation_after_sector(mo):
    mo.md(r"""
    ### Correlations after the sector layer

    Does removing the sector factors take out the within-sector blocks? The left matrix below is the market residuals
    $e^{(1)}$ from before, and the right matrix is the residuals $e^{(2)}$ once the sector factors are removed too,
    for the same window (set by the slider) and with the same stock ordering and color scale. Within a sector the
    residuals $e^{(2)}$ sum to exactly zero, so they cannot be uncorrelated: for a sector of $\lvert S \rvert$ stocks
    with equal variances, their average correlation is $-1/(\lvert S \rvert - 1)$. The diagonal blocks should
    therefore come out slightly blue rather than white.
    """)
    return


@app.cell
def plot_correlation_after_sector(
    correlation_end_slider,
    correlation_window_days,
    draw_correlation_matrix,
    label_sector_blocks,
    mark_sector_blocks,
    market_residual_returns,
    mo,
    np,
    plt,
    same_sector_mask,
    sector_blocks,
    sector_indices,
    sector_of,
    sector_residual_returns,
    sectors_by_size,
    symbols,
    title_figure,
    universe,
    universe_dates,
    within_and_across,
):
    _DIM = "#6b7280"
    _n = len(universe)

    _end = correlation_end_slider.value
    _window = slice(_end - correlation_window_days, _end)
    _corr_market_removed = np.corrcoef(market_residual_returns[_window], rowvar=False)
    _corr_sector_removed = np.corrcoef(sector_residual_returns[_window], rowvar=False)

    _sector_rows = sectors_by_size(sector_indices(universe, sector_of))
    _order, _block_starts, _block_ends = sector_blocks(_sector_rows)

    _fig, _axes = plt.subplots(1, 2, figsize=(15, 9))
    _fig.patch.set_facecolor("#ffffff")
    for _ax, _matrix, _name in (
        (_axes[0], _corr_market_removed, "Market removed"),
        (_axes[1], _corr_sector_removed, "Market and sector removed"),
    ):
        _image = draw_correlation_matrix(_ax, _matrix, _order, universe, _name, highlight=symbols)
        mark_sector_blocks(_ax, _block_ends)
    label_sector_blocks(_axes[1], _sector_rows, _block_starts, _block_ends, _n)

    _same_sector = same_sector_mask(universe, sector_of)
    _within_market, _across_market = within_and_across(_corr_market_removed, _same_sector)
    _within_sector, _across_sector = within_and_across(_corr_sector_removed, _same_sector)

    _fig.subplots_adjust(left=0.04, right=0.84, bottom=0.2, top=0.9, wspace=0.08)
    _fig.colorbar(_image, cax=_fig.add_axes([0.3, 0.085, 0.25, 0.015]), orientation="horizontal", label="correlation")
    _end_date = str(universe_dates[_end - 1])[:10]
    title_figure(_fig, f"Residual correlations over {correlation_window_days} days ending {_end_date}")
    _fig.text(
        0.0,
        0.005,
        "Average correlation of pairs in the same sector / in different sectors: "
        f"{_within_market:.2f} / {_across_market:.2f} with the market removed, "
        f"{_within_sector:.2f} / {_across_sector:.2f} with the sector removed too.",
        color=_DIM,
        fontsize=8,
        ha="left",
    )
    mo.vstack([correlation_end_slider, _fig])
    return


@app.cell
def show_variance_decomposition(
    market_residual_returns,
    mo,
    np,
    sector_residual_returns,
    sector_rows,
    universe_returns_arr,
    variance_shares,
):
    _market_share, _sector_share, _idio_share = variance_shares(
        universe_returns_arr, market_residual_returns, sector_residual_returns
    )
    np.testing.assert_allclose(_market_share + _sector_share + _idio_share, 1.0, atol=1e-10)

    _rows = ["| Sector | Stocks | Market | Sector | Idiosyncratic |", "|---|---|---|---|---|"]
    for _sector, _idx in sector_rows.items():
        _rows.append(
            f"| {_sector} | {len(_idx)} | {_market_share[_idx].mean():.0%} | {_sector_share[_idx].mean():.0%} "
            f"| {_idio_share[_idx].mean():.0%} |"
        )
    _rows.append(
        f"| **All** | {len(_market_share)} | **{_market_share.mean():.0%}** | **{_sector_share.mean():.0%}** "
        f"| **{_idio_share.mean():.0%}** |"
    )
    mo.md("Share of each stock's variance explained by each layer (average within sector):\n\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def section_pca_covariance(mo):
    mo.md(r"""
    # PCA Covariance (Diagonal plus Low Rank)

    The hierarchical model chose its factors from economic labels, the market and then the sectors. PCA chooses them
    from the data. Let $R \in \mathbf{R}^{T \times n}$ be the matrix of returns and
    $S = \tfrac{1}{T} R^T R \in \mathbf{R}^{n \times n}$ its second-moment matrix (no means subtracted, as in the rest
    of this notebook). Its eigendecomposition is

    $$S = \sum_{i=1}^n \lambda_i\, q_i q_i^T, \qquad \lambda_1 \ge \cdots \ge \lambda_n \ge 0,$$

    and keeping only the $k$ largest eigenvalues gives the best rank-$k$ approximation of $S$ in the Frobenius norm,

    $$S_k = \sum_{i=1}^k \lambda_i\, q_i q_i^T = F F^T, \qquad F = \begin{bmatrix} \sqrt{\lambda_1}\, q_1 & \cdots & \sqrt{\lambda_k}\, q_k \end{bmatrix} \in \mathbf{R}^{n \times k}.$$

    This is not yet a covariance estimate. It has rank $k < n$, so it is singular and the Gaussian log-likelihood of a
    return is not defined. The discarded eigenvalues are the fix. They are idiosyncratic variance, so we add back the
    diagonal of what was thrown away,

    $$\hat{\Sigma}_k = F F^T + D, \qquad D = \mathbf{diag}(S - S_k), \qquad D_{ii} = \sum_{j > k} \lambda_j\, q_{j,i}^2 \;\ge\; 0 .$$

    Two things follow. The diagonal of $\hat{\Sigma}_k$ equals the diagonal of $S$ exactly, so every stock keeps its
    sample variance and only the correlations are approximated. And $D \succ 0$ whenever each stock loads on at least
    one discarded component, so $\hat{\Sigma}_k$ is positive definite.

    This is the same low rank plus diagonal form as the hierarchical model, with two differences. The columns of $F$
    are dense and orthogonal instead of being tied to a market and to sectors, and there is no fixed economic order:
    the first column $q_1$ is simply the direction of largest variance, which for stocks is close to equal weights, that
    is, the market. The number of components $k$ is a choice. The plot below shows how much of the total variance
    $\mathbf{tr}(S) = \sum_i \lambda_i$ each component carries. Its default is the number of factors in the hierarchical
    model, the market plus one per sector. As before, everything is estimated on the full sample, so it is in-sample.

    The second plot compares the approximation with the sample matrix as correlations, ordered by sector like the
    heatmaps above. $\hat{\Sigma}_k$ and $S$ have the same diagonal, so dividing both by the same standard deviations
    loses nothing. The third panel is their difference, which is what the discarded components still carry.

    A different way to choose $k$ is eigenvalue cleaning, from random matrix theory. For the correlation matrix of $n$
    independent stocks observed over $T$ days, the sample eigenvalues are not all equal to one. They fall in the
    Marchenko-Pastur band $\left[(1-\sqrt{q})^2,\, (1+\sqrt{q})^2\right]$ with $q = n/T$. Eigenvalues above the upper
    edge are distinguishable from noise and are kept, and the rest are replaced by their average. This tests the
    eigenvalues against a noise benchmark instead of using a threshold on the variance explained, and it works with
    correlations, so the stocks are standardized first. The plain edge is conservative for stock returns: the market
    eigenvalue takes such a large share of the total variance that the remaining noise has less variance than the
    benchmark assumes, so it keeps fewer components than the default $k$ above. The usual fix is to fit the noise level
    to the bulk of the spectrum.
    """)
    return


@app.cell
def pca_functions(np):
    def pca_factor_model(returns: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Diagonal plus low rank covariance from the top k eigenvalues of the second-moment matrix: F F^T + diag(d).

        returns: (T, n) returns, one column per stock.
        k: number of components kept.

        Returns (F, d, eigenvalues): the (n, k) loadings sqrt(lambda_i) q_i, the (n,) diagonal d, and all n eigenvalues
        of the second-moment matrix in descending order. d is chosen so that diag(F F^T + diag(d)) is the diagonal of
        the second-moment matrix.
        """
        second_moment = returns.T @ returns / len(returns)
        eigenvalues, eigenvectors = np.linalg.eigh(second_moment)  # ascending
        eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

        loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
        diagonal = np.diag(second_moment) - (loadings**2).sum(axis=1)
        return loadings, diagonal, eigenvalues

    return (pca_factor_model,)


@app.cell
def check_pca_factor_model(np, pca_factor_model, universe_returns_arr):
    _second_moment = universe_returns_arr.T @ universe_returns_arr / len(universe_returns_arr)
    _n = _second_moment.shape[0]

    # keeping all eigenvalues recovers the second-moment matrix, and leaves nothing for the diagonal
    _F, _d, _ = pca_factor_model(universe_returns_arr, _n)
    np.testing.assert_allclose(_F @ _F.T, _second_moment, atol=1e-12)
    np.testing.assert_allclose(_d, 0.0, atol=1e-12)

    # with fewer components every sample variance is kept, and the estimate is positive definite
    for _k in [1, 5, 12]:
        _F, _d, _ = pca_factor_model(universe_returns_arr, _k)
        _cov = _F @ _F.T + np.diag(_d)
        np.testing.assert_allclose(np.diag(_cov), np.diag(_second_moment), rtol=1e-10)
        assert (_d > 0).all() and np.linalg.eigvalsh(_cov).min() > 0
    return


@app.cell(hide_code=True)
def pca_components_selector(mo, sector_of):
    pca_components_slider = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=len(set(sector_of.values())) + 1,  # the market plus one factor per sector
        label="Components k",
        show_value=True,
        full_width=True,
    )
    return (pca_components_slider,)


@app.cell
def plot_pca_spectrum(
    mo,
    new_grid,
    np,
    pca_components_slider,
    pca_factor_model,
    plt,
    style_plain_ax,
    title_axes,
    title_figure,
    universe_returns_arr,
):
    _DIM = "#6b7280"
    _k = pca_components_slider.value

    _, _, _eigenvalues = pca_factor_model(universe_returns_arr, _k)
    _share = _eigenvalues / _eigenvalues.sum()  # each component's share of the total variance tr(S)
    _cumulative = np.cumsum(_share)
    _components = np.arange(1, len(_share) + 1)
    _kept = _components <= _k

    _fig, (_ax_share, _ax_cumulative) = new_grid(plt, 1, 2, figsize=(12, 4.2))

    _ax_share.bar(_components[_kept], _share[_kept], color="tab:blue", alpha=0.9, zorder=2, label="kept")
    _ax_share.bar(_components[~_kept], _share[~_kept], color="#9ca3af", alpha=0.9, zorder=2, label="discarded")
    _ax_share.set_yscale("log")
    _ax_share.set_xlabel("component", color=_DIM, fontsize=8)
    _ax_share.legend(frameon=False, fontsize=8, labelcolor=_DIM)
    style_plain_ax(_ax_share, labelsize=8, y_fmt=lambda v, _: f"{v:.1%}")
    title_axes(_ax_share, "Share of total variance per component", fontsize=9, pad=6)

    _ax_cumulative.plot(_components, _cumulative, color="tab:blue", linewidth=1.4, zorder=2)
    _ax_cumulative.scatter([_k], [_cumulative[_k - 1]], color="tab:blue", s=30, zorder=3)
    _ax_cumulative.axvline(_k, color=_DIM, linewidth=0.8, linestyle=":", zorder=1)
    _ax_cumulative.annotate(
        f"k = {_k}: {_cumulative[_k - 1]:.0%}",
        (_k, _cumulative[_k - 1]),
        textcoords="offset points",
        xytext=(8, -14),
        color=_DIM,
        fontsize=8.5,
    )
    _ax_cumulative.set_ylim(0, 1)
    _ax_cumulative.set_xlabel("number of components kept", color=_DIM, fontsize=8)
    style_plain_ax(_ax_cumulative, labelsize=8, y_fmt=lambda v, _: f"{v:.0%}")
    title_axes(_ax_cumulative, "Cumulative share of total variance", fontsize=9, pad=6)

    title_figure(_fig, f"Eigenvalues of the second-moment matrix of {len(_share)} stocks")
    _fig.tight_layout(rect=[0, 0, 1, 0.94])
    mo.vstack([pca_components_slider, _fig])
    return


@app.cell
def plot_pca_approximation(
    draw_correlation_matrix,
    label_sector_blocks,
    mark_sector_blocks,
    np,
    pca_components_slider,
    pca_factor_model,
    plt,
    sector_blocks,
    sector_indices,
    sector_of,
    sectors_by_size,
    symbols,
    title_figure,
    universe,
    universe_returns_arr,
):
    _DIM = "#6b7280"
    _k = pca_components_slider.value
    _n = len(universe)

    _F, _d, _eigenvalues = pca_factor_model(universe_returns_arr, _k)
    _second_moment = universe_returns_arr.T @ universe_returns_arr / len(universe_returns_arr)
    _approximation = _F @ _F.T + np.diag(_d)

    # the diagonals are identical, so the same rescaling to correlations loses nothing for either matrix
    _inverse_std = 1 / np.sqrt(np.diag(_second_moment))
    _corr = _second_moment * np.outer(_inverse_std, _inverse_std)
    _corr_approximation = _approximation * np.outer(_inverse_std, _inverse_std)
    _error = _corr - _corr_approximation
    _mean_abs_error = np.abs(_error[~np.eye(_n, dtype=bool)]).mean()
    _error_scale = max(np.abs(_error).max(), 1e-3)

    _sector_rows = sectors_by_size(sector_indices(universe, sector_of))
    _order, _block_starts, _block_ends = sector_blocks(_sector_rows)

    _fig, _axes = plt.subplots(1, 3, figsize=(22, 9))
    _fig.patch.set_facecolor("#ffffff")
    _images = []
    for _ax, _matrix, _name, _vmax in (
        (_axes[0], _corr, "Sample (full window)", 1.0),
        (_axes[1], _corr_approximation, f"{_k} components plus a diagonal", 1.0),
        (_axes[2], _error, "Difference (sample minus approximation)", _error_scale),
    ):
        _images.append(draw_correlation_matrix(_ax, _matrix, _order, universe, _name, highlight=symbols, vmax=_vmax))
        mark_sector_blocks(_ax, _block_ends)
    label_sector_blocks(_axes[2], _sector_rows, _block_starts, _block_ends, _n)

    _fig.subplots_adjust(left=0.03, right=0.90, bottom=0.2, top=0.9, wspace=0.08)
    _fig.colorbar(_images[0], cax=_fig.add_axes([0.1, 0.085, 0.3, 0.015]), orientation="horizontal", label="correlation")
    _fig.colorbar(_images[2], cax=_fig.add_axes([0.66, 0.085, 0.2, 0.015]), orientation="horizontal", label="difference")
    title_figure(_fig, f"Correlation matrix of {_n} stocks: sample against {_k} principal components plus a diagonal")
    _fig.text(
        0.0,
        0.005,
        f"The {_k} components carry {_eigenvalues[:_k].sum() / _eigenvalues.sum():.0%} of the total variance. "
        f"Mean absolute off-diagonal correlation error: {_mean_abs_error:.3f}. The diagonals are identical by construction.",
        color=_DIM,
        fontsize=8,
        ha="left",
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def section_cm_iewma(mo):
    mo.md(r"""
    # Combining Predictors (CM-IEWMA)

    No single pair of half-lives $(H^{\text{vol}}, H^{\text{cor}})$ is best in every market regime. Fast
    predictors react quickly when volatility jumps, and slow predictors are less noisy when markets are calm.
    Given $K$ IEWMA predictors $\hat{\Sigma}^{(k)}_t$, Johansson et al. (2023) combine them with weights
    chosen from how well each has predicted the last $N$ days.

    ## Log-likelihood in terms of the precision Cholesky factor

    Write the precision matrix of each predictor as $(\hat{\Sigma}^{(k)}_t)^{-1} = L^{(k)}_t (L^{(k)}_t)^T$,
    with $L^{(k)}_t$ lower triangular with positive diagonal. Since
    $\det \hat{\Sigma}^{-1} = \prod_i L_{ii}^2$ and $r^T \hat{\Sigma}^{-1} r = \|L^T r\|_2^2$, the
    Gaussian log-likelihood of a return $r_t$ is, up to a constant,

    $$\ell_t(L) = \sum_{i=1}^n \log L_{ii} - \tfrac{1}{2} \|L^T r_t\|_2^2 .$$

    The combined predictor mixes the **factors** (not the covariances) with weights $w \in \mathbf{R}^K$ in the simplex,

    $$L_t(w) = \sum_{k=1}^K w_k L^{(k)}_t, \qquad w \succeq 0, \quad \mathbf{1}^T w = 1, \qquad
    \hat{\Sigma}_t = \left(L_t(w) L_t(w)^T\right)^{-1}.$$

    A mixture of lower triangular matrices with positive diagonals is again one, so $\hat{\Sigma}_t$ is always a
    valid covariance matrix.

    ## The weight problem

    The prediction $L^{(k)}_s$ for day $s$ only uses returns through day $s-1$. To choose the weights for day $t$,
    we score the combined predictor on the last $N$ days, whose returns are now known:

    $$\begin{aligned}
    \underset{w}{\text{maximize}} \quad & \sum_{s=t-N}^{t-1} \left[ \sum_{i=1}^n \log L_{s,ii}(w) - \tfrac{1}{2} \|L_s(w)^T r_s\|_2^2 \right] \\
    \text{subject to} \quad & w \succeq 0, \quad \mathbf{1}^T w = 1 .
    \end{aligned}$$

    The solution is then applied to the factors for day $t$ itself. The only variable is $w \in \mathbf{R}^K$, and $\succeq$ denotes componentwise inequality.

    ## Why this is a convex problem

    We are maximizing a sum of concave functions over a polyhedron. Term by term, using the composition
    rules for convex functions:

    1. **Everything inside is affine in $w$.** $L_{s,ii}(w) = \sum_k w_k L^{(k)}_{s,ii}$ and
       $L_s(w)^T r_s = \sum_k w_k \, L^{(k)T}_s r_s$ are both linear in $w$.
    2. **The log terms are concave.** $\log$ is concave and nondecreasing, and it is composed with an affine
       function of $w$, so $\log L_{s,ii}(w)$ is concave. Its argument is positive on the feasible set because
       each $L^{(k)}_{s,ii} > 0$ and $w$ is a nonnegative vector summing to one.
    3. **The quadratic terms are convex, so their negatives are concave.** $\|\cdot\|_2$ is convex and
       nonnegative, and $u \mapsto u^2$ is convex and nondecreasing on $u \ge 0$. Composing the two with the
       affine map $w \mapsto L_s(w)^T r_s$ gives a convex function, so $-\tfrac{1}{2}\|L_s(w)^T r_s\|_2^2$ is concave.
    4. **Sums preserve concavity.** The objective is a sum of concave functions of $w$, and the constraints
       $w \succeq 0$, $\mathbf{1}^T w = 1$ are polyhedral. Maximizing a concave function over a convex set is a
       convex problem, so any local maximum is global and a solver finds it reliably.

    ## Why mix Cholesky factors

    Mixing the covariances $\sum_k w_k \hat{\Sigma}^{(k)}$ does **not** work: the log-likelihood
    $-\tfrac{1}{2}\log\det\hat{\Sigma} - \tfrac{1}{2} r^T \hat{\Sigma}^{-1} r$ has a convex first term and a
    concave second term, so it is not concave in $\hat{\Sigma}$.

    Mixing the precision matrices $S = \hat{\Sigma}^{-1}$ would also give a concave objective,
    $\tfrac{1}{2}\log\det S - \tfrac{1}{2} r^T S r$. The paper does not say why it prefers Cholesky factors.
    A practical advantage is that $\log\det$ becomes a sum of $\log$s of linear functions of $w$, so the problem
    needs only logs and a sum of squares, not a log-determinant.
    """)
    return


@app.cell
def prepare_universe_returns(
    daily_returns,
    returns_matrix,
    universe,
    universe_df,
):
    # the mixture experiments use every ticker in the universe, not just the five that are plotted above
    universe_dates, universe_returns_arr = returns_matrix(daily_returns(universe_df), universe)  # (T,), (T, n)
    return universe_dates, universe_returns_arr


@app.cell
def mixture_components(iterated_ewma, np, universe_returns_arr):
    import cvxpy as cp

    # Half-life pairs (h_vol, h_corr) in days — "Stocks" configuration from Johansson et al. 2023
    half_life_pairs = [(10, 21), (21, 63), (63, 125), (125, 250), (250, 500)]

    # Days of history used to choose the mixture weights at each time step
    look_back_days = 20

    def precision_cholesky_factors(
        cov_matrices: list[np.ndarray],  # K one-step-ahead predictions, each (N, N, T)
    ) -> list[np.ndarray]:
        """Compute L^(k) s.t. (Σ^(k))^{-1} = L^(k) (L^(k))^T for each component k."""
        result = []
        for cov_k in cov_matrices:
            cov_t = cov_k.transpose(2, 0, 1)  # (T, N, N)
            prec_t = np.linalg.inv(cov_t)  # (T, N, N)
            result.append(np.linalg.cholesky(prec_t))  # (T, N, N), lower triangular
        return result

    def mixture_covariance(L_all: list[np.ndarray], t: int, weights: np.ndarray) -> np.ndarray:
        """
        Covariance prediction for day t: (L L^T)^{-1} with L = sum_k weights[k] * L^(k)_t, an (N, N) matrix.

        L_all: the K precision Cholesky factors from precision_cholesky_factors, each (T, N, N).
        t: day to predict.
        weights: (K,) mixture weights.
        """
        L_t = sum(weights[k] * L_all[k][t] for k in range(len(L_all)))
        return np.linalg.inv(L_t @ L_t.T)

    cov_matrices = [iterated_ewma(universe_returns_arr, h_vol=_h_vol, h_corr=_h_corr) for _h_vol, _h_corr in half_life_pairs]
    return (
        cov_matrices,
        cp,
        half_life_pairs,
        look_back_days,
        mixture_covariance,
        precision_cholesky_factors,
    )


@app.cell
def precision_mixture_solver_naive(
    cov_matrices,
    cp,
    look_back_days,
    mixture_covariance,
    np,
    precision_cholesky_factors,
    universe_returns_arr,
):
    # Direct formulation: build the full N x N cvxpy expression for every day in the window, every solve.
    import time as _time

    def _solve_optimal_weights(
        cholesky_factors: list[np.ndarray],  # K arrays, each (window, N, N)
        returns: np.ndarray,  # (window, N)
    ) -> np.ndarray | None:
        """
        Maximize the average log-likelihood of the combined predictor over the lookback window:

            max_w  sum_t [ sum_i log(L_t_ii) - 0.5 * ||L_t^T r_t||^2 ]
            s.t.   L_t = sum_k w_k * L_t^(k),  w >= 0,  sum(w) = 1

        The objective is concave in w because log(diag(L_t)) is concave in w (affine composition).

        Returns None if the solver fails, so callers can see failures instead of silently
        getting uniform weights.
        """
        K = len(cholesky_factors)
        window, _ = returns.shape
        w = cp.Variable(K, nonneg=True)

        obj = 0
        for t in range(window):
            # (N, K): column k holds diagonal of L^(k)_t; diags @ w gives diag(L_combined_t)
            diags = np.column_stack([np.diag(cholesky_factors[k][t]) for k in range(K)])
            L_t = sum(w[k] * cholesky_factors[k][t] for k in range(K))
            obj += cp.sum(cp.log(diags @ w)) - 0.5 * cp.sum_squares(L_t.T @ returns[t])

        try:
            cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1]).solve()
        except cp.SolverError:
            return None

        return w.value

    def solve_precision_mixture_naive(
        returns: np.ndarray,  # (T, N)
        cov_matrices: list[np.ndarray],  # K arrays, each (N, N, T)
        look_back: int = 10,
    ) -> np.ndarray:
        """
        CM-IEWMA (Johansson et al. 2023): for each time t, find mixture weights that maximize
        log-likelihood over [t-look_back, t), then apply them to predict the covariance at t.

        returns: (T, N) returns, one row per day.
        cov_matrices: the K one-step-ahead predictions to mix, each (N, N, T).
        look_back: number of past days used to score the mixture.

        Returns covariance predictions of shape (N, N, T - look_back - 1), aligned with
        dates[look_back + 1:]. Days where the weight solve fails are NaN.
        """
        _, _, T = cov_matrices[0].shape
        n_days = T - look_back - 1

        L_all = precision_cholesky_factors(cov_matrices)

        cov_preds = []
        n_failed = 0
        start = _time.perf_counter()
        for i, t in enumerate(range(look_back + 1, T)):  # index 0 has no history (see _one_step_ahead)
            if i % 100 == 0:
                print(f"naive: day {i}/{n_days} ({_time.perf_counter() - start:.0f}s elapsed)")

            w_opt = _solve_optimal_weights([Lk[t - look_back : t] for Lk in L_all], returns[t - look_back : t])
            if w_opt is None:
                n_failed += 1
                print(f"naive: weight solve failed on day {i}")
                cov_preds.append(np.full(cov_matrices[0].shape[:2], np.nan))
                continue
            cov_preds.append(mixture_covariance(L_all, t, w_opt))

        print(f"naive: done in {_time.perf_counter() - start:.0f}s, {n_failed} of {n_days} weight solves failed (NaN)")

        return np.stack(cov_preds, axis=-1)  # (N, N, T - look_back - 1)

    cov_predicted_naive = solve_precision_mixture_naive(universe_returns_arr, cov_matrices, look_back=look_back_days)
    return (cov_predicted_naive,)


@app.cell
def precision_mixture_solver_kxk(
    cov_matrices,
    cp,
    look_back_days,
    mixture_covariance,
    np,
    precision_cholesky_factors,
    universe_returns_arr,
):
    # K x K formulation: the data enters only through A and P, and the problem is built once (DPP).

    def _weight_problem(n_stocks: int, K: int, look_back: int):
        """
        Build the weight problem once, with the data as cvxpy Parameters (DPP), so each day only
        swaps in new parameter values instead of re-canonicalizing the problem.

        With L_s(w) = sum_k w_k L_s^(k), both terms of the objective are simple in w:

            sum_i log L_s,ii(w)  =  sum(log(A @ w)),   A stacks diag(L_s^(k)) for the window
                                                       (shape n_stocks * look_back x K)
            sum_s ||L_s(w)^T r_s||^2  =  ||sum_k w_k B_s[:, k]||^2 summed over s  =  w^T P w,
                                                       P = sum_s B_s^T B_s  (shape K x K),
                                                       B_s[:, k] = L_s^(k)T r_s

        so the data enters only through A and the K x K matrix P (passed as a square root P_half).
        """
        w = cp.Variable(K, nonneg=True)
        A = cp.Parameter((n_stocks * look_back, K))
        P_half = cp.Parameter((K, K))  # P = P_half @ P_half.T
        objective = cp.sum(cp.log(A @ w)) - 0.5 * cp.sum_squares(P_half.T @ w)
        problem = cp.Problem(cp.Maximize(objective), [cp.sum(w) == 1])
        return problem, w, A, P_half

    def _psd_square_root(P: np.ndarray) -> np.ndarray:
        """S with S @ S.T = P. Uses eigh instead of Cholesky so a singular P does not fail."""
        eigvals, eigvecs = np.linalg.eigh(P)
        return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))

    def _per_day_weight_data(L_all: list[np.ndarray], returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-day inputs of the weight problem, for all days at once: diags (T, N, K) and BtB (T, K, K)."""
        diags = np.stack([np.diagonal(L_k, axis1=1, axis2=2) for L_k in L_all], axis=-1)  # (T, N, K)
        B = np.stack([np.einsum("tji,tj->ti", L_k, returns) for L_k in L_all], axis=-1)  # (T, N, K), L_k^T r_t
        return diags, np.einsum("tnk,tnl->tkl", B, B)

    def _solve_window_weights(problem, w, A, P_half, diags, BtB, window: slice) -> np.ndarray | None:
        """Put the window's data on the problem's parameters and solve it. Returns None if the solver fails."""
        K = diags.shape[-1]
        A.value = diags[window].reshape(-1, K)
        P_half.value = _psd_square_root(BtB[window].sum(axis=0))
        try:
            problem.solve()
        except cp.SolverError:
            return None
        return w.value

    def solve_precision_mixture_kxk(
        returns: np.ndarray,  # (T, N)
        cov_matrices: list[np.ndarray],  # K arrays, each (N, N, T)
        look_back: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Same predictor as solve_precision_mixture_naive, but with the K x K reformulation of the weight problem.

        returns: (T, N) returns, one row per day.
        cov_matrices: the K one-step-ahead predictions to mix, each (N, N, T).
        look_back: number of past days used to score the mixture.

        Returns (covariances, weights) with shapes (N, N, T - look_back - 1) and (K, T - look_back - 1),
        both aligned with dates[look_back + 1:]. Days where the weight solve fails are NaN.
        """
        K = len(cov_matrices)
        n_stocks, _, T = cov_matrices[0].shape

        L_all = precision_cholesky_factors(cov_matrices)  # K arrays, each (T, N, N)

        diags, BtB = _per_day_weight_data(L_all, returns)  # computed once for all days
        problem, w, A, P_half = _weight_problem(n_stocks, K, look_back)

        cov_preds = []
        weights = []
        n_failed = 0
        for t in range(look_back + 1, T):  # index 0 has no history (see _one_step_ahead)
            w_opt = _solve_window_weights(problem, w, A, P_half, diags, BtB, slice(t - look_back, t))
            if w_opt is None:
                n_failed += 1
                cov_preds.append(np.full((n_stocks, n_stocks), np.nan))
                weights.append(np.full(K, np.nan))
                continue

            cov_preds.append(mixture_covariance(L_all, t, w_opt))
            weights.append(w_opt.copy())

        if n_failed:
            print(f"Weight solve failed on {n_failed} of {T - look_back - 1} days (predictions are NaN)")

        return np.stack(cov_preds, axis=-1), np.stack(weights, axis=-1)  # (N, N, T'), (K, T'), T' = T - look_back - 1

    cov_predicted_kxk, weights_kxk = solve_precision_mixture_kxk(universe_returns_arr, cov_matrices, look_back=look_back_days)
    return cov_predicted_kxk, weights_kxk


@app.cell
def check_kxk_matches_naive(cov_predicted_kxk, cov_predicted_naive, np):
    # both formulations solve the same problem, so predictions agree up to solver tolerance;
    # a failed solve is NaN, so compare only the days that both formulations solved
    _solved = ~np.isnan(cov_predicted_naive).any(axis=(0, 1)) & ~np.isnan(cov_predicted_kxk).any(axis=(0, 1))
    np.testing.assert_allclose(
        cov_predicted_kxk[..., _solved], cov_predicted_naive[..., _solved], rtol=1e-3, atol=1e-6
    )
    print(f"Formulations agree on {_solved.sum()} of {len(_solved)} days (the rest failed in at least one solve)")
    return


@app.cell(hide_code=True)
def weights_smoothing_selector(mo):
    weights_smoothing_radio = mo.ui.radio(
        options={"None (daily)": 1, "21 days": 21, "63 days": 63, "126 days": 126},
        value="63 days",
        label="Trailing average",
        inline=True,
    )
    return (weights_smoothing_radio,)


@app.cell
def weight_smoothing_functions(np):
    def normalize_weights(w: np.ndarray) -> np.ndarray:
        """
        Clip the tiny negatives that solver tolerance leaves, and rescale each column to sum to one.

        w: (K, T) mixture weights, one column per day.
        """
        w = np.clip(w, 0.0, None)
        return w / w.sum(axis=0, keepdims=True)

    def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
        """
        Mean of each row over the last `window` columns (fewer at the start).

        x: (rows, columns) array, for example the mixture weights over time.
        window: number of columns averaged.

        Averages of simplex points stay on the simplex.
        """
        csum = np.cumsum(np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1), axis=1)
        n_seen = np.arange(1, x.shape[1] + 1)
        start = np.maximum(n_seen - window, 0)
        return (csum[:, 1:] - csum[:, start]) / (n_seen - start)

    return normalize_weights, trailing_mean


@app.cell
def plot_mixture_weights(
    half_life_pairs,
    mo,
    normalize_weights,
    np,
    plt,
    style_ax,
    title_axes,
    trailing_mean,
    universe,
    universe_dates,
    weights_kxk,
    weights_smoothing_radio,
):
    _INK = "#111827"
    _DIM = "#6b7280"
    _k = weights_smoothing_radio.value

    # weights are aligned with the last T - look_back - 1 dates
    _dates = universe_dates[-weights_kxk.shape[1] :]

    _w = trailing_mean(normalize_weights(weights_kxk), _k)

    _colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(half_life_pairs)))  # fast (dark) to slow (light)
    _labels = [f"{_h_vol}/{_h_cor}" for _h_vol, _h_cor in half_life_pairs]

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _ax.stackplot(_dates, _w, colors=_colors, labels=_labels, linewidth=0, alpha=0.95, zorder=2)
    style_ax(_ax, fig=_fig, date_interval=6, y_fmt=lambda v, _: f"{v:.0%}")
    _ax.set_ylim(0, 1)
    _ax.margins(x=0)
    _legend = _ax.legend(
        title=r"$H^{vol}$ / $H^{cor}$ (days)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        labelcolor=_INK,
        fontsize=8.5,
        title_fontsize=8.5,
    )
    _legend.get_title().set_color(_DIM)
    title_axes(
        _ax,
        f"{len(universe)} Stocks  ·  CM-IEWMA Component Weights" + ("" if _k == 1 else f"  ({_k}d trailing average)"),
    )
    _fig.tight_layout()
    mo.vstack([weights_smoothing_radio, _fig])
    return


if __name__ == "__main__":
    app.run()

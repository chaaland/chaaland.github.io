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

    def draw_correlation_matrix(ax, corr: np.ndarray, order: np.ndarray, labels: list[str], name: str):
        """Heatmap of corr with rows and columns taken in `order`, on a fixed scale from -1 to 1."""
        image = ax.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
        ordered_labels = [labels[i] for i in order]
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
def load_stock_data(Path, cs, end_date, pl, start_date, yf):
    data_file = Path("stocks.parquet")

    # The first five are plotted throughout the notebook; the rest widen the universe for the mixture experiments
    symbols = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
    colors = ["tab:blue", "tab:gray", "tab:purple", "tab:red", "tab:green"]
    symbol_to_color = dict(zip(symbols, colors))
    extra_symbols = [
        "MSFT", "NVDA", "TSLA", "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG",
        "MA", "HD", "CVX", "MRK", "LLY", "ABBV", "KO", "PEP", "BAC", "COST",
        "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "T",
        "VZ", "DIS", "NKE", "MCD", "PFE", "TMO", "ABT", "WFC", "CAT", "IBM",
        "HON", "AMGN", "GS", "LOW", "UPS",
    ]  # fmt: skip
    universe = symbols + extra_symbols

    # Only download tickers that are not already in the cache, then write everything back
    cached_df = pl.read_parquet(data_file) if data_file.exists() else None
    cached_tickers = set() if cached_df is None else set(cached_df["Ticker"].unique())
    missing_symbols = [symbol for symbol in universe if symbol not in cached_tickers]

    if missing_symbols:
        frames = [] if cached_df is None else [cached_df]

        for symbol in missing_symbols:
            history = yf.Ticker(symbol).history(start=start_date, end=end_date)
            assert len(history) > 0, f"No data returned for {symbol}"
            frames.append(pl.from_pandas(history.reset_index()).with_columns(pl.lit(symbol).alias("Ticker")))

        cached_df = pl.concat(frames, how="diagonal_relaxed").sort("Ticker", "Date")
        cached_df.write_parquet(data_file)

    universe_df = cached_df.filter(pl.col("Ticker").is_in(universe))
    universe_df = universe_df.select(cs.by_name("Ticker", "Date"), ~cs.by_name("Ticker", "Date"))

    # the plots below only use the five tickers in `symbols`
    df = universe_df.filter(pl.col("Ticker").is_in(symbols))
    df
    return df, symbol_to_color, symbols, universe, universe_df


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
def compute_returns(df, pl):
    returns_df = df.select("Ticker", "Date", Return=pl.col("Close").pct_change().over("Ticker")).filter(
        pl.col("Return").is_not_null()
    )

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
        # arr.shape = (stocks, time)
        N, T = arr.shape
        covariances = []

        for t in range(T):
            samples = arr[:, max(0, t - window + 1) : t + 1]  # N x t
            r_mean = np.mean(samples, axis=1)  # (N,)
            if demean:
                samples = samples - r_mean[:, None]

            n_stocks, n_samples = samples.shape
            cov_t = 1 / n_samples * samples @ samples.T  # N x N
            covariances.append(cov_t)

        return np.stack(covariances, axis=-1)

    def rolling_window_covariance_estimate(arr: np.ndarray, window: int) -> np.ndarray:
        N, T = arr.shape  # arr.shape = (stocks, time)

        r_last = np.zeros((N,))
        r_t = arr[:, 0]
        cov_t = np.outer(r_t, r_t)
        covariances = [cov_t]

        for t in range(1, T):
            if t >= window:
                r_last = arr[:, t - window]

            r_t = arr[:, t]
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
def prepare_returns_array(returns_df, symbols):
    returns_wide_df = returns_df.pivot(on="Ticker", index="Date")
    returns_arr = returns_wide_df.select(symbols).to_numpy().T  # stocks x time
    ma_window = 50

    returns_wide_df
    return returns_arr, returns_wide_df


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
    returns_wide_df,
    rolling_window_covariance_estimate_naive,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    _DIM = "#6b7280"
    _dates = returns_wide_df["Date"].to_numpy()
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
        _ax.plot(_dates, _std_raw[_i], color=_c, linewidth=1.0, alpha=0.9, zorder=2, label="raw")
        _ax.plot(
            _dates,
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
        # zero-mean EWMA second moment, as in the paper: matches ewma_covariance_estimate
        N, T = samples.shape
        weights = (beta ** np.arange(T))[::-1]
        factor = (1 - beta) / (1 - beta**T)
        cov_t = factor * (weights[None, :] * samples) @ samples.T  # (N, N)
        return cov_t

    def ewma_covariance_estimate_naive(arr, beta: float, shrinkage_alpha: float = 0.01):
        # arr.shape = (stocks, time)
        N, T = arr.shape
        covariances = []

        for t in range(1, T + 1):
            samples = arr[:, :t]  # (N, t)
            cov_t = ewma_cov(samples, beta)
            cov_t = _regularize_covariance(cov_t, shrinkage_alpha)
            covariances.append(cov_t)

        return np.stack(covariances, axis=-1)

    def ewma_covariance_estimate(arr: np.ndarray, beta: float, shrinkage_alpha: float = 0.01):
        # arr.shape = (stocks, time)
        N, T = arr.shape

        r_t = arr[:, 0]
        cov_t = np.outer(r_t, r_t)  # raw recursion state, never shrunk

        # shrink only the stored output, so every step is regularized like the naive version
        covariances = [_regularize_covariance(cov_t, shrinkage_alpha)]

        for t in range(1, T):
            r_t = arr[:, t]

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
    _slice = returns_arr[:, :200]
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
    returns_wide_df,
    rolling_window_covariance_estimate,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    _DIM = "#6b7280"
    _dates = returns_wide_df["Date"].to_numpy()
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
        _ax.plot(_dates, _std_ewma[_i], color=_c, linewidth=1.0, alpha=0.9, zorder=2, label="ewma")
        _ax.plot(
            _dates,
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
    returns_wide_df,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
):
    import itertools

    _INK = "#111827"
    _dates = returns_wide_df["Date"].to_numpy()

    _half_life = 20
    _cov_ewma = ewma_covariance_estimate(returns_arr, beta=0.5 ** (1 / _half_life))

    _fig, _axes = new_grid(plt, 5, 5, figsize=(14, 12))

    for _i, (_symbol_1, _symbol_2) in enumerate(itertools.product(symbols, symbols)):
        x, y = divmod(_i, 5)
        _ax = _axes[x, y]
        _c = _INK if _symbol_1 != _symbol_2 else symbol_to_color[_symbol_1]

        _ax.plot(_dates, _cov_ewma[x, y, :], color=_c, linewidth=0.9, alpha=0.85, zorder=2)
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
        entry [..., t] is the prediction for day t made from returns[:, :t] only.

        clip_at winsorizes the standardized returns at +-clip_at before the correlation EWMA
        (the paper's rule of thumb is 4.2). Pass None to disable; the paper's reference code
        (cvxgrp/cov_pred_finance) does not winsorize in its stock experiment.
        """
        covs = ewma_covariance_estimate(returns, beta=0.5 ** (1 / h_vol))  # (N, N, T)
        std_devs = _one_step_ahead(np.diagonal(covs, axis1=0, axis2=1).T ** 0.5)  # (N, T)

        returns_standardized = returns / std_devs  # (N, T), sigma_t excludes r_t
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
    returns_wide_df,
):
    _h_vol, _h_corr = 20, 60
    _cov_mat = iterated_ewma(returns_arr, h_vol=_h_vol, h_corr=_h_corr)
    _dates = returns_wide_df["Date"].to_numpy()

    plot_covariance_matrices(_cov_mat, _dates, _h_vol, _h_corr)
    return


@app.cell(hide_code=True)
def section_hierarchical_residuals(mo):
    mo.md(r"""
    # Hierarchical Residuals (Market, Sector, Idiosyncratic)

    A different way to build a covariance matrix is to peel off layers of common variation one at a time,
    in a fixed economic order, much like Gram-Schmidt. Let $r_t \in \mathbf{R}^n$ be the returns of $n$ stocks.

    ## Layer 1: market

    Take an equal-weighted market return $m_t = \tfrac{1}{n}\mathbf{1}^T r_t$ and regress every
    stock on it, using second moments as in the rest of this notebook (no intercept):

    $$\beta_i = {\mathbf{cov}(r_i, m) \over \mathbf{var}(m)} = {\sum_t r_{i,t}\, m_t \over \sum_t m_t^2}, \qquad e^{(1)}_{i,t} = r_{i,t} - \beta_i m_t .$$

    Treat each stock's return series $r_i \in \mathbf{R}^T$ and the market series $m \in \mathbf{R}^T$ as vectors
    indexed by time. Then $\beta_i m$ is the projection of $r_i$ onto $m$, and $e^{(1)}_i = r_i - \beta_i m$ is what
    is left over, exactly orthogonal to $m$. This is one Gram-Schmidt step with $m$ as the first vector, and the
    residual is the part of the stock's return that the market does not explain. Here $\mathbf{cov}$ and
    $\mathbf{var}$ are second moments (the means are not subtracted, the zero-mean convention used in the rest of
    this notebook), which moves the betas by less than 0.01 on this data.
    """)
    return


@app.cell
def plot_market_residual_series(
    market_factor,
    market_residual_returns,
    new_grid,
    np,
    plt,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
    title_figure,
    universe,
    universe_returns_wide_df,
):
    _DIM = "#6b7280"
    _dates = universe_returns_wide_df["Date"].to_numpy()

    # first the market return itself, then the part of each FAANG stock's return that the market does not explain
    _panels = [("Market", market_factor, "#374151")] + [
        (_ticker, market_residual_returns[universe.index(_ticker)], symbol_to_color[_ticker]) for _ticker in symbols
    ]
    _all_series = np.stack([_series for _, _series, _ in _panels])

    # shared y-limits cover the central 99.5% of points, so a handful of extreme days do not squash the rest
    _ylim = 1.1 * np.quantile(np.abs(_all_series), 0.995)
    _n_clipped = int((np.abs(_all_series) > _ylim).sum())

    _fig, _axes = new_grid(plt, 2, 3, figsize=(12, 5), sharey=True)
    for _ax, (_name, _series, _color) in zip(_axes.flat, _panels):
        _ax.plot(_dates, _series, color=_color, linewidth=0.9, alpha=0.75, zorder=2)
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
    _ylim = 1.1 * np.quantile(np.abs(universe_returns_arr[_idx]), 0.995)
    _n_clipped = int((np.abs(universe_returns_arr[_idx]) > _ylim).sum())

    for _ax, _ticker in zip(_axes, symbols):
        _i = universe.index(_ticker)
        _r = universe_returns_arr[_i]
        _e = market_residual_returns[_i]
        _r_squared = 1 - (_e**2).sum() / (_r**2).sum()  # second-moment R^2, matching the no-intercept fit

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
def hierarchical_residual_functions(np):
    def project_out(returns: np.ndarray, factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project every row of returns onto factor (no intercept) and remove it, one Gram-Schmidt step.

        returns: (n, T), factor: (T,). Returns (betas, residuals) with
            betas[i] = <returns[i], factor> / <factor, factor>,   residuals[i] = returns[i] - betas[i] * factor,
        and every residual row is orthogonal to factor.
        """
        betas = returns @ factor / (factor @ factor)
        return betas, returns - np.outer(betas, factor)

    def rolling_betas(returns: np.ndarray, factor: np.ndarray, window: int) -> np.ndarray:
        """
        Look-ahead-free version of project_out's betas, as a time series of shape (n, T).

        betas[i, t] uses only days t - window, ..., t - 1, so it is known before day t. Until a full window
        of history exists (t < window) the beta is set to 1.
        """
        n, T = returns.shape
        # rolling sums of returns * factor and factor^2, from cumulative sums with a leading zero
        cross = np.cumsum(np.concatenate([np.zeros((n, 1)), returns * factor], axis=1), axis=1)  # (n, T + 1)
        square = np.cumsum(np.concatenate([[0.0], factor**2]))  # (T + 1,)

        betas = np.ones((n, T))
        betas[:, window:] = (cross[:, window:T] - cross[:, : T - window]) / (square[window:T] - square[: T - window])
        return betas

    def sector_indices(universe: list[str], sector_of: dict[str, str]) -> dict[str, np.ndarray]:
        """Map each sector to the row indices of its stocks in universe, in the order the sectors are listed."""
        return {
            sector: np.array([i for i, ticker in enumerate(universe) if sector_of[ticker] == sector])
            for sector in dict.fromkeys(sector_of.values())
        }

    def project_out_sector_factors(
        residuals: np.ndarray,  # (n, T), returns after the market layer
        sector_rows: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Second layer: within each sector, the factor is the average of the stocks' residuals, and each stock's
        residual is projected onto it. Returns (sector_factors, betas, residuals), with betas of shape (n,).
        """
        betas = np.zeros(residuals.shape[0])
        new_residuals = residuals.copy()
        sector_factors = {}
        for sector, rows in sector_rows.items():
            factor = residuals[rows].mean(axis=0)
            betas[rows], new_residuals[rows] = project_out(residuals[rows], factor)
            sector_factors[sector] = factor
        return sector_factors, betas, new_residuals

    def variance_shares(
        returns: np.ndarray,
        market_residuals: np.ndarray,
        sector_residuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Share of each stock's second-moment variance explained by the market, sector, and idiosyncratic layers.

        The three parts (returns - market_residuals, market_residuals - sector_residuals, sector_residuals) are
        mutually orthogonal, so the shares add up to exactly 1 for every stock.
        """
        total = (returns**2).mean(axis=1)
        parts = (returns - market_residuals, market_residuals - sector_residuals, sector_residuals)
        market_share, sector_share, idiosyncratic_share = ((part**2).mean(axis=1) / total for part in parts)
        return market_share, sector_share, idiosyncratic_share

    def sector_blocks(sector_rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row order that groups stocks by sector, with the start and end position of each sector's block."""
        order = np.concatenate(list(sector_rows.values()))
        block_ends = np.cumsum([len(rows) for rows in sector_rows.values()])
        return order, np.concatenate([[0], block_ends[:-1]]), block_ends

    def same_sector_mask(n: int, sector_rows: dict[str, np.ndarray]) -> np.ndarray:
        """(n, n) boolean matrix that is True for pairs of stocks in the same sector (including i = j)."""
        mask = np.zeros((n, n), dtype=bool)
        for rows in sector_rows.values():
            mask[np.ix_(rows, rows)] = True
        return mask

    def within_and_across(corr: np.ndarray, same_sector: np.ndarray) -> tuple[float, float]:
        """Average correlation of distinct pairs in the same sector, and of pairs in different sectors."""
        off_diagonal = ~np.eye(len(corr), dtype=bool)
        return corr[same_sector & off_diagonal].mean(), corr[~same_sector].mean()

    def first_principal_component(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        First principal component of the (uncentered) returns, matching the second-moment convention above.

        Returns (loadings, scores, explained variance ratio), with unit loadings of shape (n,) whose sign is
        fixed so that they sum to a positive number, and scores of shape (T,).
        """
        U, singular_values, _ = np.linalg.svd(returns, full_matrices=False)
        loadings = U[:, 0] * np.sign(U[:, 0].sum())
        return loadings, loadings @ returns, singular_values[0] ** 2 / np.sum(singular_values**2)

    def rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Correlation of x and y over each trailing window of the given length, starting at day window - 1."""
        return np.array([np.corrcoef(x[t - window : t], y[t - window : t])[0, 1] for t in range(window, len(x) + 1)])

    return (
        first_principal_component,
        project_out,
        project_out_sector_factors,
        rolling_betas,
        rolling_correlation,
        same_sector_mask,
        sector_blocks,
        sector_indices,
        variance_shares,
        within_and_across,
    )


@app.cell
def market_layer(project_out, universe_returns_arr):
    market_factor = universe_returns_arr.mean(axis=0)  # equal-weighted market return, (T,)
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
    universe_returns_arr,
    universe_returns_wide_df,
):
    _DIM = "#6b7280"
    _dates = universe_returns_wide_df["Date"].to_numpy()
    _R = universe_returns_arr
    _window = 63

    _loadings, _pc1, _explained = first_principal_component(_R)
    _cos_equal_weights = _loadings.sum() / np.sqrt(len(_loadings))  # cosine between PC1 loadings and the equal-weight market
    _corr = np.corrcoef(market_factor, _pc1)[0, 1]
    _residual_corr = np.array([np.corrcoef(_e, _pc1)[0, 1] for _e in market_residual_returns])
    _rolling = rolling_correlation(market_factor, _pc1, _window)

    _fig, (_ax_scatter, _ax_roll) = new_grid(plt, 1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2]})

    _ax_scatter.scatter(market_factor, _pc1, s=6, color="tab:blue", alpha=0.35, linewidths=0, zorder=2)
    _ax_scatter.set_xlabel("market factor (equal-weighted return)", color=_DIM, fontsize=8.5)
    _ax_scatter.set_ylabel("PC1 score", color=_DIM, fontsize=8.5)
    style_plain_ax(_ax_scatter, labelsize=8, x_fmt=lambda v, _: f"{v:.0%}")
    title_axes(_ax_scatter, f"corr = {_corr:.3f}", fontsize=9, pad=6)

    _ax_roll.plot(_dates[_window - 1 :], _rolling, color="tab:blue", linewidth=1.2, zorder=2)
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
    estimated from the previous $W = 40$ days only:

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


@app.cell
def rolling_market_beta(
    market_factor,
    np,
    rolling_betas,
    universe_returns_arr,
):
    beta_window_days = 40

    rolling_market_betas = rolling_betas(universe_returns_arr, market_factor, beta_window_days)  # (n, T)
    rolling_market_residual_returns = universe_returns_arr - rolling_market_betas * market_factor

    # m is the average of the stocks, so the betas average to exactly 1 in every window, which is why beta = 1 is the default
    np.testing.assert_allclose(rolling_market_betas.mean(axis=0), 1.0)
    return beta_window_days, rolling_market_betas


@app.cell
def plot_rolling_market_betas(
    beta_window_days,
    legend_above_axes,
    market_betas,
    plt,
    reference_line,
    rolling_market_betas,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
    universe,
    universe_returns_wide_df,
):
    _dates = universe_returns_wide_df["Date"].to_numpy()

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _ax.axvspan(_dates[0], _dates[beta_window_days - 1], color="#e5e7eb", alpha=0.7, linewidth=0, zorder=0)
    reference_line(_ax, 1)
    for _ticker in symbols:  # same order as the close price legend
        _i = universe.index(_ticker)
        _c = symbol_to_color[_ticker]
        _ax.plot(_dates, rolling_market_betas[_i], color=_c, linewidth=1.2, alpha=0.9, zorder=2, label=_ticker)
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
    plt.gcf()
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
    slightly negative. Fifty days is short for fifty stocks, so these estimates are noisy, and the residuals use the
    full-sample betas from Layer 1.
    """)
    return


@app.cell
def sector_map(universe):
    # Best-guess GICS-style sectors for the 50 tickers (V and MA were reclassified to Financials in 2023)
    sector_tickers = {
        "Information Technology": [
            "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "IBM",
        ],
        "Communication Services": ["META", "GOOG", "NFLX", "DIS", "T", "VZ"],
        "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD", "LOW"],
        "Consumer Staples": ["WMT", "PG", "KO", "PEP", "COST"],
        "Health Care": ["UNH", "JNJ", "MRK", "LLY", "ABBV", "PFE", "TMO", "ABT", "AMGN"],
        "Financials": ["JPM", "V", "MA", "BAC", "WFC", "GS"],
        "Industrials": ["CAT", "HON", "UPS"],
        "Energy": ["XOM", "CVX"],
    }  # fmt: skip
    sector_of = {ticker: sector for sector, tickers in sector_tickers.items() for ticker in tickers}
    assert set(sector_of) == set(universe), f"sector map does not match the universe: {set(sector_of) ^ set(universe)}"
    return (sector_of,)


@app.cell(hide_code=True)
def correlation_window_selector(mo, universe_returns_arr):
    correlation_window_days = 50
    correlation_end_slider = mo.ui.slider(
        start=correlation_window_days,
        stop=universe_returns_arr.shape[1],
        step=1,
        value=universe_returns_arr.shape[1],
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
    title_figure,
    universe,
    universe_returns_arr,
    universe_returns_wide_df,
    within_and_across,
):
    _DIM = "#6b7280"
    _n = len(universe)

    _end = correlation_end_slider.value
    _window = slice(_end - correlation_window_days, _end)
    _corr = np.corrcoef(universe_returns_arr[:, _window])  # (n, n)
    _corr_residual = np.corrcoef(market_residual_returns[:, _window])  # same window, market removed

    _random_order = np.random.default_rng(0).permutation(_n)
    _sector_rows = sector_indices(universe, sector_of)
    _sector_order, _block_starts, _block_ends = sector_blocks(_sector_rows)

    _fig, _axes = plt.subplots(1, 3, figsize=(22, 7.5))
    _fig.patch.set_facecolor("#ffffff")
    _panels = (
        (_axes[0], _corr, _random_order, "Arbitrary order"),
        (_axes[1], _corr, _sector_order, "Grouped by sector"),
        (_axes[2], _corr_residual, _sector_order, "Grouped by sector, market removed"),
    )
    for _ax, _matrix, _order, _name in _panels:
        _image = draw_correlation_matrix(_ax, _matrix, _order, universe, _name)

    # mark the sector blocks on the reordered matrices, and name them to the right of the last one
    for _ax in _axes[1:]:
        mark_sector_blocks(_ax, _block_ends)
    label_sector_blocks(_axes[2], _sector_rows, _block_starts, _block_ends, _n)

    # average correlation of pairs in the same sector against pairs in different sectors
    _same_sector = same_sector_mask(_n, _sector_rows)
    _within, _across = within_and_across(_corr, _same_sector)
    _within_residual, _across_residual = within_and_across(_corr_residual, _same_sector)

    _fig.subplots_adjust(left=0.03, right=0.91, bottom=0.2, top=0.9, wspace=0.08)
    _fig.colorbar(_image, cax=_fig.add_axes([0.37, 0.085, 0.2, 0.015]), orientation="horizontal", label="correlation")
    _end_date = str(universe_returns_wide_df["Date"][_end - 1])[:10]
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

    Taking the covariance of the three terms $\beta m$, $\Gamma s$ and $e^{(2)}$ gives a variance for each plus cross
    terms. Some cross terms vanish exactly: $m$ is uncorrelated with every sector factor and every $e^{(2)}_i$, and
    each $e^{(2)}_i$ is uncorrelated with its own sector's factor. Others do not:

    - different sectors' factors can be correlated with each other,
    - the residuals $e^{(2)}_i$ of different stocks can be correlated, and within a sector they sum to exactly zero
      because $\sum_{i \in S} \gamma_i = \lvert S \rvert$,
    - a sector factor can be correlated with the residuals of stocks in other sectors.

    Dropping those terms gives the model covariance

    $$\hat{\Sigma} = \sigma_m^2\, \beta\beta^T + \sum_S \sigma_S^2\, \gamma_S \gamma_S^T + D,$$

    where $\sigma_m^2 = \mathbf{var}(m)$, $\sigma_S^2 = \mathbf{var}(s_S)$, $\gamma_S$ is $\gamma$ restricted to sector $S$
    (zero elsewhere), and $D = \mathbf{diag}\left(\mathbf{var}(e^{(2)}_1), \ldots, \mathbf{var}(e^{(2)}_n)\right)$.
    This is an approximation to the covariance of $r_t$, not an identity. The matrix has about $2n$ parameters plus
    the factor variances, and it is full rank because $D \succ 0$.

    Caveats. The sector map above is my best guess at GICS-style sectors, so check it. The betas in both layers
    are estimated on the full sample, so the residuals are in-sample. A predictor would need betas from past data
    only, for example from EWMAs. Energy and Industrials have only two and three stocks, so their sector factors
    are weak.
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
        _ylim = 1.1 * np.quantile(np.abs(market_residual_returns[_stocks]), 0.995)
        _n_clipped += int((np.abs(market_residual_returns[_stocks]) > _ylim).sum())

        for _col, _i in enumerate(_stocks):
            _ax = _axes[_row, _col]
            _market_residual = market_residual_returns[_i]
            # share of the market residual's second moment that the sector factor explains
            _r_squared = 1 - (sector_residual_returns[_i] ** 2).sum() / (_market_residual**2).sum()

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
    sector_factor_returns,
    sector_of,
    sector_residual_returns,
    style_ax,
    symbol_to_color,
    symbols,
    title_axes,
    title_figure,
    universe,
    universe_returns_wide_df,
):
    _INK = "#111827"
    _DIM = "#6b7280"
    _dates = universe_returns_wide_df["Date"].to_numpy()

    # for each FAANG stock: what is left after the market and the sector, and the sector factor that was removed
    _panels = []
    for _ticker in symbols:
        _sector = sector_of[_ticker]
        _panels.append(
            (_ticker, _sector, sector_residual_returns[universe.index(_ticker)], sector_factor_returns[_sector])
        )
    _all_series = np.stack([_series for _, _, _residual, _factor in _panels for _series in (_residual, _factor)])

    # shared y-limits cover the central 99.5% of points, so a handful of extreme days do not squash the rest
    _ylim = 1.1 * np.quantile(np.abs(_all_series), 0.995)
    _n_clipped = int((np.abs(_all_series) > _ylim).sum())

    _fig, _axes = new_grid(plt, 2, 3, figsize=(12, 5), sharey=True)
    for _ax, (_ticker, _sector, _residual, _factor) in zip(_axes.flat, _panels):
        _ax.plot(_dates, _residual, color=symbol_to_color[_ticker], linewidth=0.9, alpha=0.75, zorder=2)
        _ax.plot(_dates, _factor, color=_INK, linewidth=1.0, alpha=0.9, zorder=3)
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
    title_figure,
    universe,
    universe_returns_wide_df,
    within_and_across,
):
    _DIM = "#6b7280"
    _n = len(universe)

    _end = correlation_end_slider.value
    _window = slice(_end - correlation_window_days, _end)
    _corr_market_removed = np.corrcoef(market_residual_returns[:, _window])
    _corr_sector_removed = np.corrcoef(sector_residual_returns[:, _window])

    _sector_rows = sector_indices(universe, sector_of)
    _order, _block_starts, _block_ends = sector_blocks(_sector_rows)

    _fig, _axes = plt.subplots(1, 2, figsize=(15, 7.5))
    _fig.patch.set_facecolor("#ffffff")
    for _ax, _matrix, _name in (
        (_axes[0], _corr_market_removed, "Market removed"),
        (_axes[1], _corr_sector_removed, "Market and sector removed"),
    ):
        _image = draw_correlation_matrix(_ax, _matrix, _order, universe, _name)
        mark_sector_blocks(_ax, _block_ends)
    label_sector_blocks(_axes[1], _sector_rows, _block_starts, _block_ends, _n)

    _same_sector = same_sector_mask(_n, _sector_rows)
    _within_market, _across_market = within_and_across(_corr_market_removed, _same_sector)
    _within_sector, _across_sector = within_and_across(_corr_sector_removed, _same_sector)

    _fig.subplots_adjust(left=0.04, right=0.84, bottom=0.2, top=0.9, wspace=0.08)
    _fig.colorbar(_image, cax=_fig.add_axes([0.3, 0.085, 0.25, 0.015]), orientation="horizontal", label="correlation")
    _end_date = str(universe_returns_wide_df["Date"][_end - 1])[:10]
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
def prepare_universe_returns(pl, universe, universe_df):
    # the mixture experiments use every ticker in the universe, not just the five that are plotted above
    universe_returns_df = universe_df.select(
        "Ticker", "Date", Return=pl.col("Close").pct_change().over("Ticker")
    ).filter(pl.col("Return").is_not_null())
    universe_returns_wide_df = universe_returns_df.pivot(on="Ticker", index="Date")
    universe_returns_arr = universe_returns_wide_df.select(universe).to_numpy().T  # stocks x time
    return universe_returns_arr, universe_returns_wide_df


@app.cell
def mixture_components(iterated_ewma, np, universe_returns_arr):
    import cvxpy as cp

    # Half-life pairs (h_vol, h_corr) in days — "Stocks" configuration from Boyd et al. 2023
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
        """Covariance prediction for day t: (L L^T)^{-1} with L = sum_k weights[k] * L^(k)_t, an (N, N) matrix."""
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
        returns: np.ndarray,  # (N, window)
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
        _, window = returns.shape
        w = cp.Variable(K, nonneg=True)

        obj = 0
        for t in range(window):
            # (N, K): column k holds diagonal of L^(k)_t; diags @ w gives diag(L_combined_t)
            diags = np.column_stack([np.diag(cholesky_factors[k][t]) for k in range(K)])
            L_t = sum(w[k] * cholesky_factors[k][t] for k in range(K))
            obj += cp.sum(cp.log(diags @ w)) - 0.5 * cp.sum_squares(L_t.T @ returns[:, t])

        try:
            cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1]).solve()
        except cp.SolverError:
            return None

        return w.value

    def solve_precision_mixture_naive(
        returns: np.ndarray,  # (N, T)
        cov_matrices: list[np.ndarray],  # K arrays, each (N, N, T)
        look_back: int = 10,
    ) -> np.ndarray:
        """
        CW-IEWMA (Boyd et al. 2023): for each time t, find mixture weights that maximize
        log-likelihood over [t-look_back, t), then apply them to predict the covariance at t.

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

            w_opt = _solve_optimal_weights([Lk[t - look_back : t] for Lk in L_all], returns[:, t - look_back : t])
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
        B = np.stack([np.einsum("tji,jt->ti", L_k, returns) for L_k in L_all], axis=-1)  # (T, N, K), L_k^T r_t
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
        returns: np.ndarray,  # (N, T)
        cov_matrices: list[np.ndarray],  # K arrays, each (N, N, T)
        look_back: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Same predictor as solve_precision_mixture_naive, but with the K x K reformulation of the weight problem.

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
def plot_mixture_weights(
    half_life_pairs,
    mo,
    np,
    plt,
    style_ax,
    title_axes,
    universe,
    universe_returns_wide_df,
    weights_kxk,
    weights_smoothing_radio,
):
    _INK = "#111827"
    _DIM = "#6b7280"
    _k = weights_smoothing_radio.value

    # weights are aligned with the last T - look_back - 1 dates
    _dates = universe_returns_wide_df["Date"].to_numpy()[-weights_kxk.shape[1] :]

    # solver tolerance can leave tiny negatives, so clip and renormalize onto the simplex
    _w = np.clip(weights_kxk, 0.0, None)
    _w = _w / _w.sum(axis=0, keepdims=True)

    # trailing mean over the last _k days (shorter at the start); averages of simplex points stay on the simplex
    _csum = np.cumsum(np.concatenate([np.zeros((_w.shape[0], 1)), _w], axis=1), axis=1)
    _start = np.maximum(np.arange(1, _w.shape[1] + 1) - _k, 0)
    _w = (_csum[:, 1:] - _csum[:, _start]) / (np.arange(1, _w.shape[1] + 1) - _start)

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

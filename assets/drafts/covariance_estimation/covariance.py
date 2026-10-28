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


@app.cell(hide_code=True)
def section_load_stock_returns(mo):
    mo.md(r"""
    # Load Stock Returns
    """)
    return


@app.cell
def load_stock_data(Path, cs, end_date, pl, start_date, yf):
    data_file = Path("stocks.parquet")
    symbols = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
    colors = ["tab:blue", "tab:gray", "tab:purple", "tab:red", "tab:green"]
    symbol_to_color = dict(zip(symbols, colors))

    if data_file.exists():
        df = pl.read_parquet(data_file)
    else:
        frames = []

        for symbol in symbols:
            df = yf.Ticker(symbol).history(start=start_date, end=end_date)
            df = pl.from_pandas(df.reset_index()).with_columns(pl.lit(symbol).alias("Ticker"))
            frames.append(df)

        df = pl.concat(frames, how="vertical").sort("Ticker", "Date")
        df.write_parquet(data_file)

    df = df.select(cs.by_name("Ticker", "Date"), ~cs.by_name("Ticker", "Date"))
    df
    return df, symbol_to_color, symbols


@app.cell
def plot_close_prices(df, plt, style_ax, symbol_to_color):
    _INK = "#111827"

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    for (ticker,), _sub_df in df.partition_by("Ticker", as_dict=True, maintain_order=True).items():
        _c = symbol_to_color[ticker]
        _dates = _sub_df["Date"].to_numpy()
        _closes = _sub_df["Close"].to_numpy()
        _ax.plot(_dates, _closes, color=_c, linewidth=1.6, alpha=0.9, zorder=2)
        _ax.scatter([_dates[-1]], [_closes[-1]], color=_c, s=22, zorder=5, linewidths=0)

    style_ax(_ax, fig=_fig, date_interval=6, y_fmt=lambda v, _: f"${v:,.0f}")
    _ax.set_ylim(0, 1000)
    _handles = [
        plt.Line2D([0], [0], color=symbol_to_color[t], linewidth=2, label=t)
        for t in ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
    ]
    _ax.legend(
        handles=_handles,
        frameon=True,
        loc="upper left",
        facecolor="#f9fafb",
        edgecolor="#d1d5db",
        labelcolor=_INK,
        fontsize=8.5,
        handlelength=1.3,
        borderpad=0.9,
        borderaxespad=0.8,
    )
    _ax.set_title(
        "FAANG  ·  Daily Close Price  (2020 – 2024)",
        color=_INK,
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
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
def plot_return_timeseries(plt, returns_df, style_ax, symbol_to_color):
    _INK = "#111827"

    _fig, _axes = plt.subplots(2, 3, figsize=(12, 5))
    _fig.patch.set_facecolor("#ffffff")
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
        _ax.set_title(_ticker, color=_INK, fontsize=9, fontweight="bold", loc="left", pad=5)

    _axes.flat[-1].set_visible(False)
    _fig.suptitle(
        "FAANG  ·  Daily Close Return  (2020 – 2024)",
        color=_INK,
        fontsize=11,
        fontweight="bold",
        x=0.0,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.gcf()
    return


@app.cell
def plot_cumulative_returns(
    pl,
    plt,
    returns_df,
    style_ax,
    symbol_to_color,
    symbols,
):
    _INK = "#111827"

    _cumulative_df = returns_df.with_columns(
        CumReturn=(1 + pl.col("Return")).cum_prod().over("Ticker") - 1,
    )

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _by_ticker = _cumulative_df.partition_by("Ticker", as_dict=True)
    for _ticker in symbols:  # same order as the close price legend
        _sub_df = _by_ticker[(_ticker,)]
        _c = symbol_to_color[_ticker]
        _dates = _sub_df["Date"].to_numpy()
        _cum = _sub_df["CumReturn"].to_numpy()
        _ax.plot(_dates, _cum, color=_c, linewidth=1.6, alpha=0.9, zorder=2, label=_ticker)
        _ax.scatter([_dates[-1]], [_cum[-1]], color=_c, s=22, zorder=5, linewidths=0)

    _ax.axhline(0, color="#9ca3af", linewidth=0.8, zorder=1)
    style_ax(_ax, fig=_fig, date_interval=6, y_fmt=lambda v, _: f"{v:.0%}")
    _ax.legend(
        frameon=True,
        loc="upper left",
        facecolor="#f9fafb",
        edgecolor="#d1d5db",
        labelcolor=_INK,
        fontsize=8.5,
        handlelength=1.3,
        borderpad=0.9,
        borderaxespad=0.8,
    )
    _ax.set_title(
        "FAANG  ·  Cumulative Return  (2020 – 2024)",
        color=_INK,
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
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
    ma_returns_window_radio,
    mo,
    pl,
    plt,
    returns_df,
    style_ax,
    symbol_to_color,
    symbols,
):
    _INK = "#111827"
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

    _ax.axhline(0, color="#9ca3af", linewidth=0.8, zorder=1)
    style_ax(_ax, fig=_fig, date_interval=6, y_fmt=lambda v, _: f"{v:.1%}")
    _ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),  # above the axes, level with the title, so it never covers data
        ncol=5,
        labelcolor=_INK,
        fontsize=8.5,
        handlelength=1.3,
    )
    _ax.set_title(
        f"FAANG  ·  Close Return MA  (window = {_w}d)",
        color=_INK,
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
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
    ma_window_radio,
    mo,
    np,
    plt,
    returns_arr,
    returns_wide_df,
    rolling_window_covariance_estimate_naive,
    style_ax,
    symbol_to_color,
    symbols,
):
    _INK = "#111827"
    _DIM = "#6b7280"
    _dates = returns_wide_df["Date"].to_numpy()
    _ma_window = ma_window_radio.value

    def _std_devs(demean):
        _cov = rolling_window_covariance_estimate_naive(returns_arr, window=_ma_window, demean=demean)
        return np.stack([_cov[_i, _i, :] ** 0.5 for _i in range(len(symbols))])  # (stocks, time)

    _std_raw = _std_devs(demean=False)
    _std_dm = _std_devs(demean=True)
    _ymax = 1.05 * max(_std_raw.max(), _std_dm.max())  # shared across panels, so nothing is cut off

    _fig, _axes = plt.subplots(2, 3, figsize=(12, 5))
    _fig.patch.set_facecolor("#ffffff")
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
        _ax.set_title(_symbol, color=_INK, fontsize=9, fontweight="bold", loc="left", pad=5)
        if _i == 0:
            _ax.legend(frameon=False, fontsize=7, labelcolor=_DIM)

    _axes.flat[-1].set_visible(False)
    _fig.suptitle(
        f"FAANG  ·  Rolling Std Dev  (window = {_ma_window}d)",
        color=_INK,
        fontsize=11,
        fontweight="bold",
        x=0.0,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.94])
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

    $$\mathrm{Var}(\hat{\Sigma}) \propto \sum_i w_i^2 \quad\Longleftrightarrow\quad N_{\text{eff}} = {1 \over \sum_i w_i^2},$$

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
    equivalent_window,
    ewma_covariance_estimate,
    ewma_half_life_radio,
    mo,
    np,
    plt,
    returns_arr,
    returns_wide_df,
    rolling_window_covariance_estimate,
    style_ax,
    symbol_to_color,
    symbols,
):
    _INK = "#111827"
    _DIM = "#6b7280"
    _dates = returns_wide_df["Date"].to_numpy()
    _half_life = ewma_half_life_radio.value

    _window = equivalent_window(_half_life)  # same effective sample size as the EWMA
    _cov_ewma = ewma_covariance_estimate(returns_arr, beta=0.5 ** (1 / _half_life))
    _cov_ma = rolling_window_covariance_estimate(returns_arr, window=_window)
    _std_ewma = np.stack([_cov_ewma[_i, _i, :] ** 0.5 for _i in range(len(symbols))])  # (stocks, time)
    _std_ma = np.stack([_cov_ma[_i, _i, :] ** 0.5 for _i in range(len(symbols))])
    _ymax = 1.05 * max(_std_ewma.max(), _std_ma.max())  # shared across panels, so nothing is cut off

    _fig, _axes = plt.subplots(2, 3, figsize=(12, 5))
    _fig.patch.set_facecolor("#ffffff")
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
        _ax.set_title(_symbol, color=_INK, fontsize=9, fontweight="bold", loc="left", pad=5)
        if _i == 0:
            _ax.legend(frameon=False, fontsize=7, labelcolor=_DIM)

    _axes.flat[-1].set_visible(False)
    _fig.suptitle(
        rf"FAANG  ·  EWMA vs MA Std Dev  ($H^{{vol}}$ = {_half_life}d, MA window = {_window}d)",
        color=_INK,
        fontsize=11,
        fontweight="bold",
        x=0.0,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.94])
    mo.vstack([ewma_half_life_radio, _fig])
    return


@app.cell
def plot_full_covariance_grid(
    ewma_covariance_estimate,
    plt,
    returns_arr,
    returns_wide_df,
    style_ax,
    symbol_to_color,
    symbols,
):
    import itertools

    _INK = "#111827"
    _dates = returns_wide_df["Date"].to_numpy()

    _half_life = 20
    _cov_ewma = ewma_covariance_estimate(returns_arr, beta=0.5 ** (1 / _half_life))

    _fig, _axes = plt.subplots(5, 5, figsize=(14, 12))
    _fig.patch.set_facecolor("#ffffff")

    for _i, (_symbol_1, _symbol_2) in enumerate(itertools.product(symbols, symbols)):
        x, y = divmod(_i, 5)
        _ax = _axes[x, y]
        _c = _INK if _symbol_1 != _symbol_2 else symbol_to_color[_symbol_1]

        _ax.plot(_dates, _cov_ewma[x, y, :], color=_c, linewidth=0.9, alpha=0.85, zorder=2)
        style_ax(_ax, date_interval=24, labelsize=6.5, tick_length=3, pad=3)
        _ax.set_ylim(0, 0.005)

        if x == 0:
            _ax.set_title(_symbol_2, color=_INK, fontsize=8, fontweight="bold", loc="left", pad=4)
        if y == 0:
            _ax.set_ylabel(_symbol_1, color=_INK, fontsize=8, fontweight="bold")

    _fig.suptitle(
        f"FAANG  ·  EWMA Covariance  (H = {_half_life}d)",
        color=_INK,
        fontsize=11,
        fontweight="bold",
        x=0.0,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    itertools,
    np,
    plt,
    style_ax,
    symbol_to_color,
    symbols,
):
    _INK = "#111827"
    _DIM = "#6b7280"

    def plot_covariance_matrices(cov_mat: np.ndarray, dates: np.ndarray, h_vol: float, h_corr: float):
        _fig, _axes = plt.subplots(5, 5, figsize=(14, 12))
        _fig.patch.set_facecolor("#ffffff")

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
                _ax.set_title(_symbol_2, color=_INK, fontsize=8, fontweight="bold", loc="left", pad=4)

        _fig.suptitle(
            rf"FAANG  ·  IEWMA Correlation & Std Dev  ($H^{{vol}}$ = {h_vol}d, $H^{{cor}}$ = {h_corr}d)",
            color=_INK,
            fontsize=11,
            fontweight="bold",
            x=0.0,
            ha="left",
        )
        _fig.tight_layout(rect=[0, 0, 1, 0.97])
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
def mixture_components(iterated_ewma, np, returns_arr):
    import cvxpy as cp

    # Half-life pairs (h_vol, h_corr) in days — "Stocks" configuration from Boyd et al. 2023
    half_life_pairs = [(10, 21), (21, 63), (63, 125), (125, 250), (250, 500)]

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

    cov_matrices = [iterated_ewma(returns_arr, h_vol=_h_vol, h_corr=_h_corr) for _h_vol, _h_corr in half_life_pairs]
    return cov_matrices, cp, precision_cholesky_factors


@app.cell
def precision_mixture_solver_naive(
    cov_matrices,
    cp,
    np,
    precision_cholesky_factors,
    returns_arr,
):
    # Direct formulation: build the full N x N cvxpy expression for every day in the window, every solve.

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
        K = len(cov_matrices)
        _, _, T = cov_matrices[0].shape

        L_all = precision_cholesky_factors(cov_matrices)

        cov_preds = []
        n_failed = 0
        for t in range(look_back + 1, T):  # index 0 has no history (see _one_step_ahead)
            w_opt = _solve_optimal_weights([Lk[t - look_back : t] for Lk in L_all], returns[:, t - look_back : t])
            if w_opt is None:
                n_failed += 1
                cov_preds.append(np.full(cov_matrices[0].shape[:2], np.nan))
                continue
            L_t = sum(w_opt[k] * L_all[k][t] for k in range(K))
            cov_preds.append(np.linalg.inv(L_t @ L_t.T))

        if n_failed:
            print(f"Weight solve failed on {n_failed} of {T - look_back - 1} days (predictions are NaN)")

        return np.stack(cov_preds, axis=-1)  # (N, N, T - look_back - 1)

    cov_predicted_naive = solve_precision_mixture_naive(returns_arr, cov_matrices, look_back=10)
    return (cov_predicted_naive,)


@app.cell
def precision_mixture_solver_kxk(
    cov_matrices,
    cp,
    np,
    precision_cholesky_factors,
    returns_arr,
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

    def solve_precision_mixture_kxk(
        returns: np.ndarray,  # (N, T)
        cov_matrices: list[np.ndarray],  # K arrays, each (N, N, T)
        look_back: int = 10,
    ) -> np.ndarray:
        """
        Same predictor as solve_precision_mixture_naive, but with the K x K reformulation of the weight problem.

        Returns covariance predictions of shape (N, N, T - look_back - 1), aligned with
        dates[look_back + 1:]. Days where the weight solve fails are NaN.
        """
        K = len(cov_matrices)
        n_stocks, _, T = cov_matrices[0].shape

        L_all = precision_cholesky_factors(cov_matrices)  # K arrays, each (T, N, N)

        # Per-day data for the weight problem, computed once for all days
        diags = np.stack([np.diagonal(L_k, axis1=1, axis2=2) for L_k in L_all], axis=-1)  # (T, N, K)
        B = np.stack([np.einsum("tji,jt->ti", L_k, returns) for L_k in L_all], axis=-1)  # (T, N, K), L_k^T r_t
        BtB = np.einsum("tnk,tnl->tkl", B, B)  # (T, K, K)

        problem, w, A, P_half = _weight_problem(n_stocks, K, look_back)

        cov_preds = []
        n_failed = 0
        for t in range(look_back + 1, T):  # index 0 has no history (see _one_step_ahead)
            window = slice(t - look_back, t)
            A.value = diags[window].reshape(-1, K)
            P_half.value = _psd_square_root(BtB[window].sum(axis=0))

            try:
                problem.solve()
            except cp.SolverError:
                w.value = None

            if w.value is None:
                n_failed += 1
                cov_preds.append(np.full((n_stocks, n_stocks), np.nan))
                continue

            L_t = sum(w.value[k] * L_all[k][t] for k in range(K))
            cov_preds.append(np.linalg.inv(L_t @ L_t.T))

        if n_failed:
            print(f"Weight solve failed on {n_failed} of {T - look_back - 1} days (predictions are NaN)")

        return np.stack(cov_preds, axis=-1)  # (N, N, T - look_back - 1)

    cov_predicted_kxk = solve_precision_mixture_kxk(returns_arr, cov_matrices, look_back=10)
    return (cov_predicted_kxk,)


@app.cell
def check_kxk_matches_naive(cov_predicted_kxk, cov_predicted_naive, np):
    # both formulations solve the same problem, so predictions agree up to solver tolerance
    np.testing.assert_allclose(cov_predicted_kxk, cov_predicted_naive, rtol=1e-3, atol=1e-6)
    return


if __name__ == "__main__":
    app.run()

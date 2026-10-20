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
    import scipy
    import marimo as mo

    from pathlib import Path

    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    return Path, cs, end_date, mo, np, pl, plt, scipy, start_date, yf


@app.cell
def plot_helpers():
    def make_cartesian_plane(ax):
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")


    def remove_spines(ax):
        ax.spines[["right", "top"]].set_visible(False)

    return make_cartesian_plane, remove_spines


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
    window = 5 * 4 * 3  # days in week x weeks in month x weeks in quarter

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
def define_panel_timeseries_plot(plt, remove_spines, symbol_to_color):
    def plot_panel_timeseries(df):
        plt.figure(figsize=(10, 5))

        for _i, ((_ticker,), _sub_df) in enumerate(df.partition_by("Ticker", as_dict=True, maintain_order=True).items()):
            plt.subplot(2, 3, _i + 1)
            plt.plot(_sub_df["Date"], _sub_df["Return"], label=_ticker, alpha=0.7, color=symbol_to_color[_ticker])
            remove_spines(plt.gca())
            plt.title(_ticker)
            plt.xticks(rotation=45)

        return plt.gcf()

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


@app.cell(hide_code=True)
def section_fixed_window_ma(mo):
    mo.md(r"""
    # Fixed Window Moving Average
    """)
    return


@app.cell
def plot_ma_returns(pl, plt, returns_df, style_ax, symbol_to_color):
    _INK = "#111827"

    for _w in [5, 20, 60]:  # 1 week, 1 month, 1 quarter
        _returns_ma = returns_df.select(
            "Ticker",
            "Date",
            pl.col("Return").rolling_mean(window_size=_w, min_samples=1).over("Ticker"),
            pl.row_index().over("Ticker").alias("index"),
        ).filter(pl.col("index").ge(_w))

        _fig, _axes = plt.subplots(2, 3, figsize=(12, 5))
        _fig.patch.set_facecolor("#ffffff")
        for _i, ((_ticker,), _sub_df) in enumerate(
            _returns_ma.partition_by("Ticker", as_dict=True, maintain_order=True).items()
        ):
            _ax = _axes.flat[_i]
            _ax.plot(
                _sub_df["Date"].to_numpy(),
                _sub_df["Return"].to_numpy(),
                color=symbol_to_color[_ticker],
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )
            style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4, y_fmt=lambda v, _: f"{v:.1%}")
            _ax.set_title(_ticker, color=_INK, fontsize=9, fontweight="bold", loc="left", pad=5)

        _axes.flat[-1].set_visible(False)
        _fig.suptitle(
            f"FAANG  ·  Close Return MA  (window = {_w}d)",
            color=_INK,
            fontsize=11,
            fontweight="bold",
            x=0.0,
            ha="left",
        )
        _fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()
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
    return ma_window, returns_arr, returns_wide_df


@app.cell
def plot_ma_std_dev_comparison(
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

    for _ma_window in [5, 20, 60]:
        _cov = rolling_window_covariance_estimate_naive(returns_arr, window=_ma_window, demean=False)
        _cov_dm = rolling_window_covariance_estimate_naive(returns_arr, window=_ma_window, demean=True)

        _fig, _axes = plt.subplots(2, 3, figsize=(12, 5))
        _fig.patch.set_facecolor("#ffffff")
        for _i, _symbol in enumerate(symbols):
            _ax = _axes.flat[_i]
            _c = symbol_to_color[_symbol]
            _ax.plot(_dates, _cov[_i, _i, :] ** 0.5, color=_c, linewidth=1.0, alpha=0.9, zorder=2, label="raw")
            _ax.plot(
                _dates,
                _cov_dm[_i, _i, :] ** 0.5,
                color=_c,
                linewidth=1.0,
                alpha=0.55,
                linestyle="--",
                zorder=2,
                label="demeaned",
            )
            style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4, y_fmt=lambda v, _: f"{v:.1%}")
            _ax.set_ylim(0, 0.05)
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

    plt.show()
    return


@app.cell
def plot_rolling_window_std_dev(
    ma_window,
    plt,
    returns_arr,
    returns_wide_df,
    rolling_window_covariance_estimate,
    style_ax,
    symbol_to_color,
    symbols,
):
    _INK = "#111827"
    _dates = returns_wide_df["Date"].to_numpy()

    cov_ma_2 = rolling_window_covariance_estimate(returns_arr, window=ma_window)

    _fig, _axes = plt.subplots(2, 3, figsize=(12, 5))
    _fig.patch.set_facecolor("#ffffff")
    for _i, _symbol in enumerate(symbols):
        _ax = _axes.flat[_i]
        _ax.plot(_dates, cov_ma_2[_i, _i, :] ** 0.5, color=symbol_to_color[_symbol], linewidth=1.0, alpha=0.9, zorder=2)
        style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4, y_fmt=lambda v, _: f"{v:.1%}")
        _ax.set_ylim(0, 0.05)
        _ax.set_title(_symbol, color=_INK, fontsize=9, fontweight="bold", loc="left", pad=5)

    _axes.flat[-1].set_visible(False)
    _fig.suptitle(
        f"FAANG  ·  Rolling Std Dev  (window = {ma_window}d)",
        color=_INK,
        fontsize=11,
        fontweight="bold",
        x=0.0,
        ha="left",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.gcf()
    return (cov_ma_2,)


@app.cell
def section_ewma_formula(mo):
    mo.md(r"""
    ## Exponentially Weighted Moving Average

    $$\hat{\Sigma}_{t+1} = {\beta - \beta^t \over 1-\beta^t}\hat{\Sigma}_t + {1-\beta\over 1- \beta^t} r_tr_t^T $$
    """)
    return


@app.cell
def ewma_kernel_scratch():
    # _beta = 0.7
    # plt.figure(figsize=(6, 6))

    # _kernel = _beta ** np.arange(10)
    # markerline, stemlines, baseline = plt.stem(np.arange(10), _kernel, basefmt=" ")
    # plt.setp(stemlines, "linewidth", 2)
    # plt.setp(markerline, markersize=7)
    # make_cartesian_plane(plt.gca())
    # plt.tight_layout()

    # plt.figure(figsize=(6, 6))
    # _y = np.array([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0 , 0])
    # markerline, stemlines, baseline = plt.stem(np.arange(len(_y)), _y, "tab:orange", basefmt=" ")
    # plt.setp(stemlines, "linewidth", 2)
    # plt.setp(markerline, markersize=7)
    # make_cartesian_plane(plt.gca())
    # plt.tight_layout()

    # plt.show()
    return


@app.cell
def define_discrete_convolution(np):
    def discrete_convolution(x1, n1, x2, n2):
        """
        Compute the discrete convolution of two signals x1[n] and x2[n].

        Parameters:
        -----------
        x1 : array-like
            First discrete signal values.
        n1 : array-like
            Corresponding sample indices for x1.
        x2 : array-like
            Second discrete signal values.
        n2 : array-like
            Corresponding sample indices for x2.

        Returns:
        --------
        y : np.ndarray
            Convolved signal values.
        n : np.ndarray
            Corresponding sample indices for the convolution result.
        """
        y = np.convolve(x1, x2)
        n_start = n1[0] + n2[0]
        n_end = n1[-1] + n2[-1]
        n = np.arange(n_start, n_end + 1)
        return y, n

    return (discrete_convolution,)


@app.cell
def ewma_sliders(mo):
    K = 10
    beta_slider = mo.ui.slider(0.5, 0.9, 0.05)
    tau_slider = mo.ui.slider(1, K, 1)
    return K, beta_slider, tau_slider


@app.cell
def plot_convolution_demo(
    K,
    beta_slider,
    discrete_convolution,
    make_cartesian_plane,
    np,
    plt,
    tau_slider,
):
    _beta = beta_slider.value

    plt.figure(figsize=(6, 6))
    plt.subplot(221)
    kernel = _beta ** np.arange(K)

    markerline, stemlines, baseline = plt.stem(np.arange(kernel.size), kernel, basefmt=" ")
    plt.setp(stemlines, "linewidth", 2)
    plt.setp(markerline, markersize=7)
    plt.xlim([-3, 9])
    make_cartesian_plane(plt.gca())

    plt.subplot(223)
    _y = np.array([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0, 0])
    markerline, stemlines, baseline = plt.stem(np.arange(len(_y)), _y, "tab:orange", basefmt=" ")
    plt.setp(stemlines, "linewidth", 2)
    plt.setp(markerline, markersize=7)
    plt.xlim([-3, 9])

    make_cartesian_plane(plt.gca())

    plt.subplot(222)

    _tau = tau_slider.value
    _x = np.arange(kernel.size)
    _x = _tau - _x
    markerline, stemlines, baseline = plt.stem(_x + 0.1, kernel, basefmt=" ")
    plt.setp(stemlines, "linewidth", 2, alpha=0.7)
    plt.setp(markerline, markersize=7, alpha=0.7)

    markerline, stemlines, baseline = plt.stem(np.arange(len(_y)) - 0.1, _y, "tab:orange", basefmt=" ")
    plt.setp(stemlines, "linewidth", 2)
    plt.setp(markerline, markersize=7)
    plt.xlim([-3.1, 9])
    make_cartesian_plane(plt.gca())
    plt.tight_layout()

    plt.subplot(224)
    _out, _x = discrete_convolution(kernel, np.arange(kernel.size), _y, np.arange(_y.size))
    markerline, stemlines, baseline = plt.stem(_x[:_tau], _out[:_tau], "g", basefmt=" ")
    plt.setp(stemlines, "linewidth", 2, alpha=0.5)
    plt.setp(markerline, markersize=7, alpha=0.5)
    markerline, stemlines, baseline = plt.stem(_x[_tau], _out[_tau], "g", basefmt=" ")
    plt.setp(stemlines, "linewidth", 2)
    plt.setp(markerline, markersize=7)

    plt.xlim([-3.1, 9])
    plt.ylim([0, 3])
    make_cartesian_plane(plt.gca())
    plt.tight_layout()

    fig = plt.gcf()
    return (fig,)


@app.cell
def display_convolution_widget(beta_slider, fig, mo, tau_slider):
    mo.hstack([fig, mo.md(f"tau: {tau_slider}"), mo.md(f"beta: {beta_slider}")])
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
        means = [r_t.copy()]

        mean_t = r_t
        for t in range(1, T):
            r_t = arr[:, t]

            a_1 = (beta - beta ** (t + 1)) / (1 - beta ** (t + 1))
            a_2 = (1 - beta) / (1 - beta ** (t + 1))
            mean_t = a_1 * mean_t + a_2 * r_t
            cov_t = (beta - beta ** (t + 1)) / (1 - beta ** (t + 1)) * cov_t
            cov_t += (1 - beta) / (1 - beta ** (t + 1)) * np.outer(r_t, r_t)  # (N, N)

            covariances.append(_regularize_covariance(cov_t, shrinkage_alpha))

        return np.stack(covariances, axis=-1)

    return ewma_covariance_estimate, ewma_covariance_estimate_naive


@app.cell
def test_correlate2d(np, scipy):
    _in2 = (0.5 ** np.arange(5))[None, ::-1]
    scipy.signal.correlate2d(np.ones((3, 5)), _in2)
    return


@app.cell
def plot_ewma_vs_ma_std_dev(
    ewma_covariance_estimate_naive,
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

    for _half_life in [5, 20, 60]:
        _cov_ewma = ewma_covariance_estimate_naive(returns_arr, beta=0.5 ** (1 / _half_life))
        _cov_ma = rolling_window_covariance_estimate(returns_arr, window=_half_life)

        _fig, _axes = plt.subplots(2, 3, figsize=(12, 5))
        _fig.patch.set_facecolor("#ffffff")
        for _i, _symbol in enumerate(symbols):
            _ax = _axes.flat[_i]
            _c = symbol_to_color[_symbol]
            _ax.plot(_dates, _cov_ewma[_i, _i, :] ** 0.5,
                     color=_c, linewidth=1.0, alpha=0.9, zorder=2, label="ewma")
            _ax.plot(_dates, _cov_ma[_i, _i, :] ** 0.5,
                     color=_c, linewidth=1.0, alpha=0.55, linestyle="--", zorder=2, label="ma")
            style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4,
                     y_fmt=lambda v, _: f"{v:.1%}")
            _ax.set_ylim(0, 0.06)
            _ax.set_title(_symbol, color=_INK, fontsize=9, fontweight="bold", loc="left", pad=5)
            if _i == 0:
                _ax.legend(frameon=False, fontsize=7, labelcolor=_DIM)

        _axes.flat[-1].set_visible(False)
        _fig.suptitle(
            rf"FAANG  ·  EWMA vs MA Std Dev  ($H^{{vol}}$ = {_half_life}d)",
            color=_INK, fontsize=11, fontweight="bold", x=0.0, ha="left",
        )
        _fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()
    return


@app.cell
def plot_ewma_covariance_std_dev(
    cov_ma_2,
    ewma_covariance_estimate,
    plt,
    returns_arr,
    returns_wide_df,
    style_ax,
    symbol_to_color,
    symbols,
):
    _INK = "#111827"
    _DIM = "#6b7280"
    _dates = returns_wide_df["Date"].to_numpy()

    half_life = 20
    cov_ewma_2 = ewma_covariance_estimate(returns_arr, beta=0.5 ** (1 / half_life))

    _fig, _axes = plt.subplots(2, 3, figsize=(12, 5))
    _fig.patch.set_facecolor("#ffffff")
    for _i, _symbol in enumerate(symbols):
        _ax = _axes.flat[_i]
        _c = symbol_to_color[_symbol]
        _ax.plot(_dates, cov_ewma_2[_i, _i, :] ** 0.5,
                 color=_c, linewidth=1.0, alpha=0.9, zorder=2, label="ewma")
        _ax.plot(_dates, cov_ma_2[_i, _i, :] ** 0.5,
                 color=_c, linewidth=1.0, alpha=0.55, linestyle="--", zorder=2, label="ma")
        style_ax(_ax, date_interval=12, labelsize=7.5, tick_length=4, pad=4,
                 y_fmt=lambda v, _: f"{v:.1%}")
        _ax.set_ylim(0, 0.06)
        _ax.set_title(_symbol, color=_INK, fontsize=9, fontweight="bold", loc="left", pad=5)
        if _i == 0:
            _ax.legend(frameon=False, fontsize=7, labelcolor=_DIM)

    _axes.flat[-1].set_visible(False)
    _fig.suptitle(
        rf"FAANG  ·  EWMA vs MA Std Dev  ($H^{{vol}}$ = {half_life}d)",
        color=_INK, fontsize=11, fontweight="bold", x=0.0, ha="left",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.gcf()
    return cov_ewma_2, half_life


@app.cell
def plot_full_covariance_grid(
    cov_ewma_2,
    half_life,
    plt,
    returns_wide_df,
    style_ax,
    symbol_to_color,
    symbols,
):
    import itertools

    _INK = "#111827"
    _dates = returns_wide_df["Date"].to_numpy()

    _fig, _axes = plt.subplots(5, 5, figsize=(14, 12))
    _fig.patch.set_facecolor("#ffffff")

    for _i, (_symbol_1, _symbol_2) in enumerate(itertools.product(symbols, symbols)):
        x, y = divmod(_i, 5)
        _ax = _axes[x, y]
        _c = _INK if _symbol_1 != _symbol_2 else symbol_to_color[_symbol_1]

        _ax.plot(_dates, cov_ewma_2[x, y, :], color=_c, linewidth=0.9, alpha=0.85, zorder=2)
        style_ax(_ax, date_interval=24, labelsize=6.5, tick_length=3, pad=3)
        _ax.set_ylim(0, 0.005)

        if x == 0:
            _ax.set_title(_symbol_2, color=_INK, fontsize=8, fontweight="bold", loc="left", pad=4)
        if y == 0:
            _ax.set_ylabel(_symbol_1, color=_INK, fontsize=8, fontweight="bold")

    _fig.suptitle(
        f"FAANG  ·  EWMA Covariance  (H = {half_life}d)",
        color=_INK, fontsize=11, fontweight="bold", x=0.0, ha="left",
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

    def iterated_ewma(returns: np.ndarray, h_vol: float, h_corr: float) -> np.ndarray:
        """
        IEWMA predictor. Returns covariance predictions of shape (N, N, T), where
        entry [..., t] is the prediction for day t made from returns[:, :t] only.
        """
        covs = ewma_covariance_estimate(returns, beta=0.5 ** (1 / h_vol))  # (N, N, T)
        std_devs = _one_step_ahead(np.diagonal(covs, axis1=0, axis2=1).T ** 0.5)  # (N, T)

        returns_standardized = returns / std_devs  # (N, T), sigma_t excludes r_t
        winsorized_returns = np.clip(returns_standardized, a_min=-4.2, a_max=4.2)

        R = ewma_covariance_estimate(winsorized_returns, 0.5 ** (1 / h_corr))  # (N, N, T)
        std_devs_R = np.diagonal(R, axis1=0, axis2=1).T ** 0.5  # (N, T)
        R_corr = R / std_devs_R[None, :, :] / std_devs_R[:, None, :]  # (N, N, T)
        R_corr = _one_step_ahead(R_corr)

        return std_devs[None, :, :] * R_corr * std_devs[:, None, :]

    return (iterated_ewma,)


@app.cell
def define_plot_covariance_matrices(
    half_life,
    itertools,
    np,
    plt,
    style_ax,
    symbol_to_color,
    symbols,
):
    _INK = "#111827"
    _DIM = "#6b7280"

    def plot_covariance_matrices(cov_mat: np.ndarray, dates: np.ndarray):
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

            if y == 0:
                _ax.set_ylabel(_symbol_1, color=_INK, fontsize=8, fontweight="bold")
            if x == 0:
                _ax.set_title(_symbol_2, color=_INK, fontsize=8, fontweight="bold", loc="left", pad=4)

        _fig.suptitle(
            f"FAANG  ·  EWMA Correlation & Std Dev  (H = {half_life}d)",
            color=_INK, fontsize=11, fontweight="bold", x=0.0, ha="left",
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
    _cov_mat = iterated_ewma(returns_arr, h_vol=20, h_corr=60)
    _dates = returns_wide_df["Date"].to_numpy()

    plot_covariance_matrices(_cov_mat, _dates)
    return


@app.cell
def precision_mixture_solver(iterated_ewma, np, returns_arr):
    import cvxpy as cp

    # Half-life pairs (h_vol, h_corr) in days — "Stocks" configuration from Boyd et al. 2023
    _HALF_LIFE_PAIRS = [(10, 21), (21, 63), (63, 125), (125, 250), (250, 500)]

    def _precision_cholesky_factors(
        cov_matrices: list[np.ndarray],  # K one-step-ahead predictions, each (N, N, T)
    ) -> list[np.ndarray]:
        """Compute L^(k) s.t. (Σ^(k))^{-1} = L^(k) (L^(k))^T for each component k."""
        result = []
        for cov_k in cov_matrices:
            cov_t = cov_k.transpose(2, 0, 1)         # (T, N, N)
            prec_t = np.linalg.inv(cov_t)            # (T, N, N)
            result.append(np.linalg.cholesky(prec_t))  # (T, N, N), lower triangular
        return result

    def _solve_optimal_weights(
        cholesky_factors: list[np.ndarray],  # K arrays, each (window, N, N)
        returns: np.ndarray,                  # (N, window)
    ) -> np.ndarray:
        """
        Maximize the average log-likelihood of the combined predictor over the lookback window:

            max_w  sum_t [ sum_i log(L_t_ii) - 0.5 * ||L_t^T r_t||^2 ]
            s.t.   L_t = sum_k w_k * L_t^(k),  w >= 0,  sum(w) = 1

        The objective is concave in w because log(diag(L_t)) is concave in w (affine composition).
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

        cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1]).solve()

        return w.value if w.value is not None else np.ones(K) / K

    def solve_precision_mixture(
        returns: np.ndarray,             # (N, T)
        cov_matrices: list[np.ndarray],  # K arrays, each (N, N, T)
        look_back: int = 10,
    ) -> np.ndarray:
        """
        CW-IEWMA (Boyd et al. 2023): for each time t, find mixture weights that maximize
        log-likelihood over [t-look_back, t), then apply them to predict the covariance at t.

        Returns covariance predictions of shape (N, N, T - look_back - 1), aligned with
        dates[look_back + 1:].
        """
        K = len(cov_matrices)
        _, _, T = cov_matrices[0].shape

        L_all = _precision_cholesky_factors(cov_matrices)

        cov_preds = []
        for t in range(look_back + 1, T):  # index 0 has no history (see _one_step_ahead)
            w_opt = _solve_optimal_weights([Lk[t - look_back:t] for Lk in L_all], returns[:, t - look_back:t])
            L_t = sum(w_opt[k] * L_all[k][t] for k in range(K))
            cov_preds.append(np.linalg.inv(L_t @ L_t.T))

        return np.stack(cov_preds, axis=-1)  # (N, N, T - look_back - 1)

    cov_matrices = [
        iterated_ewma(returns_arr, h_vol=_h_vol, h_corr=_h_corr)
        for _h_vol, _h_corr in _HALF_LIFE_PAIRS
    ]
    cov_predicted = solve_precision_mixture(returns_arr, cov_matrices, look_back=10)
    return (cov_matrices,)


@app.cell
def check_cov_eigenvalues(cov_matrices, np):
    np.linalg.eigh(cov_matrices[0][:, :, 10])
    return


if __name__ == "__main__":
    app.run()

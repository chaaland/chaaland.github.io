import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full", app_title="Covariance Estimation")


@app.cell
def _():
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
def _():
    def make_cartesian_plane(ax):
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")


    def remove_spines(ax):
        ax.spines[["right", "top"]].set_visible(False)
    return make_cartesian_plane, remove_spines


@app.cell
def _(mo):
    mo.md(r"""
    # Load Stock Returns
    """)
    return


@app.cell
def _(Path, cs, end_date, pl, start_date, yf):
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
def _(df, plt, remove_spines, symbol_to_color):
    plt.figure(figsize=(5, 5))

    for (ticker,), _sub_df in df.partition_by("Ticker", as_dict=True, maintain_order=True).items():
        plt.plot(_sub_df["Date"], _sub_df["Close"], label=ticker, color=symbol_to_color[ticker])

    plt.xticks(rotation=45)
    plt.legend(frameon=False)
    plt.ylim([0, 1000])
    remove_spines(plt.gca())
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title("Stock Close Price")
    plt.tight_layout()

    # mo.mpl.interactive(plt.gcf())
    plt.gcf()
    return


@app.cell
def _(df, pl):
    returns_df = df.select("Ticker", "Date", Return=pl.col("Close").pct_change().over("Ticker")).filter(
        pl.col("Return").is_not_null()
    )

    returns_df
    return (returns_df,)


@app.cell
def _(pl, returns_df):
    # check expected returns are near 0
    returns_df.group_by("Ticker").agg(pl.mean("Return"))
    return


@app.cell
def _(plt, remove_spines, symbol_to_color):
    def plot_panel_timeseries(df):
        plt.figure(figsize=(10, 5))

        for _i, ((_ticker,), _sub_df) in enumerate(df.partition_by("Ticker", as_dict=True, maintain_order=True).items()):
            plt.subplot(2, 3, _i + 1)
            plt.plot(_sub_df["Date"], _sub_df["Return"], label=_ticker, alpha=0.7, color=symbol_to_color[_ticker])
            remove_spines(plt.gca())
            plt.title(_ticker)
            plt.xticks(rotation=45)

        return plt.gcf()
    return (plot_panel_timeseries,)


@app.cell
def _(plot_panel_timeseries, plt, returns_df):
    _fig = plot_panel_timeseries(returns_df)
    plt.suptitle("Close Return")
    # plt.setp(plt.gca(), ylim=[-0.1,0.1])

    plt.tight_layout()

    # mo.mpl.interactive(_fig)
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Fixed Window Moving Average
    """)
    return


@app.cell
def _(pl, plot_panel_timeseries, plt, returns_df):
    for _w in [5, 20, 60]:  # 1 week, 1 month, 1 quarter
        _returns_ma = returns_df.select(
            "Ticker",
            "Date",
            pl.col("Return").rolling_mean(window_size=_w, min_samples=1).over("Ticker"),
            pl.row_index().over("Ticker").alias("index"),
        ).filter(pl.col("index").ge(_w))

        _fig = plot_panel_timeseries(_returns_ma)
        plt.suptitle(f"Close Return MA (window={_w})")
        plt.tight_layout()

        # mo.mpl.interactive(_fig)
        # plt.gcf()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Moving Average

    $$\hat{\Sigma}_{t+1} = {\alpha_{t+1} \over \alpha_t}\hat{\Sigma}_t + \alpha_{t+1} \left(r_tr_t^T - r_{t-M}r_{t-M}^T\right)$$
    """)
    return


@app.cell
def _(np):
    def rolling_window_covariance_estimate_naive(arr, window: int, demean: bool = False) -> np.ndarray:
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


    def rolling_window_covariance_estimate(arr, window: int) -> np.ndarray:
        N, T = arr.shape  # arr.shape = (time, stocks)

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
def _(returns_df, symbols):
    returns_wide_df = returns_df.pivot(on="Ticker", index="Date")
    returns_arr = returns_wide_df.select(symbols).to_numpy().T  # stocks x time
    ma_window = 50

    returns_wide_df
    return ma_window, returns_arr, returns_wide_df


@app.cell
def _(
    plt,
    remove_spines,
    returns_arr,
    returns_wide_df,
    rolling_window_covariance_estimate_naive,
    symbol_to_color,
    symbols,
):
    for _ma_window in [5, 20, 60]:
        cov_ma = rolling_window_covariance_estimate_naive(returns_arr, window=_ma_window, demean=False)
        cov_ma_demeaned = rolling_window_covariance_estimate_naive(returns_arr, window=_ma_window, demean=True)

        plt.figure(figsize=(10, 5))
        for _i, _symbol in enumerate(symbols):
            plt.subplot(2, 3, _i + 1)
            plt.plot(
                returns_wide_df["Date"].to_numpy(),
                cov_ma[_i, _i, :] ** 0.5,
                label=_symbol,
                alpha=0.7,
                color=symbol_to_color[_symbol],
            )
            plt.plot(
                returns_wide_df["Date"].to_numpy(),
                cov_ma_demeaned[_i, _i, :] ** 0.5,
                label=_symbol,
                alpha=0.7,
                color=symbol_to_color[_symbol],
                linestyle="--",
            )
            remove_spines(plt.gca())
            plt.title(_symbol)
            plt.xticks(rotation=45)
            plt.ylim([0, 0.05])
            if _i % 3 == 0:
                plt.ylabel(f"Rolling Std Dev (w={_ma_window})")

        plt.suptitle(f"Close Return Std (window={_ma_window})")
        plt.tight_layout()
    # mo.mpl.interactive(plt.gcf())
    # plt.gcf()
    plt.show()
    return


@app.cell
def _(
    ma_window,
    plt,
    remove_spines,
    returns_arr,
    returns_wide_df,
    rolling_window_covariance_estimate,
    symbol_to_color,
    symbols,
):
    cov_ma_2 = rolling_window_covariance_estimate(returns_arr, window=ma_window)

    plt.figure(figsize=(10, 5))
    for _i, _symbol in enumerate(symbols):
        plt.subplot(2, 3, _i + 1)
        plt.plot(
            returns_wide_df["Date"].to_numpy(),
            cov_ma_2[_i, _i, :] ** 0.5,
            label=_symbol,
            alpha=0.7,
            color=symbol_to_color[_symbol],
        )
        remove_spines(plt.gca())
        plt.title(_symbol)
        plt.xticks(rotation=45)
        plt.ylim([0, 0.05])
        if _i % 3 == 0:
            plt.ylabel(f"Rolling Std Dev (w={ma_window})")

    plt.suptitle(f"Close Return Std (window={ma_window})")
    plt.tight_layout()
    # mo.mpl.interactive(plt.gcf())
    plt.gcf()
    return (cov_ma_2,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exponentially Weighted Moving Average

    $$\hat{\Sigma}_{t+1} = {\beta - \beta^t \over 1-\beta^t}\hat{\Sigma}_t + {1-\beta\over 1- \beta^t} r_tr_t^T $$
    """)
    return


@app.cell
def _():
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
def _(np):
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
def _(mo):
    K = 10
    beta_slider = mo.ui.slider(0.5, 0.9, 0.05)
    tau_slider = mo.ui.slider(1, K, 1)
    return K, beta_slider, tau_slider


@app.cell
def _(
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
def _(beta_slider, fig, mo, tau_slider):
    mo.hstack([fig, mo.md(f"tau: {tau_slider}"), mo.md(f"beta: {beta_slider}")])
    return


@app.cell
def _(mean_t, np, scipy):
    def _regularize_covariance(cov: np.ndarray, shrinkage_alpha: float = 0.01):
        cov = (1 - shrinkage_alpha) * cov + shrinkage_alpha * np.diag(np.diag(cov))  # ensure full rank
        return cov


    def ewma(samples, beta):
        _, T = samples.shape
        weights = (beta ** np.arange(T))
        cum_weight = np.cumsum(weights)[None, :]
        means = scipy.signal.convolve2d(samples, weights[None, :])[:, :T]
        means /= cum_weight  # (N, T)

        return means


    def ewma_cov(samples, beta):
        N, T = samples.shape
        weights = (beta ** np.arange(T))[::-1]
        factor = (1 - beta) / (1 - beta**T)
        means = ewma(samples, beta)
        diff = samples - means
        # diff[:, 1:] -= means[:, :-1]  # subtract mean from one time step previous
        cov_t = factor * (weights[None, :] * diff) @ diff.T  # (N, N)
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
        cov_t = np.outer(r_t, r_t)
        cov_t = (1 - shrinkage_alpha) * cov_t + shrinkage_alpha * np.diag(np.diag(cov_t))  # ensure full rank

        covariances = [cov_t]
        means = [r_t.copy()]

        for t in range(1, T):
            r_t = arr[:, t]

            a_1 = (beta - beta ** (t + 1)) / (1 - beta ** (t + 1))
            a_2 = (1 - beta) / (1 - beta ** (t + 1))
            mean_t = a_1 * mean_t + a_2 * r_t
            cov_t = (beta - beta ** (t + 1)) / (1 - beta ** (t + 1)) * cov_t
            cov_t += (1 - beta) / (1 - beta ** (t + 1)) * np.outer(r_t, r_t)  # (N, N)

            covariances.append(cov_t)

        return np.stack(covariances, axis=-1)
    return ewma, ewma_covariance_estimate, ewma_covariance_estimate_naive


@app.cell
def _(ewma, returns_arr):
    ewma(returns_arr[:, :8], beta=0.5)
    # np.mean(returns_arr, axis=1)
    return


@app.cell
def _(np, scipy):
    _in2 = (0.5 ** np.arange(5))[None, ::-1]
    scipy.signal.correlate2d(np.ones((3, 5)), _in2)
    return


@app.cell
def _(ewma_covariance_estimate_naive, np):
    np.linalg.eig(ewma_covariance_estimate_naive(np.ones((3, 2)), 0.5)[:, :, 1])
    return


@app.cell
def _(
    ewma_covariance_estimate_naive,
    plt,
    remove_spines,
    returns_arr,
    returns_wide_df,
    rolling_window_covariance_estimate,
    symbol_to_color,
    symbols,
):
    for _half_life in [5, 20, 60]:
        # half_life = 20
        _cov_ewma = ewma_covariance_estimate_naive(returns_arr, beta=0.5 ** (1 / _half_life))
        _cov_ma_2 = rolling_window_covariance_estimate(returns_arr, window=_half_life)

        plt.figure(figsize=(10, 5))
        for _i, _symbol in enumerate(symbols):
            plt.subplot(2, 3, _i + 1)
            plt.plot(
                returns_wide_df["Date"].to_numpy(),
                _cov_ewma[_i, _i, :] ** 0.5,
                label=_symbol,
                alpha=0.7,
                color=symbol_to_color[_symbol],
            )
            plt.plot(
                returns_wide_df["Date"].to_numpy(),
                _cov_ma_2[_i, _i, :] ** 0.5,
                alpha=0.7,
                color=symbol_to_color[_symbol],
                linestyle="--",
            )

            remove_spines(plt.gca())
            plt.title(_symbol)
            plt.xticks(rotation=45)
            plt.ylim([0, 0.06])
            if _i % 3 == 0:
                plt.ylabel(rf"Rolling Std Dev ($\beta$={0.5 ** (1 / _half_life):.2f})")

        plt.suptitle(rf"Close Return Std EWMA ($H^{{vol}}$={_half_life})")
        plt.tight_layout()
    # plt.gcf()
    plt.show()
    return


@app.cell
def _(
    cov_ma_2,
    ewma_covariance_estimate,
    plt,
    remove_spines,
    returns_arr,
    returns_wide_df,
    symbol_to_color,
    symbols,
):
    half_life = 20
    cov_ewma_2 = ewma_covariance_estimate(returns_arr, beta=0.5 ** (1 / half_life))

    plt.figure(figsize=(10, 5))
    for _i, _symbol in enumerate(symbols):
        plt.subplot(2, 3, _i + 1)
        plt.plot(
            returns_wide_df["Date"].to_numpy(),
            cov_ewma_2[_i, _i, :] ** 0.5,
            label=_symbol,
            alpha=0.7,
            color=symbol_to_color[_symbol],
        )
        plt.plot(
            returns_wide_df["Date"].to_numpy(),
            cov_ma_2[_i, _i, :] ** 0.5,
            alpha=0.7,
            color=symbol_to_color[_symbol],
            linestyle="--",
        )

        remove_spines(plt.gca())
        plt.title(_symbol)
        plt.xticks(rotation=45)
        plt.ylim([0, 0.06])
        if _i % 3 == 0:
            plt.ylabel(rf"Rolling Std Dev ($\beta$={0.5 ** (1 / half_life):.2f})")

    plt.suptitle(rf"Close Return Std EWMA ($H^{{vol}}$={half_life})")
    plt.tight_layout()
    # mo.mpl.interactive(plt.gcf())
    plt.gcf()
    return cov_ewma_2, half_life


@app.cell
def _(
    cov_ewma_2,
    half_life,
    plt,
    remove_spines,
    returns_wide_df,
    symbol_to_color,
    symbols,
):
    plt.figure(figsize=(10, 10))
    import itertools

    for _i, (_symbol_1, _symbol_2) in enumerate(itertools.product(symbols, symbols)):
        plt.subplot(5, 5, _i + 1)
        _c = "k" if _symbol_1 != _symbol_2 else symbol_to_color[_symbol_1]

        plt.plot(
            returns_wide_df["Date"].to_numpy(),
            cov_ewma_2[_i // 5, _i % 5, :],
            label=f"({_symbol_1}, {_symbol_2})",
            alpha=0.7,
            color=_c,
        )

        remove_spines(plt.gca())
        if _i % 5 == 0:
            plt.ylabel(_symbol_1)

        if _i // 5 == 0:
            plt.title(_symbol_2)

        plt.xticks(rotation=45)
        plt.ylim([0, 0.005])

    plt.suptitle(f"Close Return Covariance EWMA (H={half_life})")
    plt.tight_layout()
    # mo.mpl.interactive(plt.gcf())
    plt.gcf()
    return (itertools,)


@app.cell
def _(ewma_covariance_estimate, np):
    def iterated_ewma(returns: np.ndarray, h_vol: float, h_corr: float) -> np.ndarray:
        N, T = returns.shape
        covs = ewma_covariance_estimate(returns, beta=0.5 ** (1 / h_vol))  # (N, N, T)
        std_devs = np.diagonal(covs, axis1=0, axis2=1) ** 0.5  # (T, N)
        std_devs = std_devs.T

        N, T = returns.shape
        returns_standardized = returns / std_devs  # (N, T)
        winsorized_returns = np.clip(returns_standardized, a_min=-4.2, a_max=4.2)

        R = ewma_covariance_estimate(winsorized_returns, 0.5 ** (1 / h_corr))  # (N, N, T)
        std_devs_R = np.diagonal(R, axis1=0, axis2=1) ** 0.5  # (T, N)
        std_devs_R = std_devs_R.T

        R_corr = R / std_devs_R[None, :, :] / std_devs_R[:, None, :]  # (N, N, T)
        cov_mat = std_devs[None, :, :] * R_corr * std_devs[:, None, :]

        return cov_mat
    return (iterated_ewma,)


@app.cell
def _(half_life, itertools, np, plt, remove_spines, symbol_to_color, symbols):
    def plot_covariance_matrices(cov_mat: np.ndarray, dates: np.ndarray):
        plt.figure(figsize=(10, 10))

        for _i, (_symbol_1, _symbol_2) in enumerate(itertools.product(symbols, symbols)):
            plt.subplot(5, 5, _i + 1)
            is_off_diag = _symbol_1 != _symbol_2
            _c = "k" if is_off_diag else symbol_to_color[_symbol_1]

            x, y = divmod(_i, 5)
            if x > y:
                continue

            std_dev_1 = cov_mat[x, x, :] ** 0.5
            std_dev_2 = cov_mat[y, y, :] ** 0.5
            _ts = cov_mat[x, y, :] / (std_dev_1 * std_dev_2) if is_off_diag else cov_mat[x, y, :] ** 0.5
            plt.plot(
                dates,
                _ts,
                label=f"({_symbol_1}, {_symbol_2})",
                alpha=0.7,
                color=_c,
            )

            remove_spines(plt.gca())
            if y == 0:
                plt.ylabel(_symbol_1)

            if x == 0:
                plt.title(_symbol_2)

            plt.xticks(rotation=45)
            if is_off_diag:
                plt.ylim([0, 1])
            else:
                plt.ylim([0, 0.06])

        plt.suptitle(f"Close Return Correlation EWMA (H={half_life})")
        plt.tight_layout()
        return plt.gcf()
    return (plot_covariance_matrices,)


@app.cell
def _(iterated_ewma, plot_covariance_matrices, returns_arr, returns_wide_df):
    _cov_mat = iterated_ewma(returns_arr, h_vol=20, h_corr=60)
    _dates = returns_wide_df["Date"].to_numpy()

    plot_covariance_matrices(_cov_mat, _dates)
    return


@app.cell
def _(K, iterated_ewma, k, np, returns_arr):
    import cvxpy as cp


    def _optimal_precision_factor(cholesky_factors: list[np.ndarray], returns: np.ndarray):
        # Build cvxpy problem once (reuse expressions); variables will be re-used across restarts
        breakpoint()
        w_var = cp.Variable(K, nonneg=True)
        constraints = [cp.sum(w_var) == 1]

        # TODO caseyh: where to encode lower triangular bit? Nowhere maybe?
        _, look_back = returns.shape
        logdet_term = 0
        quadratic_term = 0
        for t in range(look_back):
            # Stack diagonals into N x K matrix for fast affine diag computation
            diag_mat = np.stack([np.diag(Lk[t]) for Lk in cholesky_factors], axis=1)  # shape (N, K)
            diagL_expr = diag_mat @ w_var  # (N,)
            logdet_term += cp.sum(cp.log(diagL_expr))

            L_expr = sum(w_var[k] * Lk[t] for Lk in cholesky_factors)  # N x N expression
            quadratic_term += cp.sum_squares(L_expr.T @ returns[:, t])

        objective = cp.Maximize(logdet_term - 0.5 * quadratic_term)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return L_expr.value
        # TODO: return the optimal L


    def solve_precision_mixture(returns: np.ndarray, cov_matrices: list[np.ndarray], look_back: int = 1):
        """
        Solve:
            minimize -n * sum_j log(diag(L)_j) + 0.5 * sum_i || L.T x_i ||^2
        subject to:
            L = sum_k w_k * L_k
            w >= 0, sum(w) = 1
            diag(L) >= eps

        Inputs:

        Returns:
          dict with keys: 'w', 'objective', 'status', 'L', 'diagL'
        """
        K = len(cov_matrices)
        print(cov_matrices[0].shape)

        n_stocks, _, T = cov_matrices[0].shape
        cov_matrices = [cov.transpose(2, 0, 1) for cov in cov_matrices]  # (T, N, N)
        cholesky_factors = [np.linalg.cholesky(cov) for cov in cov_matrices]  # K arrays of shape (T, N, N)

        precision_matrices = []
        for t in range(1, T):
            lower = max(0, t - look_back)
            # TODO caseyh: how do we index this problem? Should we skip first look_back covariance matrices because they aren't PSD?
            L_t = _optimal_precision_factor([Lk[lower:t] for Lk in cholesky_factors], returns[:, lower:t])
            precision_matrices.append(L_t @ L_t.T)

        return precision_matrices


    _h_vols = [20]
    _h_corrs = [60]
    cov_matrices = [iterated_ewma(returns_arr, h_vol=h_vol, h_corr=h_corr) for h_vol, h_corr in zip(_h_vols, _h_corrs)]
    solve_precision_mixture(returns_arr, cov_matrices, look_back=1)
    # Do multiple restarts: set variable.value as initial guess and warm_start
    # for r in range(max(1, n_restarts)):
    #     # create a feasible random initial w (simplex)
    #     rand_w = np.random.rand(K)
    #     rand_w = rand_w / rand_w.sum()

    #     # warm-start by assigning a value
    #     w_var.value = rand_w

    #     try:
    #         problem.solve(solver=solver, verbose=verbose, warm_start=True)
    #     except Exception as e:
    #         # Some solvers may error on warm start or non-DCP; try without warm_start
    #         try:
    #             problem.solve(solver=solver, verbose=verbose)
    #         except Exception as e2:
    #             # skip this restart if solver fails
    #             print(f"Solver failed on restart {r}: {e2}")
    #             continue

    #     if w_var.value is None:
    #         continue

    #     obj_val = problem.value
    #     if obj_val is None:
    #         continue

    #     if obj_val < best["obj"]:
    #         # compute L numeric and store
    #         w_opt = np.array(w_var.value).reshape(-1)
    #         L_opt = sum(w_opt[k] * L_list[k] for k in range(K))
    #         best.update({"obj": obj_val, "w": w_opt, "status": problem.status, "L": L_opt, "diagL": np.diag(L_opt)})

    # return best
    return (cov_matrices,)


@app.cell
def _(cov_matrices, np):
    np.linalg.eigh(cov_matrices[0][:, :, 10])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

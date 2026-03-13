import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import least_squares
import yfinance as yf
import matplotlib.dates as mdates
import time


def lppl(t, A, B, C, tc, m, omega, phi):
    return A + B * (tc - t) ** m + C * (tc - t) ** m * np.cos(omega * np.log(tc - t) + phi)


def normalize_ohlc_dataframe(df):
    if df.empty:
        return df

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for col in df.columns:
            if isinstance(col, tuple):
                if col[0] in ("Date", "Datetime"):
                    flattened.append(col[0])
                else:
                    flattened.append(col[0] if col[0] else col[1])
            else:
                flattened.append(col)
        df.columns = flattened

    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"Downloaded data missing required columns. Columns: {list(df.columns)}")

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    return df


def fetch_sp500_data(end_date, retries=3, pause_seconds=1.5):
    ticker = "^GSPC"
    start_date = "2000-01-01"
    end_exclusive = end_date + pd.Timedelta(days=1)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_exclusive,
                progress=False,
                auto_adjust=False,
            )
            if not raw.empty:
                return normalize_ohlc_dataframe(raw)
            last_error = RuntimeError("yf.download returned an empty DataFrame.")
        except Exception as exc:
            last_error = exc
        time.sleep(pause_seconds * attempt)

    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(ticker).history(
                start=start_date,
                end=end_exclusive,
                auto_adjust=False,
            )
            if not raw.empty:
                return normalize_ohlc_dataframe(raw)
            last_error = RuntimeError("Ticker.history returned an empty DataFrame.")
        except Exception as exc:
            last_error = exc
        time.sleep(pause_seconds * attempt)

    raise RuntimeError(f"No data returned from yfinance after retries. Last error: {last_error}")


def fit_lppl_smart(t, price, previous_params=None):
    n = len(t)
    price_min = float(np.min(price))
    price_max = float(np.max(price))
    price_range = max(price_max - price_min, 1.0)

    lower = np.array(
        [
            price_min * 0.5,   # A
            -20 * price_range,  # B
            -20 * price_range,  # C
            n + 1,              # tc
            0.1,                # m
            4.0,                # omega
            -np.pi,             # phi
        ],
        dtype=float,
    )
    upper = np.array(
        [
            price_max * 1.5,   # A
            20 * price_range,  # B
            20 * price_range,  # C
            n * 2.5,           # tc
            0.99,              # m
            15.0,              # omega
            np.pi,             # phi
        ],
        dtype=float,
    )

    def residuals(params):
        tc = params[3]
        if tc <= np.max(t):
            return np.full_like(price, 1e9, dtype=float)
        model = lppl(t, *params)
        if not np.all(np.isfinite(model)):
            return np.full_like(price, 1e9, dtype=float)
        return model - price

    starts = []
    starts.append(
        np.array(
            [
                float(np.mean(price)),
                -price_range,
                0.1 * price_range,
                n * 1.2,
                0.5,
                8.0,
                0.0,
            ],
            dtype=float,
        )
    )

    if previous_params is not None:
        starts.append(np.clip(np.asarray(previous_params, dtype=float), lower, upper))

    rng = np.random.default_rng(7)
    for _ in range(12):
        starts.append(
            np.array(
                [
                    rng.uniform(price_min, price_max),
                    rng.uniform(-10 * price_range, 0),
                    rng.uniform(-2 * price_range, 2 * price_range),
                    rng.uniform(n * 1.05, n * 2.0),
                    rng.uniform(0.2, 0.9),
                    rng.uniform(5.0, 13.0),
                    rng.uniform(-np.pi, np.pi),
                ],
                dtype=float,
            )
        )

    best_result = None
    best_score = np.inf
    robust_scale = max(float(np.std(price)) * 0.15, 1.0)

    for start in starts:
        x0 = np.clip(start, lower, upper)
        result = least_squares(
            residuals,
            x0=x0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=robust_scale,
            max_nfev=20000,
        )
        if not result.success:
            continue
        rmse = float(np.sqrt(np.mean(result.fun ** 2)))
        if rmse < best_score:
            best_score = rmse
            best_result = result

    if best_result is None:
        raise RuntimeError("LPPL optimization failed for all starts.")

    return best_result.x


def main():
    end_date = pd.Timestamp.today().normalize()
    df = fetch_sp500_data(end_date)
    df = df[df["Date"] <= end_date].copy().reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No usable price data available after cleaning.")

    cluster_threshold_days = 91
    risk_window_days = 91

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)

    ax.plot([], [], color="black", label="S&P 500")
    ax.plot([], [], "--", color="red", label="LPPL Fit")
    ax.scatter([], [], color="blue", label="Positive Inflection", zorder=5)
    ax.scatter([], [], color="orange", label="Negative Inflection", zorder=5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("S&P 500 LPPL Fit with Adjustable Window")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider_window = Slider(ax_slider, "Weeks Window", 100, 500, valinit=400, valstep=10)
    previous_params = None

    def update(_val):
        nonlocal previous_params
        weeks_window = int(slider_window.val)
        start_date = end_date - pd.Timedelta(weeks=weeks_window)
        df_window = df[df["Date"] >= start_date].copy().reset_index(drop=True)

        df_window["t"] = np.arange(len(df_window))
        t = df_window["t"].values.flatten()
        price = df_window["Close"].values.flatten()

        try:
            params = fit_lppl_smart(t, price, previous_params=previous_params)
            previous_params = params
            fitted = lppl(t, *params)

            second_derivative = np.gradient(np.gradient(fitted))
            signs = np.sign(second_derivative)
            pos_idx = np.where((signs[:-1] < 0) & (signs[1:] > 0))[0]
            neg_idx = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0]
            pos_dates = df_window.loc[pos_idx, "Date"]
            neg_dates = df_window.loc[neg_idx, "Date"]

            all_inflections = np.sort(np.concatenate([pos_dates.values, neg_dates.values]))
            risk_windows = []
            if len(all_inflections) > 0:
                current_cluster = [all_inflections[0]]
                for d in all_inflections[1:]:
                    delta = (d - current_cluster[-1]).astype("timedelta64[D]").astype(int)
                    if delta <= cluster_threshold_days:
                        current_cluster.append(d)
                    else:
                        risk_windows.append(current_cluster)
                        current_cluster = [d]
                risk_windows.append(current_cluster)

            ax.cla()
            ax.plot(df_window["Date"], price, color="black", label="S&P 500")
            ax.plot(df_window["Date"], fitted, "--", color="red", label="LPPL Fit")
            ax.scatter(
                pos_dates,
                df_window.loc[pos_idx, "Close"],
                color="blue",
                label="Positive Inflection",
                zorder=5,
            )
            ax.scatter(
                neg_dates,
                df_window.loc[neg_idx, "Close"],
                color="orange",
                label="Negative Inflection",
                zorder=5,
            )

            for cluster in risk_windows:
                if len(cluster) < 2:
                    continue
                start = cluster[0]
                end = cluster[-1]
                window_width = (end - start).astype("timedelta64[D]").astype(int)
                if window_width < risk_window_days:
                    expand = (risk_window_days - window_width) // 2
                    start -= np.timedelta64(expand, "D")
                    end += np.timedelta64(expand, "D")
                ax.axvspan(start, end, color="red", alpha=0.25)

            grid = pd.date_range(
                start=df_window["Date"].min(),
                end=df_window["Date"].max(),
                freq="3MS",
            )
            for g in grid:
                ax.axvline(g, color="gray", linestyle=":", alpha=0.5)

            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title(f"S&P 500 LPPL Fit (last {weeks_window} weeks ending {end_date.date()})")
            ax.legend()
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.xticks(rotation=45)
            fig.canvas.draw_idle()

        except Exception as e:
            print("Fit failed:", e)

    slider_window.on_changed(update)
    update(400)
    plt.show()


if __name__ == "__main__":
    main()

"""Interactive Prosperity backtest dashboard.

This Streamlit app parses Prosperity backtest logs with the standard sections:

* Sandbox logs
* Activities log
* Trade History

The dashboard focuses on rapid debugging of spread, z-score, PnL, and fill
quality. Exact limit-order fill rate requires order submission prints in
``lambdaLog``. When those prints are unavailable, the app shows a conservative
timestamp-level fill proxy from own trades.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import StringIO
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover - exercised only without dashboard deps.
    px = None
    go = None

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised only without dashboard deps.
    st = None


def cache_data(*args, **kwargs):
    """Use Streamlit cache when available, otherwise return a no-op decorator."""

    if st is not None:
        return st.cache_data(*args, **kwargs)

    def decorator(func):
        return func

    return decorator


LOG_DIR = Path("backtests")
PRODUCT_COLUMN = "product"
SUBMISSION = "SUBMISSION"
DEFAULT_TICKS_PER_DAY = 1_000_000
DEFAULT_DAYS_PER_YEAR = 365
SMILE_FIT_METHODS = ["quadratic", "cubic", "rolling mean per strike"]


@dataclass(frozen=True)
class ParsedBacktestLog:
    """Container for parsed Prosperity backtest log sections."""

    path: Path
    activities: pd.DataFrame
    trades: pd.DataFrame
    sandbox: pd.DataFrame
    debug_lines: pd.DataFrame
    order_intents: pd.DataFrame
    indicators: pd.DataFrame


def _section_bounds(text: str) -> tuple[str, str, str]:
    """Split a Prosperity log into sandbox, activities, and trade sections."""

    activities_marker = "Activities log:"
    trades_marker = "Trade History:"

    activities_start = text.find(activities_marker)
    trades_start = text.find(trades_marker)

    if activities_start == -1:
        return text, "", ""

    sandbox_text = text[:activities_start]
    if trades_start == -1:
        activities_text = text[activities_start + len(activities_marker) :].strip()
        trades_text = ""
    else:
        activities_text = text[activities_start + len(activities_marker) : trades_start].strip()
        trades_text = text[trades_start + len(trades_marker) :].strip()

    return sandbox_text, activities_text, trades_text


def _parse_sandbox_logs(sandbox_text: str) -> pd.DataFrame:
    """Parse concatenated JSON sandbox rows emitted by the local backtester."""

    payload = sandbox_text.replace("Sandbox logs:", "", 1).strip()
    if not payload:
        return pd.DataFrame(columns=["timestamp", "sandboxLog", "lambdaLog"])

    decoder = json.JSONDecoder()
    rows: list[dict[str, object]] = []
    idx = 0
    while idx < len(payload):
        while idx < len(payload) and payload[idx].isspace():
            idx += 1
        if idx >= len(payload):
            break
        try:
            obj, end = decoder.raw_decode(payload, idx)
        except json.JSONDecodeError:
            next_open = payload.find("{", idx + 1)
            if next_open == -1:
                break
            idx = next_open
            continue
        rows.append(obj)
        idx = end

    if not rows:
        return pd.DataFrame(columns=["timestamp", "sandboxLog", "lambdaLog"])

    frame = pd.DataFrame(rows)
    for column in ["sandboxLog", "lambdaLog"]:
        if column not in frame:
            frame[column] = ""
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").astype("Int64")
    return frame[["timestamp", "sandboxLog", "lambdaLog"]]


def _parse_activities(activities_text: str) -> pd.DataFrame:
    """Parse the semicolon-delimited Activities log section."""

    if not activities_text:
        return pd.DataFrame()

    frame = pd.read_csv(StringIO(activities_text), sep=";")
    numeric_columns = [column for column in frame.columns if column != PRODUCT_COLUMN]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["best_bid"] = frame["bid_price_1"]
    frame["best_ask"] = frame["ask_price_1"]
    frame["spread"] = frame["best_ask"] - frame["best_bid"]
    frame["quoted_bid_volume"] = frame[[c for c in frame.columns if c.startswith("bid_volume_")]].sum(
        axis=1,
        min_count=1,
    )
    frame["quoted_ask_volume"] = frame[[c for c in frame.columns if c.startswith("ask_volume_")]].sum(
        axis=1,
        min_count=1,
    )
    frame.loc[frame["mid_price"] <= 0, "mid_price"] = np.nan
    frame["mid_price"] = frame.groupby(["day", PRODUCT_COLUMN])["mid_price"].transform(
        lambda series: series.ffill().bfill()
    )
    return frame


def _parse_trades(trades_text: str) -> pd.DataFrame:
    """Parse the Trade History section, tolerating trailing commas."""

    columns = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]
    if not trades_text:
        return pd.DataFrame(columns=columns)

    cleaned = re.sub(r",(\s*[}\]])", r"\1", trades_text.strip())
    try:
        rows = json.loads(cleaned)
    except json.JSONDecodeError:
        rows = []

    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=columns + ["side", "signed_quantity", "notional", "is_own_trade"])

    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["is_own_trade"] = (frame["buyer"] == SUBMISSION) | (frame["seller"] == SUBMISSION)
    frame["side"] = np.select(
        [frame["buyer"] == SUBMISSION, frame["seller"] == SUBMISSION],
        ["BUY", "SELL"],
        default="MARKET",
    )
    frame["signed_quantity"] = np.select(
        [frame["buyer"] == SUBMISSION, frame["seller"] == SUBMISSION],
        [frame["quantity"], -frame["quantity"]],
        default=0,
    )
    frame["notional"] = frame["price"] * frame["quantity"]
    return frame


ORDER_PATTERNS = [
    re.compile(
        r"ORDER\s+product=(?P<product>[A-Z0-9_]+)\s+side=(?P<side>BUY|SELL)\s+"
        r"price=(?P<price>-?\d+(?:\.\d+)?)\s+qty=(?P<qty>-?\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<side>BUY|SELL)\s+(?P<qty>\d+)x\s+(?P<price>-?\d+(?:\.\d+)?)"
        r"(?:\s+(?P<product>[A-Z0-9_]+))?",
        re.IGNORECASE,
    ),
]


def _extract_debug_lines(sandbox: pd.DataFrame) -> pd.DataFrame:
    """Expand multi-line lambda logs into one row per timestamped debug line."""

    rows: list[dict[str, object]] = []
    if sandbox.empty or "lambdaLog" not in sandbox:
        return pd.DataFrame(columns=["timestamp", "line"])

    for row in sandbox.itertuples(index=False):
        timestamp = getattr(row, "timestamp")
        text = getattr(row, "lambdaLog", "") or ""
        for line in str(text).splitlines():
            line = line.strip()
            if line:
                rows.append({"timestamp": timestamp, "line": line})
    return pd.DataFrame(rows, columns=["timestamp", "line"])


def _infer_order_intents(debug_lines: pd.DataFrame, products: Iterable[str]) -> pd.DataFrame:
    """Infer order submissions from common debug print formats."""

    known_products = set(products)
    rows: list[dict[str, object]] = []
    if debug_lines.empty:
        return pd.DataFrame(columns=["timestamp", "product", "side", "price", "quantity", "line"])

    for row in debug_lines.itertuples(index=False):
        line = str(row.line)
        for pattern in ORDER_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            data = match.groupdict()
            product = (data.get("product") or "").upper()
            if not product:
                mentioned = [candidate for candidate in known_products if candidate in line]
                product = mentioned[0] if len(mentioned) == 1 else "UNKNOWN"
            rows.append(
                {
                    "timestamp": row.timestamp,
                    "product": product,
                    "side": data["side"].upper(),
                    "price": float(data["price"]),
                    "quantity": abs(int(float(data["qty"]))),
                    "line": line,
                }
            )
            break

    return pd.DataFrame(rows, columns=["timestamp", "product", "side", "price", "quantity", "line"])


def _flatten_numeric(prefix: str, value: Any) -> Iterable[tuple[str, float]]:
    """Yield numeric leaves from nested logger JSON."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_numeric(child_prefix, child)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _flatten_numeric(child_prefix, child)
    elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        yield prefix, float(value)


def _parse_logged_indicators(debug_lines: pd.DataFrame, products: Iterable[str]) -> pd.DataFrame:
    """Parse JSON debug prints into timestamped numeric indicator series.

    This supports the Frankfurt Hedgehogs-style logger where each lambdaLog line
    is a JSON object grouped by strategy module, e.g. ``{"OPTION": {"delta": ...}}``.
    """

    columns = ["timestamp", "group", "indicator", "product", "value", "source_line"]
    known_products = set(products)
    rows: list[dict[str, object]] = []
    if debug_lines.empty:
        return pd.DataFrame(columns=columns)

    for row in debug_lines.itertuples(index=False):
        line = str(row.line)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        timestamp = payload.get("GENERAL", {}).get("TIMESTAMP", row.timestamp)
        for group, group_payload in payload.items():
            for indicator, value in _flatten_numeric("", group_payload):
                if not indicator:
                    continue
                matched_products = [product for product in known_products if product in indicator]
                rows.append(
                    {
                        "timestamp": timestamp,
                        "group": group,
                        "indicator": indicator,
                        "product": matched_products[0] if len(matched_products) == 1 else "",
                        "value": value,
                        "source_line": line,
                    }
                )

    return pd.DataFrame(rows, columns=columns)


@cache_data(show_spinner=False)
def parse_backtest_log(path: str) -> ParsedBacktestLog:
    """Parse a backtest log path into dashboard-ready DataFrames."""

    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8")
    return parse_backtest_text(text, source_name=log_path.name)


@cache_data(show_spinner=False)
def parse_backtest_text(text: str, source_name: str = "uploaded.log") -> ParsedBacktestLog:
    """Parse raw backtest log text into dashboard-ready DataFrames."""

    log_path = Path(source_name)
    sandbox_text, activities_text, trades_text = _section_bounds(text)

    activities = _parse_activities(activities_text)
    trades = _parse_trades(trades_text)
    sandbox = _parse_sandbox_logs(sandbox_text)
    debug_lines = _extract_debug_lines(sandbox)
    products = activities[PRODUCT_COLUMN].dropna().unique().tolist() if not activities.empty else []
    order_intents = _infer_order_intents(debug_lines, products)
    indicators = _parse_logged_indicators(debug_lines, products)

    return ParsedBacktestLog(
        path=log_path,
        activities=activities,
        trades=trades,
        sandbox=sandbox,
        debug_lines=debug_lines,
        order_intents=order_intents,
        indicators=indicators,
    )


def add_rolling_z_scores(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add product-level rolling z-scores of mid-price deviations."""

    if frame.empty:
        return frame

    enriched = frame.sort_values([PRODUCT_COLUMN, "timestamp"]).copy()
    grouped = enriched.groupby(PRODUCT_COLUMN, group_keys=False)["mid_price"]
    rolling_mean = grouped.transform(lambda series: series.rolling(window, min_periods=max(5, window // 5)).mean())
    rolling_std = grouped.transform(lambda series: series.rolling(window, min_periods=max(5, window // 5)).std())
    enriched["rolling_mean"] = rolling_mean
    enriched["rolling_std"] = rolling_std.replace(0, np.nan)
    enriched["z_score"] = (enriched["mid_price"] - enriched["rolling_mean"]) / enriched["rolling_std"]
    return enriched


def build_pair_spreads(activities: pd.DataFrame, window: int) -> pd.DataFrame:
    """Build pairwise mid-price spread series for all products in the log."""

    if activities.empty:
        return pd.DataFrame(columns=["timestamp", "pair", "spread", "z_score"])

    pivot = activities.pivot_table(index="timestamp", columns=PRODUCT_COLUMN, values="mid_price", aggfunc="last")
    rows: list[pd.DataFrame] = []
    for left, right in combinations(pivot.columns.dropna(), 2):
        spread = pivot[left] - pivot[right]
        rolling_mean = spread.rolling(window, min_periods=max(5, window // 5)).mean()
        rolling_std = spread.rolling(window, min_periods=max(5, window // 5)).std().replace(0, np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": spread.index,
                    "pair": f"{left} - {right}",
                    "spread": spread.values,
                    "z_score": ((spread - rolling_mean) / rolling_std).values,
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "pair", "spread", "z_score"])
    return pd.concat(rows, ignore_index=True)


def build_orderbook_levels(activities: pd.DataFrame, products: list[str], max_level: int = 3) -> pd.DataFrame:
    """Return long-form bid/ask levels for microstructure visualization."""

    rows: list[dict[str, object]] = []
    if activities.empty:
        return pd.DataFrame(columns=["timestamp", "product", "side", "level", "price", "volume"])

    subset = activities[activities[PRODUCT_COLUMN].isin(products)]
    for row in subset.itertuples(index=False):
        payload = row._asdict()
        for level in range(1, max_level + 1):
            for side, price_prefix, volume_prefix in [
                ("BID", "bid_price", "bid_volume"),
                ("ASK", "ask_price", "ask_volume"),
            ]:
                price = payload.get(f"{price_prefix}_{level}")
                volume = payload.get(f"{volume_prefix}_{level}")
                if pd.isna(price) or pd.isna(volume):
                    continue
                rows.append(
                    {
                        "timestamp": payload["timestamp"],
                        "product": payload[PRODUCT_COLUMN],
                        "side": side,
                        "level": level,
                        "price": float(price),
                        "volume": abs(float(volume)),
                    }
                )

    return pd.DataFrame(rows, columns=["timestamp", "product", "side", "level", "price", "volume"])


def downsample_by_timestamp(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Downsample dense time series by keeping evenly spaced timestamps."""

    if frame.empty or max_points <= 0:
        return frame
    timestamps = np.array(sorted(frame["timestamp"].dropna().unique()))
    if len(timestamps) <= max_points:
        return frame
    keep_idx = np.linspace(0, len(timestamps) - 1, max_points, dtype=int)
    keep = set(timestamps[keep_idx])
    return frame[frame["timestamp"].isin(keep)]


def build_normalized_mid_series(
    activities: pd.DataFrame,
    products: list[str],
    normalizer: str,
    indicators: pd.DataFrame,
) -> pd.DataFrame:
    """Build mid-price deviations versus a selected fair-value proxy."""

    subset = activities[activities[PRODUCT_COLUMN].isin(products)].copy()
    if subset.empty or normalizer == "none":
        return pd.DataFrame(columns=["timestamp", "product", "normalized_mid", "normalizer"])

    if normalizer == "rolling_mean":
        subset["normalizer"] = subset["rolling_mean"]
    else:
        if indicators.empty:
            return pd.DataFrame(columns=["timestamp", "product", "normalized_mid", "normalizer"])
        indicator_subset = indicators[indicators["label"] == normalizer][["timestamp", "value"]].copy()
        indicator_subset = indicator_subset.rename(columns={"value": "normalizer"})
        subset = subset.merge(indicator_subset, on="timestamp", how="left")

    subset["normalized_mid"] = subset["mid_price"] - subset["normalizer"]
    return subset[["timestamp", PRODUCT_COLUMN, "normalized_mid", "normalizer"]].rename(
        columns={PRODUCT_COLUMN: "product"}
    )


def prepare_indicator_labels(indicators: pd.DataFrame) -> pd.DataFrame:
    """Add a stable human-readable indicator label column."""

    if indicators.empty:
        return indicators.assign(label=pd.Series(dtype=str))
    output = indicators.copy()
    output["label"] = output["group"].astype(str) + "." + output["indicator"].astype(str)
    return output


def infer_option_chain(products: Iterable[str]) -> pd.DataFrame:
    """Infer voucher/option products, strikes, and likely underlyings from names."""

    product_list = sorted(str(product) for product in products if pd.notna(product))
    product_set = set(product_list)
    rows: list[dict[str, object]] = []
    for product in product_list:
        upper = product.upper()
        # Common Prosperity option name patterns: PRODUCT_VOUCHER_STRIKE, PRODUCT_OPTION_STRIKE,
        # or abbreviations like VEV_STRIKE (Velvetfruit Extract Voucher).
        is_option = any(term in upper for term in ["VOUCHER", "OPTION", "CALL", "PUT"])
        if not is_option:
            # Check for patterns like XXX_1234 or VEV_1234
            is_option = bool(re.search(r"_[0-9]+$", product))

        if not is_option:
            continue

        strike_match = re.search(r"(\d+(?:\.\d+)?)$", product)
        if not strike_match:
            continue

        strike = float(strike_match.group(1))
        prefix = re.sub(r"_?(?:VOUCHER|OPTION|CALL|PUT|\d+(?:\.\d+)?).*", "", product, flags=re.IGNORECASE).strip("_")
        underlying_candidates = [candidate for candidate in product_set if candidate != product and product.startswith(candidate)]
        underlying = prefix if prefix in product_set else ""
        if not underlying and underlying_candidates:
            underlying = max(underlying_candidates, key=len)
        
        # Heuristic for Round 3: VEV -> VELVETFRUIT_EXTRACT
        if not underlying and prefix == "VEV" and "VELVETFRUIT_EXTRACT" in product_set:
            underlying = "VELVETFRUIT_EXTRACT"

        rows.append({"product": product, "strike": strike, "underlying": underlying})

    return pd.DataFrame(rows, columns=["product", "strike", "underlying"])


def infer_underlying_product(products: Iterable[str], option_products: Iterable[str] | None = None) -> str | None:
    """Return the best default underlying for a detected option chain."""

    product_list = sorted(str(product) for product in products if pd.notna(product))
    option_set = set(option_products or [])
    chain = infer_option_chain(product_list)
    if option_set:
        chain = chain[chain["product"].isin(option_set)]
    candidates = [candidate for candidate in chain["underlying"].dropna().unique() if candidate in product_list]
    if candidates:
        return sorted(candidates, key=lambda value: (-len(value), value))[0]

    non_options = [product for product in product_list if product not in set(chain["product"])]
    return non_options[0] if non_options else None


def default_option_expiry_day(activities: pd.DataFrame) -> int:
    """Choose a conservative default expiry day for voucher analytics controls."""

    if activities.empty or "day" not in activities:
        return 7
    max_day = pd.to_numeric(activities["day"], errors="coerce").max()
    if not np.isfinite(max_day):
        return 7
    return max(7, int(np.floor(max_day)) + 1)


def _time_keys(frame: pd.DataFrame) -> list[str]:
    """Return timestamp keys present in a frame."""

    return ["day", "timestamp"] if "day" in frame.columns and frame["day"].notna().any() else ["timestamp"]


def _add_plot_time(frame: pd.DataFrame, ticks_per_day: int = DEFAULT_TICKS_PER_DAY) -> pd.DataFrame:
    """Add a monotonic timestamp for cross-day plots."""

    output = frame.copy()
    if "day" in output.columns:
        output["plot_time"] = output["day"].astype(float) * ticks_per_day + output["timestamp"].astype(float)
    else:
        output["plot_time"] = output["timestamp"].astype(float)
    return output


def _calculate_tte_years(
    frame: pd.DataFrame,
    expiry_day: float,
    ticks_per_day: int = DEFAULT_TICKS_PER_DAY,
    days_per_year: int = DEFAULT_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate option time-to-expiry in years from Prosperity day/timestamp fields."""

    day = pd.to_numeric(frame["day"], errors="coerce") if "day" in frame.columns else 0.0
    timestamp = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0)
    current_day = day + timestamp / ticks_per_day
    days_to_expiry = expiry_day - current_day
    tte = days_to_expiry / days_per_year
    return tte.where(tte > 0)


def _normal_pdf(x: np.ndarray | pd.Series | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return np.exp(-0.5 * values * values) / np.sqrt(2 * np.pi)


def _normal_cdf(x: np.ndarray | pd.Series | float) -> np.ndarray:
    """Vectorized normal CDF approximation without a SciPy dependency."""

    values = np.asarray(x, dtype=float)
    z = np.abs(values)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    cdf = 1.0 - _normal_pdf(z) * poly
    return np.where(values >= 0, cdf, 1.0 - cdf)


def _bs_d1_d2(
    spot: np.ndarray | pd.Series | float,
    strike: np.ndarray | pd.Series | float,
    tte: np.ndarray | pd.Series | float,
    volatility: np.ndarray | pd.Series | float,
) -> tuple[np.ndarray, np.ndarray]:
    spot_values = np.asarray(spot, dtype=float)
    strike_values = np.asarray(strike, dtype=float)
    tte_values = np.asarray(tte, dtype=float)
    vol_values = np.asarray(volatility, dtype=float)
    denominator = vol_values * np.sqrt(tte_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot_values / strike_values) + 0.5 * vol_values * vol_values * tte_values) / denominator
    d2 = d1 - denominator
    return d1, d2


def black_scholes_call_price(
    spot: np.ndarray | pd.Series | float,
    strike: np.ndarray | pd.Series | float,
    tte: np.ndarray | pd.Series | float,
    volatility: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Return zero-rate Black-Scholes call prices."""

    spot_values = np.asarray(spot, dtype=float)
    strike_values = np.asarray(strike, dtype=float)
    tte_values = np.asarray(tte, dtype=float)
    vol_values = np.asarray(volatility, dtype=float)
    intrinsic = np.maximum(spot_values - strike_values, 0.0)
    valid = (spot_values > 0) & (strike_values > 0) & (tte_values > 0) & (vol_values > 0)
    prices = np.full(np.broadcast(spot_values, strike_values, tte_values, vol_values).shape, np.nan)
    if not np.any(valid):
        return np.where((tte_values <= 0) | (vol_values <= 0), intrinsic, prices)

    d1, d2 = _bs_d1_d2(spot_values, strike_values, tte_values, vol_values)
    model = spot_values * _normal_cdf(d1) - strike_values * _normal_cdf(d2)
    prices = np.where(valid, model, prices)
    return np.where((tte_values <= 0) | (vol_values <= 0), intrinsic, prices)


def black_scholes_call_greeks(
    spot: np.ndarray | pd.Series | float,
    strike: np.ndarray | pd.Series | float,
    tte: np.ndarray | pd.Series | float,
    volatility: np.ndarray | pd.Series | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return delta, gamma, vega, annualized theta, and rho for zero-rate calls."""

    spot_values = np.asarray(spot, dtype=float)
    strike_values = np.asarray(strike, dtype=float)
    tte_values = np.asarray(tte, dtype=float)
    vol_values = np.asarray(volatility, dtype=float)
    valid = (spot_values > 0) & (strike_values > 0) & (tte_values > 0) & (vol_values > 0)
    d1, d2 = _bs_d1_d2(spot_values, strike_values, tte_values, vol_values)
    pdf = _normal_pdf(d1)
    sqrt_tte = np.sqrt(tte_values)

    delta = np.where(valid, _normal_cdf(d1), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(valid, pdf / (spot_values * vol_values * sqrt_tte), np.nan)
        vega = np.where(valid, spot_values * pdf * sqrt_tte, np.nan)
        theta = np.where(valid, -(spot_values * pdf * vol_values) / (2.0 * sqrt_tte), np.nan)
        # Rho for zero interest rate is technically T * K * N(d2) but Prosperity usually uses zero rate.
        # We include the zero-rate limit or standard formula with r=0.
        rho = np.where(valid, strike_values * tte_values * _normal_cdf(d2), np.nan)
    return delta, gamma, vega, theta, rho


def implied_volatility_call(
    market_price: np.ndarray | pd.Series,
    spot: np.ndarray | pd.Series,
    strike: np.ndarray | pd.Series,
    tte: np.ndarray | pd.Series,
    max_volatility: float = 5.0,
    iterations: int = 60,
) -> np.ndarray:
    """Solve zero-rate call IV with vectorized bisection."""

    prices = np.asarray(market_price, dtype=float)
    spot_values = np.asarray(spot, dtype=float)
    strike_values = np.asarray(strike, dtype=float)
    tte_values = np.asarray(tte, dtype=float)
    intrinsic = np.maximum(spot_values - strike_values, 0.0)
    valid = (
        np.isfinite(prices)
        & np.isfinite(spot_values)
        & np.isfinite(strike_values)
        & np.isfinite(tte_values)
        & (prices > 0)
        & (spot_values > 0)
        & (strike_values > 0)
        & (tte_values > 0)
        & (prices >= intrinsic - 1e-6)
        & (prices <= spot_values + 1e-6)
    )
    iv = np.full(prices.shape, np.nan)
    if not np.any(valid):
        return iv

    low = np.full(prices.shape, 1e-6)
    high = np.full(prices.shape, max_volatility)
    target = np.maximum(prices, intrinsic + 1e-9)
    for _ in range(iterations):
        mid = (low + high) / 2.0
        model = black_scholes_call_price(spot_values, strike_values, tte_values, mid)
        low = np.where(valid & (model < target), mid, low)
        high = np.where(valid & (model >= target), mid, high)
    iv[valid] = (low[valid] + high[valid]) / 2.0
    return iv


def build_options_analytics(
    activities: pd.DataFrame,
    option_products: list[str],
    underlying_product: str | None,
    expiry_day: float,
    ticks_per_day: int = DEFAULT_TICKS_PER_DAY,
    days_per_year: int = DEFAULT_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Build IV, moneyness, theoretical price, and Greek series for vouchers."""

    columns = [
        "day",
        "timestamp",
        "plot_time",
        "product",
        "strike",
        "strike_label",
        "underlying_product",
        "underlying_price",
        "market_price",
        "tte",
        "moneyness",
        "market_iv",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
    ]
    if activities.empty or not option_products or underlying_product is None:
        return pd.DataFrame(columns=columns)

    chain = infer_option_chain(activities[PRODUCT_COLUMN].dropna().unique())
    chain = chain[chain["product"].isin(option_products)]
    if chain.empty:
        return pd.DataFrame(columns=columns)

    time_keys = _time_keys(activities)
    option_rows = activities[activities[PRODUCT_COLUMN].isin(chain["product"])].copy()
    underlying_rows = activities[activities[PRODUCT_COLUMN].eq(underlying_product)].copy()
    if option_rows.empty or underlying_rows.empty:
        return pd.DataFrame(columns=columns)

    strike_by_product = chain.set_index("product")["strike"]
    option_rows["strike"] = option_rows[PRODUCT_COLUMN].map(strike_by_product)
    option_keep = time_keys + [PRODUCT_COLUMN, "strike", "mid_price"]
    option_rows = option_rows[option_keep].rename(
        columns={PRODUCT_COLUMN: "product", "mid_price": "market_price"}
    )
    underlying_rows = underlying_rows[time_keys + ["mid_price"]].rename(columns={"mid_price": "underlying_price"})

    merged = option_rows.merge(underlying_rows, on=time_keys, how="left")
    merged["underlying_product"] = underlying_product
    merged = _add_plot_time(merged, ticks_per_day)
    merged["tte"] = _calculate_tte_years(merged, expiry_day, ticks_per_day, days_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["moneyness"] = np.log(merged["strike"] / merged["underlying_price"]) / (np.sqrt(merged["tte"]).replace(0, np.nan))
    merged["market_iv"] = implied_volatility_call(
        merged["market_price"],
        merged["underlying_price"],
        merged["strike"],
        merged["tte"],
    )
    delta, gamma, vega, theta, rho = black_scholes_call_greeks(
        merged["underlying_price"],
        merged["strike"],
        merged["tte"],
        merged["market_iv"],
    )
    merged["delta"] = delta
    merged["gamma"] = gamma
    merged["vega"] = vega
    merged["theta"] = theta
    merged["rho"] = rho
    merged["strike_label"] = "K=" + merged["strike"].map(lambda value: f"{value:g}")
    for column in columns:
        if column not in merged:
            merged[column] = np.nan
    return merged[columns].sort_values(["product", "plot_time"])


def _fit_method_key(method: str) -> str:
    return method.strip().lower().replace(" ", "_")


def fit_volatility_surface(
    options: pd.DataFrame,
    method: str,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Add fitted IV, theoretical BS price, and residual columns."""

    output = options.copy()
    output["fitted_iv"] = np.nan
    method_key = _fit_method_key(method)
    valid = output["market_iv"].notna() & output["moneyness"].notna()

    if output.empty or not valid.any():
        output["theoretical_price"] = np.nan
        output["iv_residual"] = np.nan
        return output

    if method_key in {"quadratic", "cubic"}:
        degree = 2 if method_key == "quadratic" else 3
        group_keys = _time_keys(output)
        for _, group in output[valid].groupby(group_keys):
            x = group["moneyness"].to_numpy(dtype=float)
            y = group["market_iv"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            unique_x = np.unique(x[finite])
            if len(unique_x) >= degree + 1 and finite.sum() >= degree + 1:
                coefficients = np.polyfit(x[finite], y[finite], degree)
                output.loc[group.index[finite], "fitted_iv"] = np.polyval(coefficients, x[finite])
            elif finite.any():
                output.loc[group.index[finite], "fitted_iv"] = np.nanmean(y[finite])
    else:
        sorted_output = output.sort_values(["product", "plot_time"]).copy()
        sorted_output["fitted_iv"] = sorted_output.groupby("product")["market_iv"].transform(
            lambda series: series.rolling(rolling_window, min_periods=1).mean()
        )
        output.loc[sorted_output.index, "fitted_iv"] = sorted_output["fitted_iv"]

    output["fitted_iv"] = output["fitted_iv"].where(output["fitted_iv"] > 0)
    output["theoretical_price"] = black_scholes_call_price(
        output["underlying_price"],
        output["strike"],
        output["tte"],
        output["fitted_iv"],
    )
    output["iv_residual"] = output["market_iv"] - output["fitted_iv"]
    return output


def evenly_spaced_values(values: Iterable[float | int], count: int) -> list[float | int]:
    """Pick up to count evenly spaced values from a sorted sequence."""

    ordered = list(values)
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[index] for index in indices]


def build_position_series(activities: pd.DataFrame, trades: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """Estimate product positions and inventory PnL from own fills and mids."""

    columns = [
        "day",
        "timestamp",
        "plot_time",
        "product",
        "mid_price",
        "trade_flow",
        "position",
        "prev_position",
        "inventory_pnl",
    ]
    if activities.empty or not products:
        return pd.DataFrame(columns=columns)

    time_keys = _time_keys(activities)
    grid = activities[activities[PRODUCT_COLUMN].isin(products)][time_keys + [PRODUCT_COLUMN, "mid_price"]].copy()
    grid = grid.rename(columns={PRODUCT_COLUMN: "product"})
    if grid.empty:
        return pd.DataFrame(columns=columns)

    if trades.empty:
        flows = pd.DataFrame(columns=["timestamp", "product", "trade_flow"])
    else:
        own = trades[trades["is_own_trade"] & trades["symbol"].isin(products)].copy()
        flows = (
            own.groupby(["timestamp", "symbol"], as_index=False)["signed_quantity"].sum()
            if not own.empty
            else pd.DataFrame(columns=["timestamp", "symbol", "signed_quantity"])
        )
        flows = flows.rename(columns={"symbol": "product", "signed_quantity": "trade_flow"})

    grid = grid.merge(flows, on=["timestamp", "product"], how="left")
    grid["trade_flow"] = grid["trade_flow"].fillna(0.0)
    grid = _add_plot_time(grid).sort_values(["product", "plot_time"])
    grid["position"] = grid.groupby("product")["trade_flow"].cumsum()
    grid["prev_position"] = grid.groupby("product")["position"].shift(1).fillna(0.0)
    price_change = grid.groupby("product")["mid_price"].diff().fillna(0.0)
    grid["inventory_pnl"] = grid["prev_position"] * price_change
    for column in columns:
        if column not in grid:
            grid[column] = np.nan
    return grid[columns]


def calculate_spread_capture_pnl(
    activities: pd.DataFrame,
    trades: pd.DataFrame,
    products: list[str],
) -> pd.DataFrame:
    """Estimate fill edge against contemporaneous product mid."""

    columns = ["product", "spread_capture_pnl", "own_fill_volume", "avg_fill_edge"]
    if activities.empty or trades.empty or not products:
        return pd.DataFrame(columns=columns)

    own = trades[trades["is_own_trade"] & trades["symbol"].isin(products)].copy()
    if own.empty:
        return pd.DataFrame(columns=columns)

    mids = activities[["timestamp", PRODUCT_COLUMN, "mid_price"]].rename(columns={PRODUCT_COLUMN: "symbol"})
    own = own.merge(mids, on=["timestamp", "symbol"], how="left")
    own["fill_edge"] = np.select(
        [own["side"] == "BUY", own["side"] == "SELL"],
        [own["mid_price"] - own["price"], own["price"] - own["mid_price"]],
        default=0.0,
    )
    own["edge_pnl"] = own["fill_edge"] * own["quantity"]
    return (
        own.groupby("symbol", as_index=False)
        .agg(
            spread_capture_pnl=("edge_pnl", "sum"),
            own_fill_volume=("quantity", "sum"),
            avg_fill_edge=("fill_edge", "mean"),
        )
        .rename(columns={"symbol": "product"})
    )


def build_portfolio_greeks(
    activities: pd.DataFrame,
    options: pd.DataFrame,
    trades: pd.DataFrame,
    underlying_product: str | None,
) -> pd.DataFrame:
    """Aggregate option Greeks and underlying inventory into portfolio exposures."""

    columns = [
        "day",
        "timestamp",
        "plot_time",
        "option_delta",
        "underlying_position",
        "portfolio_delta",
        "portfolio_gamma",
        "portfolio_vega",
        "portfolio_theta",
        "portfolio_rho",
        "abs_portfolio_delta",
    ]
    if activities.empty or options.empty or underlying_product is None:
        return pd.DataFrame(columns=columns)

    option_products = sorted(options["product"].dropna().unique())
    products = option_products + [underlying_product]
    positions = build_position_series(activities, trades, products)
    if positions.empty:
        return pd.DataFrame(columns=columns)

    merge_keys = [key for key in [*_time_keys(options), "product"] if key in positions.columns]
    option_positions = options.merge(
        positions[merge_keys + ["position"]],
        on=merge_keys,
        how="left",
    )
    option_positions["position"] = option_positions["position"].fillna(0.0)
    option_positions["delta_exposure"] = option_positions["position"] * option_positions["delta"]
    option_positions["gamma_exposure"] = option_positions["position"] * option_positions["gamma"]
    option_positions["vega_exposure"] = option_positions["position"] * option_positions["vega"]
    option_positions["theta_exposure"] = option_positions["position"] * option_positions["theta"]
    option_positions["rho_exposure"] = option_positions["position"] * option_positions["rho"]

    group_keys = [key for key in [*_time_keys(option_positions), "plot_time"] if key in option_positions.columns]
    portfolio = (
        option_positions.groupby(group_keys, as_index=False)
        .agg(
            option_delta=("delta_exposure", "sum"),
            portfolio_gamma=("gamma_exposure", "sum"),
            portfolio_vega=("vega_exposure", "sum"),
            portfolio_theta=("theta_exposure", "sum"),
            portfolio_rho=("rho_exposure", "sum"),
        )
        .sort_values("plot_time")
    )

    underlying = positions[positions["product"].eq(underlying_product)][
        [key for key in ["day", "timestamp", "plot_time", "position"] if key in positions.columns]
    ].rename(columns={"position": "underlying_position"})
    merge_time_keys = [key for key in [*_time_keys(portfolio), "plot_time"] if key in underlying.columns]
    portfolio = portfolio.merge(underlying, on=merge_time_keys, how="left")
    portfolio["underlying_position"] = portfolio["underlying_position"].fillna(0.0)
    portfolio["portfolio_delta"] = portfolio["option_delta"] + portfolio["underlying_position"]
    portfolio["abs_portfolio_delta"] = portfolio["portfolio_delta"].abs()
    for column in columns:
        if column not in portfolio:
            portfolio[column] = np.nan
    return portfolio[columns]


def build_option_pnl_attribution(
    activities: pd.DataFrame,
    trades: pd.DataFrame,
    options: pd.DataFrame,
    underlying_product: str | None,
) -> pd.DataFrame:
    """Estimate option-specific PnL attribution by product and hedge leg."""

    columns = [
        "product",
        "role",
        "official_pnl",
        "inventory_pnl",
        "spread_capture_pnl",
        "hedge_pnl",
        "delta_pnl",
        "gamma_pnl",
        "vega_pnl",
        "theta_decay",
        "residual_pnl_est",
        "ending_position",
    ]
    if activities.empty or options.empty:
        return pd.DataFrame(columns=columns)

    option_products = sorted(options["product"].dropna().unique())
    all_products = option_products + ([underlying_product] if underlying_product else [])
    positions = build_position_series(activities, trades, all_products)
    spread = calculate_spread_capture_pnl(activities, trades, all_products)

    latest_pnl = (
        activities[activities[PRODUCT_COLUMN].isin(all_products)]
        .sort_values("timestamp")
        .groupby(PRODUCT_COLUMN, as_index=False)
        .tail(1)[[PRODUCT_COLUMN, "profit_and_loss"]]
        .rename(columns={PRODUCT_COLUMN: "product", "profit_and_loss": "official_pnl"})
    )
    official_by_product = (
        latest_pnl.set_index("product")["official_pnl"] if not latest_pnl.empty else pd.Series(dtype=float)
    )
    inventory_by_product = (
        positions.groupby("product")["inventory_pnl"].sum() if not positions.empty else pd.Series(dtype=float)
    )
    if positions.empty:
        ending_position_by_product = pd.Series(dtype=float)
    else:
        latest_positions = positions.sort_values("plot_time").groupby("product", as_index=False).tail(1)
        ending_position_by_product = latest_positions.set_index("product")["position"]
    spread_by_product = spread.set_index("product")["spread_capture_pnl"] if not spread.empty else pd.Series(dtype=float)

    merge_keys = [key for key in [*_time_keys(options), "product"] if key in positions.columns]
    attr_frame = options.merge(positions[merge_keys + ["position"]], on=merge_keys, how="left")
    attr_frame["position"] = attr_frame["position"].fillna(0.0)
    attr_frame = attr_frame.sort_values(["product", "plot_time"])
    
    attr_frame["prev_position"] = attr_frame.groupby("product")["position"].shift(1).fillna(0.0)
    attr_frame["prev_delta"] = attr_frame.groupby("product")["delta"].shift(1)
    attr_frame["prev_gamma"] = attr_frame.groupby("product")["gamma"].shift(1)
    attr_frame["prev_vega"] = attr_frame.groupby("product")["vega"].shift(1)
    attr_frame["prev_theta"] = attr_frame.groupby("product")["theta"].shift(1)
    attr_frame["prev_tte"] = attr_frame.groupby("product")["tte"].shift(1)
    attr_frame["prev_spot"] = attr_frame.groupby("product")["underlying_price"].shift(1)
    attr_frame["prev_iv"] = attr_frame.groupby("product")["market_iv"].shift(1)
    
    delta_s = attr_frame["underlying_price"] - attr_frame["prev_spot"]
    delta_iv = attr_frame["market_iv"] - attr_frame["prev_iv"]
    elapsed_years = (attr_frame["prev_tte"] - attr_frame["tte"]).clip(lower=0)
    
    attr_frame["delta_pnl"] = attr_frame["prev_position"] * attr_frame["prev_delta"] * delta_s
    attr_frame["gamma_pnl"] = 0.5 * attr_frame["prev_position"] * attr_frame["prev_gamma"] * (delta_s ** 2)
    attr_frame["vega_pnl"] = attr_frame["prev_position"] * attr_frame["prev_vega"] * delta_iv
    attr_frame["theta_decay"] = attr_frame["prev_position"] * attr_frame["prev_theta"] * elapsed_years
    
    sums = attr_frame.groupby("product")[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_decay"]].sum(min_count=1)

    rows: list[dict[str, object]] = []
    for product in option_products:
        inventory_pnl = float(inventory_by_product.get(product, 0.0))
        spread_capture_pnl = float(spread_by_product.get(product, 0.0))
        
        row_sums = sums.loc[product] if product in sums.index else pd.Series(0.0, index=sums.columns)
        delta_pnl = float(row_sums["delta_pnl"]) if pd.notna(row_sums["delta_pnl"]) else 0.0
        gamma_pnl = float(row_sums["gamma_pnl"]) if pd.notna(row_sums["gamma_pnl"]) else 0.0
        vega_pnl = float(row_sums["vega_pnl"]) if pd.notna(row_sums["vega_pnl"]) else 0.0
        theta_decay = float(row_sums["theta_decay"]) if pd.notna(row_sums["theta_decay"]) else 0.0
        
        official_pnl = float(official_by_product.get(product, np.nan))
        # Note: inventory_pnl for options already includes the mid-price change.
        # Greeks attribution is an alternative decomposition of inventory_pnl.
        # We show them alongside each other.
        component_sum = delta_pnl + gamma_pnl + vega_pnl + theta_decay + spread_capture_pnl
        rows.append(
            {
                "product": product,
                "role": "option",
                "official_pnl": official_pnl,
                "inventory_pnl": inventory_pnl,
                "spread_capture_pnl": spread_capture_pnl,
                "hedge_pnl": 0.0,
                "delta_pnl": delta_pnl,
                "gamma_pnl": gamma_pnl,
                "vega_pnl": vega_pnl,
                "theta_decay": theta_decay,
                "residual_pnl_est": official_pnl - component_sum if np.isfinite(official_pnl) else np.nan,
                "ending_position": float(ending_position_by_product.get(product, 0.0)),
            }
        )

    if underlying_product:
        hedge_pnl = float(inventory_by_product.get(underlying_product, 0.0))
        spread_capture_pnl = float(spread_by_product.get(underlying_product, 0.0))
        official_pnl = float(official_by_product.get(underlying_product, np.nan))
        rows.append(
            {
                "product": underlying_product,
                "role": "underlying hedge",
                "official_pnl": official_pnl,
                "inventory_pnl": 0.0,
                "spread_capture_pnl": spread_capture_pnl,
                "hedge_pnl": hedge_pnl,
                "delta_pnl": 0.0,
                "gamma_pnl": 0.0,
                "vega_pnl": 0.0,
                "theta_decay": 0.0,
                "residual_pnl_est": official_pnl - spread_capture_pnl - hedge_pnl if np.isfinite(official_pnl) else np.nan,
                "ending_position": float(ending_position_by_product.get(underlying_product, 0.0)),
            }
        )

    output = pd.DataFrame(rows, columns=columns)
    if output.empty:
        return output
    total = {
        "product": "TOTAL_OPTION_PORTFOLIO",
        "role": "portfolio",
        "official_pnl": output["official_pnl"].sum(min_count=1),
        "inventory_pnl": output["inventory_pnl"].sum(),
        "spread_capture_pnl": output["spread_capture_pnl"].sum(),
        "hedge_pnl": output["hedge_pnl"].sum(),
        "delta_pnl": output["delta_pnl"].sum(),
        "gamma_pnl": output["gamma_pnl"].sum(),
        "vega_pnl": output["vega_pnl"].sum(),
        "theta_decay": output["theta_decay"].sum(),
        "ending_position": np.nan,
    }
    total_component_sum = (
        total["delta_pnl"] + total["gamma_pnl"] + total["vega_pnl"] + total["theta_decay"] + total["spread_capture_pnl"] + total["hedge_pnl"]
    )
    total["residual_pnl_est"] = (
        total["official_pnl"] - total_component_sum if np.isfinite(total["official_pnl"]) else np.nan
    )
    return pd.concat([output, pd.DataFrame([total])], ignore_index=True)[columns]


def decompose_pnl(activities: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Estimate per-product PnL decomposition.

    The activity log supplies official mark-to-market PnL by product. Trade
    history does not tag maker vs taker fills, so market-making PnL is estimated
    as fill edge versus contemporaneous mid. Directional PnL is the residual.
    """

    columns = [
        "product",
        "official_pnl",
        "market_making_pnl_est",
        "directional_pnl_est",
        "own_fill_volume",
        "buy_volume",
        "sell_volume",
        "avg_fill_edge",
    ]
    if activities.empty:
        return pd.DataFrame(columns=columns)

    latest_pnl = (
        activities.sort_values("timestamp")
        .groupby(PRODUCT_COLUMN, as_index=False)
        .tail(1)[[PRODUCT_COLUMN, "profit_and_loss"]]
        .rename(columns={PRODUCT_COLUMN: "product", "profit_and_loss": "official_pnl"})
    )

    own = trades[trades["is_own_trade"]].copy() if not trades.empty else pd.DataFrame()
    if own.empty:
        latest_pnl["market_making_pnl_est"] = 0.0
        latest_pnl["directional_pnl_est"] = latest_pnl["official_pnl"]
        latest_pnl["own_fill_volume"] = 0
        latest_pnl["buy_volume"] = 0
        latest_pnl["sell_volume"] = 0
        latest_pnl["avg_fill_edge"] = np.nan
        return latest_pnl[columns]

    mids = activities[["timestamp", PRODUCT_COLUMN, "mid_price"]].rename(columns={PRODUCT_COLUMN: "symbol"})
    own = own.merge(mids, on=["timestamp", "symbol"], how="left")
    own["fill_edge"] = np.select(
        [own["side"] == "BUY", own["side"] == "SELL"],
        [own["mid_price"] - own["price"], own["price"] - own["mid_price"]],
        default=0.0,
    )
    own["edge_pnl"] = own["fill_edge"] * own["quantity"]

    summary = (
        own.groupby("symbol", as_index=False)
        .agg(
            market_making_pnl_est=("edge_pnl", "sum"),
            own_fill_volume=("quantity", "sum"),
            buy_volume=("quantity", lambda values: values[own.loc[values.index, "side"].eq("BUY")].sum()),
            sell_volume=("quantity", lambda values: values[own.loc[values.index, "side"].eq("SELL")].sum()),
            avg_fill_edge=("fill_edge", "mean"),
        )
        .rename(columns={"symbol": "product"})
    )

    output = latest_pnl.merge(summary, on="product", how="left").fillna(
        {
            "market_making_pnl_est": 0.0,
            "own_fill_volume": 0,
            "buy_volume": 0,
            "sell_volume": 0,
        }
    )
    output["directional_pnl_est"] = output["official_pnl"] - output["market_making_pnl_est"]
    return output[columns].sort_values("official_pnl", ascending=False)


def build_fill_report(parsed: ParsedBacktestLog, product_filter: list[str]) -> pd.DataFrame:
    """Build exact or proxy fill-rate diagnostics."""

    trades = parsed.trades
    activities = parsed.activities
    intents = parsed.order_intents
    own_trades = trades[trades["is_own_trade"]].copy() if not trades.empty else pd.DataFrame()

    if not intents.empty and not own_trades.empty:
        intents = intents[intents["product"].isin(product_filter) | intents["product"].eq("UNKNOWN")].copy()
        fills = own_trades.groupby(["timestamp", "symbol", "side"], as_index=False).agg(
            filled_quantity=("quantity", "sum"),
            fills=("quantity", "size"),
        )
        exact = intents.merge(
            fills,
            left_on=["timestamp", "product", "side"],
            right_on=["timestamp", "symbol", "side"],
            how="left",
        )
        exact["filled_quantity"] = exact["filled_quantity"].fillna(0)
        exact["fill_ratio"] = (exact["filled_quantity"] / exact["quantity"]).clip(upper=1.0)
        exact["filled"] = exact["filled_quantity"] > 0
        return (
            exact.groupby("product", as_index=False)
            .agg(
                orders_logged=("quantity", "size"),
                filled_orders=("filled", "sum"),
                submitted_qty=("quantity", "sum"),
                filled_qty=("filled_quantity", "sum"),
                avg_order_fill_ratio=("fill_ratio", "mean"),
            )
            .assign(fill_rate_mode="debug_order_logs")
        )

    if activities.empty:
        return pd.DataFrame()

    active = activities[activities[PRODUCT_COLUMN].isin(product_filter)]
    active_counts = active.groupby(PRODUCT_COLUMN, as_index=False).agg(active_timestamps=("timestamp", "nunique"))
    if own_trades.empty:
        active_counts["timestamps_with_fills"] = 0
        active_counts["fill_timestamp_rate"] = 0.0
        active_counts["fill_rate_mode"] = "timestamp_proxy_no_order_logs"
        return active_counts.rename(columns={PRODUCT_COLUMN: "product"})

    filled_counts = (
        own_trades[own_trades["symbol"].isin(product_filter)]
        .groupby("symbol", as_index=False)
        .agg(timestamps_with_fills=("timestamp", "nunique"), fill_count=("quantity", "size"))
        .rename(columns={"symbol": "product"})
    )
    report = active_counts.rename(columns={PRODUCT_COLUMN: "product"}).merge(filled_counts, on="product", how="left")
    report[["timestamps_with_fills", "fill_count"]] = report[["timestamps_with_fills", "fill_count"]].fillna(0)
    report["fill_timestamp_rate"] = report["timestamps_with_fills"] / report["active_timestamps"]
    report["fill_rate_mode"] = "timestamp_proxy_no_order_logs"
    return report


def plot_spreads(activities: pd.DataFrame, products: list[str]) -> go.Figure:
    """Create product bid-ask spread chart."""

    subset = activities[activities[PRODUCT_COLUMN].isin(products)]
    fig = px.line(
        subset,
        x="timestamp",
        y="spread",
        color=PRODUCT_COLUMN,
        title="Bid-Ask Spread By Product",
        labels={"spread": "L1 spread"},
    )
    fig.update_layout(legend_title_text="Product")
    return fig


def plot_z_scores(
    activities: pd.DataFrame,
    products: list[str],
    entry_threshold: float,
    exit_threshold: float,
) -> go.Figure:
    """Create rolling z-score plot with threshold markers."""

    subset = activities[activities[PRODUCT_COLUMN].isin(products)]
    fig = px.line(
        subset,
        x="timestamp",
        y="z_score",
        color=PRODUCT_COLUMN,
        title="Rolling Mid-Price Z-Score",
    )
    for value, name, color in [
        (entry_threshold, "entry", "firebrick"),
        (-entry_threshold, "-entry", "firebrick"),
        (exit_threshold, "exit", "gray"),
        (-exit_threshold, "-exit", "gray"),
    ]:
        fig.add_hline(y=value, line_dash="dash", line_color=color, annotation_text=name)
    return fig


def plot_pair_spreads(pair_spreads: pd.DataFrame, selected_pairs: list[str]) -> go.Figure:
    """Create pair spread chart."""

    subset = pair_spreads[pair_spreads["pair"].isin(selected_pairs)]
    fig = px.line(subset, x="timestamp", y="spread", color="pair", title="Pair/Basket Spread Series")
    fig.update_layout(legend_title_text="Spread")
    return fig


def plot_orderbook(
    activities: pd.DataFrame,
    trades: pd.DataFrame,
    products: list[str],
    max_points: int,
) -> go.Figure:
    """Create a Prosperity-style order book level and trade scatter chart."""

    levels = build_orderbook_levels(downsample_by_timestamp(activities, max_points), products)
    fig = go.Figure()
    for side, color in [("BID", "royalblue"), ("ASK", "firebrick")]:
        side_levels = levels[levels["side"] == side]
        if side_levels.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=side_levels["timestamp"],
                y=side_levels["price"],
                mode="markers",
                marker={
                    "color": color,
                    "size": np.clip(side_levels["volume"], 4, 18),
                    "opacity": 0.45,
                },
                text=side_levels["product"] + " L" + side_levels["level"].astype(str),
                name=side,
            )
        )

    if not trades.empty:
        own = trades[trades["is_own_trade"] & trades["symbol"].isin(products)]
        market = trades[(~trades["is_own_trade"]) & trades["symbol"].isin(products)]
        for name, frame, symbol, color in [
            ("Market trades", market, "triangle-up", "gray"),
            ("Own fills", own, "x", "darkorange"),
        ]:
            if frame.empty:
                continue
            frame = downsample_by_timestamp(frame.rename(columns={"symbol": PRODUCT_COLUMN}), max_points)
            fig.add_trace(
                go.Scattergl(
                    x=frame["timestamp"],
                    y=frame["price"],
                    mode="markers",
                    marker={"symbol": symbol, "color": color, "size": 9},
                    text=frame[PRODUCT_COLUMN] + " " + frame["quantity"].astype(str) + "x",
                    name=name,
                )
            )

    fig.update_layout(
        title="Order Book Levels And Trades",
        xaxis_title="timestamp",
        yaxis_title="price",
        legend_title_text="Layer",
    )
    return fig


def plot_normalized_mid(normalized: pd.DataFrame) -> go.Figure:
    """Plot mid-price deviation from selected normalizer."""

    fig = px.line(
        normalized,
        x="timestamp",
        y="normalized_mid",
        color="product",
        title="Mid Price Normalized By Selected Fair-Value Proxy",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def plot_indicators(indicators: pd.DataFrame, labels: list[str]) -> go.Figure:
    """Plot selected numeric indicators extracted from JSON lambda logs."""

    subset = indicators[indicators["label"].isin(labels)]
    fig = px.line(
        subset,
        x="timestamp",
        y="value",
        color="label",
        title="Logged Numeric Indicators",
    )
    return fig


def plot_pnl(activities: pd.DataFrame, products: list[str]) -> go.Figure:
    """Create product and total PnL chart from the activity log."""

    subset = activities[activities[PRODUCT_COLUMN].isin(products)].copy()
    fig = px.line(
        subset,
        x="timestamp",
        y="profit_and_loss",
        color=PRODUCT_COLUMN,
        title="Official Mark-To-Market PnL",
    )
    
    # Add Total PnL line
    total_pnl = subset.groupby("timestamp")["profit_and_loss"].sum().reset_index()
    fig.add_trace(
        go.Scatter(
            x=total_pnl["timestamp"],
            y=total_pnl["profit_and_loss"],
            name="TOTAL",
            line={"color": "black", "width": 3},
        )
    )
    return fig


def plot_iv_time_series(options: pd.DataFrame) -> go.Figure:
    """Plot market IV over time with one line per strike."""

    subset = options.dropna(subset=["market_iv", "plot_time"])
    fig = px.line(
        subset,
        x="plot_time",
        y="market_iv",
        color="strike_label",
        line_dash="product",
        title="IV Per Strike Time Series",
        labels={"plot_time": "timestamp", "market_iv": "market IV", "strike_label": "Strike"},
        hover_data=["product", "strike", "underlying_price", "market_price", "moneyness"],
    )
    return fig


def plot_smile_snapshot(options: pd.DataFrame, plot_time: float | int, fit_method: str) -> go.Figure:
    """Plot one timestamp's smile with the selected fit overlaid."""

    snapshot = options[options["plot_time"].eq(plot_time)].dropna(subset=["market_iv", "moneyness"])
    fig = px.scatter(
        snapshot,
        x="moneyness",
        y="market_iv",
        color="strike_label",
        title="Vol Smile Snapshot",
        labels={"moneyness": "ln(K/S) / sqrt(TTE)", "market_iv": "market IV", "strike_label": "Strike"},
        hover_data=["product", "strike", "underlying_price", "market_price", "fitted_iv"],
    )

    fit_points = snapshot.dropna(subset=["fitted_iv", "moneyness"]).sort_values("moneyness")
    method_key = _fit_method_key(fit_method)
    if method_key in {"quadratic", "cubic"} and not snapshot.empty:
        degree = 2 if method_key == "quadratic" else 3
        x = snapshot["moneyness"].to_numpy(dtype=float)
        y = snapshot["market_iv"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if len(np.unique(x[finite])) >= degree + 1 and finite.sum() >= degree + 1:
            coefficients = np.polyfit(x[finite], y[finite], degree)
            x_grid = np.linspace(np.nanmin(x[finite]), np.nanmax(x[finite]), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=np.polyval(coefficients, x_grid),
                    mode="lines",
                    name=f"{fit_method} fit",
                    line={"color": "black", "width": 2},
                )
            )
    elif not fit_points.empty:
        fig.add_trace(
            go.Scatter(
                x=fit_points["moneyness"],
                y=fit_points["fitted_iv"],
                mode="lines+markers",
                name="rolling mean per strike",
                line={"color": "black", "width": 2},
            )
        )
    return fig


def plot_smile_drift(options: pd.DataFrame, plot_times: list[float | int]) -> go.Figure:
    """Overlay smiles from selected timestamps."""

    subset = options[options["plot_time"].isin(plot_times)].dropna(subset=["market_iv", "moneyness"]).copy()
    subset["timestamp_label"] = subset["plot_time"].map(lambda value: f"{value:g}")
    fig = px.line(
        subset.sort_values(["plot_time", "moneyness"]),
        x="moneyness",
        y="market_iv",
        color="timestamp_label",
        markers=True,
        title="Smile Drift Across Timestamps",
        labels={"moneyness": "ln(K/S) / sqrt(TTE)", "market_iv": "market IV", "timestamp_label": "Timestamp"},
        hover_data=["product", "strike", "underlying_price", "market_price"],
    )
    return fig


def plot_bs_price_scatter(options: pd.DataFrame) -> go.Figure:
    """Plot fitted BS theoretical price against market price."""

    subset = options.dropna(subset=["theoretical_price", "market_price", "moneyness"])
    fig = px.scatter(
        subset,
        x="theoretical_price",
        y="market_price",
        color="moneyness",
        symbol="product",
        title="Theoretical BS Price Vs Market Price",
        labels={
            "theoretical_price": "BS theoretical price",
            "market_price": "market price",
            "moneyness": "moneyness",
        },
        hover_data=["product", "strike", "market_iv", "fitted_iv"],
        color_continuous_scale="RdBu",
    )
    values = pd.concat([subset["theoretical_price"], subset["market_price"]]).replace([np.inf, -np.inf], np.nan).dropna()
    if not values.empty:
        low = float(values.min())
        high = float(values.max())
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="y=x",
                line={"color": "black", "dash": "dash"},
            )
        )
    return fig


def plot_iv_residuals(options: pd.DataFrame) -> go.Figure:
    """Plot residuals between market IV and fitted IV."""

    subset = options.dropna(subset=["iv_residual"])
    fig = px.histogram(
        subset,
        x="iv_residual",
        nbins=40,
        title="IV Residuals: Market IV Minus Fitted IV",
        labels={"iv_residual": "market IV - fitted IV"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    return fig


def plot_greek_time_series(options: pd.DataFrame, greek: str) -> go.Figure:
    """Plot one Greek time series per voucher."""

    subset = options.dropna(subset=[greek, "plot_time"])
    title = f"{greek.capitalize()} Time Series Per Voucher"
    fig = px.line(
        subset,
        x="plot_time",
        y=greek,
        color="product",
        title=title,
        labels={"plot_time": "timestamp", greek: greek},
        hover_data=["strike", "underlying_price", "market_price", "market_iv", "moneyness"],
    )
    return fig


def plot_portfolio_greeks(portfolio: pd.DataFrame) -> go.Figure:
    """Plot aggregated portfolio delta, gamma, vega, theta, and rho."""

    if portfolio.empty:
        return go.Figure()
    melted = portfolio.melt(
        id_vars=["plot_time"],
        value_vars=["portfolio_delta", "portfolio_gamma", "portfolio_vega", "portfolio_theta", "portfolio_rho"],
        var_name="greek",
        value_name="exposure",
    )
    fig = px.line(
        melted,
        x="plot_time",
        y="exposure",
        color="greek",
        title="Portfolio Greeks",
        labels={"plot_time": "timestamp", "exposure": "exposure"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def plot_hedge_error(portfolio: pd.DataFrame, threshold: float) -> go.Figure:
    """Plot absolute portfolio delta with rebalance threshold."""

    fig = px.line(
        portfolio,
        x="plot_time",
        y="abs_portfolio_delta",
        title="Hedge Error: Absolute Portfolio Delta",
        labels={"plot_time": "timestamp", "abs_portfolio_delta": "|portfolio delta|"},
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="firebrick", annotation_text="rebalance threshold")
    return fig


def plot_option_pnl_attribution(attribution: pd.DataFrame) -> go.Figure:
    """Plot options PnL attribution components by product."""

    components = [
        "delta_pnl",
        "gamma_pnl",
        "vega_pnl",
        "theta_decay",
        "spread_capture_pnl",
        "hedge_pnl",
        "residual_pnl_est",
    ]
    subset = attribution[attribution["product"].ne("TOTAL_OPTION_PORTFOLIO")].copy()
    melted = subset.melt(
        id_vars=["product", "role"],
        value_vars=[c for c in components if c in subset.columns],
        var_name="component",
        value_name="pnl",
    )
    fig = px.bar(
        melted,
        x="product",
        y="pnl",
        color="component",
        title="Options PnL Attribution",
        labels={"product": "product", "pnl": "PnL", "component": "component"},
        barmode="relative",
    )
    return fig


def render_debug_log(debug_lines: pd.DataFrame, timestamp: int | None, limit: int = 200) -> None:
    """Render timestamp-filtered custom lambda logs."""

    st.subheader("Custom Trading Prints")
    if debug_lines.empty:
        st.info("No custom lambdaLog lines found in this backtest.")
        return

    subset = debug_lines
    if timestamp is not None:
        subset = subset[subset["timestamp"] == timestamp]
    st.dataframe(subset.tail(limit), use_container_width=True, hide_index=True)


def main() -> None:
    """Streamlit entrypoint."""

    if st is None or px is None or go is None:
        raise SystemExit("Install dashboard dependencies first: pip install streamlit plotly pandas numpy")

    st.set_page_config(page_title="Prosperity Trading Dashboard", layout="wide")
    st.title("Prosperity Trading Dashboard")

    log_files = sorted(LOG_DIR.glob("*.log"))
    if not log_files:
        st.error("No backtest logs found in backtests/*.log")
        return

    with st.sidebar:
        st.header("Inputs")
        selected_path = st.selectbox("Backtest log", log_files, format_func=lambda path: path.name)
        z_window = st.slider("Rolling z-score window", min_value=10, max_value=500, value=100, step=10)
        entry_threshold = st.number_input("Entry z-score threshold", value=2.0, min_value=0.1, step=0.1)
        exit_threshold = st.number_input("Exit z-score threshold", value=0.5, min_value=0.0, step=0.1)
        max_plot_points = st.slider("Max plotted timestamps", min_value=500, max_value=20000, value=5000, step=500)

    parsed = parse_backtest_log(str(selected_path))
    activities = add_rolling_z_scores(parsed.activities, z_window)
    trades = parsed.trades
    indicators = prepare_indicator_labels(parsed.indicators)

    if activities.empty:
        st.error("The selected log has no parseable Activities log section.")
        return

    products = sorted(activities[PRODUCT_COLUMN].dropna().unique())
    with st.sidebar:
        selected_products = st.multiselect("Products", products, default=products)
        timestamps = sorted(activities["timestamp"].dropna().unique())
        selected_timestamp = st.select_slider(
            "Debug timestamp",
            options=timestamps,
            value=timestamps[-1] if timestamps else None,
        )

    if not selected_products:
        st.warning("Select at least one product.")
        return

    option_chain = infer_option_chain(products)
    option_products: list[str] = []
    underlying_product: str | None = None
    option_expiry_day = float(default_option_expiry_day(activities))
    delta_rebalance_threshold = 20.0
    if not option_chain.empty:
        available_options = sorted(option_chain["product"].unique())
        default_options = [product for product in available_options if product in selected_products] or available_options
        with st.sidebar:
            st.subheader("Options Analytics")
            option_products = st.multiselect("Voucher products", available_options, default=default_options)
            underlying_choices = sorted(products)
            inferred_underlying = infer_underlying_product(products, option_products)
            underlying_index = (
                underlying_choices.index(inferred_underlying)
                if inferred_underlying in underlying_choices
                else 0
            )
            underlying_product = st.selectbox("Underlying product", underlying_choices, index=underlying_index)
            option_expiry_day = st.number_input(
                "Option expiry day",
                min_value=0.0,
                value=option_expiry_day,
                step=1.0,
            )
            delta_rebalance_threshold = st.number_input(
                "Delta rebalance threshold",
                min_value=0.0,
                value=delta_rebalance_threshold,
                step=1.0,
            )

    options_analytics = build_options_analytics(
        activities,
        option_products,
        underlying_product,
        option_expiry_day,
    )

    latest_total_pnl = (
        activities.sort_values("timestamp").groupby(PRODUCT_COLUMN).tail(1)["profit_and_loss"].sum()
    )
    own_fill_count = int(trades["is_own_trade"].sum()) if not trades.empty else 0
    total_debug_lines = len(parsed.debug_lines)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Products", len(products))
    metric_columns[1].metric("Latest total PnL", f"{latest_total_pnl:,.1f}")
    metric_columns[2].metric("Own fills", f"{own_fill_count:,}")
    metric_columns[3].metric("Indicators", f"{len(indicators):,}")

    tab_market, tab_volatility, tab_greeks, tab_pnl, tab_fills, tab_indicators, tab_logs, tab_raw = st.tabs(
        ["Market", "Volatility Surface", "Greeks", "PnL", "Fill Rate", "Indicators", "Logs", "Raw Data"]
    )

    with tab_market:
        st.plotly_chart(
            plot_orderbook(activities, trades, selected_products, max_plot_points),
            use_container_width=True,
        )
        st.plotly_chart(plot_spreads(activities, selected_products), use_container_width=True)
        st.plotly_chart(
            plot_z_scores(activities, selected_products, entry_threshold, exit_threshold),
            use_container_width=True,
        )
        normalizer_options = ["none", "rolling_mean"]
        if not indicators.empty:
            normalizer_options.extend(sorted(indicators["label"].unique()))
        normalizer = st.selectbox("Normalize mid-price by", normalizer_options)
        normalized = build_normalized_mid_series(activities, selected_products, normalizer, indicators)
        if not normalized.empty:
            st.plotly_chart(plot_normalized_mid(normalized), use_container_width=True)
        pair_spreads = build_pair_spreads(activities, z_window)
        if not pair_spreads.empty:
            pair_choices = sorted(pair_spreads["pair"].unique())
            selected_pairs = st.multiselect("Pair/basket spreads", pair_choices, default=pair_choices[:3])
            if selected_pairs:
                st.plotly_chart(plot_pair_spreads(pair_spreads, selected_pairs), use_container_width=True)

    with tab_volatility:
        if option_chain.empty:
            st.info("No voucher/option products detected. Expected names containing VOUCHER or OPTION with a trailing strike.")
        elif options_analytics.empty:
            st.warning("Select voucher products and a matching underlying to build volatility analytics.")
        else:
            fit_method = st.selectbox("Smile fit", SMILE_FIT_METHODS)
            fitted_options = fit_volatility_surface(options_analytics, fit_method)
            st.plotly_chart(plot_iv_time_series(fitted_options), use_container_width=True)

            option_times = sorted(fitted_options["plot_time"].dropna().unique())
            if option_times:
                default_smile_time = min(
                    option_times,
                    key=lambda value: abs(float(value) - float(selected_timestamp or value)),
                )
                smile_time = st.select_slider(
                    "Smile timestamp",
                    options=option_times,
                    value=default_smile_time,
                )
                st.plotly_chart(
                    plot_smile_snapshot(fitted_options, smile_time, fit_method),
                    use_container_width=True,
                )

                drift_defaults = evenly_spaced_values(option_times, 5)
                drift_times = st.multiselect(
                    "Smile drift timestamps",
                    option_times,
                    default=drift_defaults,
                )
                if drift_times:
                    st.plotly_chart(plot_smile_drift(fitted_options, drift_times), use_container_width=True)

            st.plotly_chart(plot_bs_price_scatter(fitted_options), use_container_width=True)
            st.plotly_chart(plot_iv_residuals(fitted_options), use_container_width=True)

    with tab_greeks:
        if options_analytics.empty:
            st.info("No option Greek series available for the current voucher/underlying selection.")
        else:
            for greek in ["delta", "gamma", "vega", "theta"]:
                st.plotly_chart(plot_greek_time_series(options_analytics, greek), use_container_width=True)

            portfolio_greeks = build_portfolio_greeks(activities, options_analytics, trades, underlying_product)
            if portfolio_greeks.empty:
                st.info("No portfolio Greek exposure available. Own-trade history may be empty.")
            else:
                st.plotly_chart(plot_portfolio_greeks(portfolio_greeks), use_container_width=True)
                st.plotly_chart(
                    plot_hedge_error(portfolio_greeks, delta_rebalance_threshold),
                    use_container_width=True,
                )
                st.dataframe(portfolio_greeks.tail(25), use_container_width=True, hide_index=True)

    with tab_pnl:
        st.plotly_chart(plot_pnl(activities, selected_products), use_container_width=True)
        pnl_table = decompose_pnl(activities[activities[PRODUCT_COLUMN].isin(selected_products)], trades)
        st.dataframe(pnl_table, use_container_width=True, hide_index=True)
        st.caption(
            "Market-making PnL is estimated from fill edge versus mid at fill time. "
            "Directional PnL is the residual against official mark-to-market PnL."
        )
        option_attribution = build_option_pnl_attribution(
            activities,
            trades,
            options_analytics,
            underlying_product,
        )
        if not option_attribution.empty:
            st.subheader("Options PnL Attribution")
            st.plotly_chart(plot_option_pnl_attribution(option_attribution), use_container_width=True)
            st.dataframe(option_attribution, use_container_width=True, hide_index=True)
            st.caption(
                "Options attribution assumes call vouchers, flat opening inventory, and hedge PnL allocated to the "
                "selected underlying leg. Theta decay uses annualized BS theta times elapsed TTE."
            )

    with tab_fills:
        fill_report = build_fill_report(parsed, selected_products)
        if fill_report.empty:
            st.info("No fill diagnostics available.")
        else:
            st.dataframe(fill_report, use_container_width=True, hide_index=True)
            numeric_columns = [
                column
                for column in ["avg_order_fill_ratio", "fill_timestamp_rate"]
                if column in fill_report.columns
            ]
            if numeric_columns:
                melted = fill_report.melt(
                    id_vars=["product"],
                    value_vars=numeric_columns,
                    var_name="metric",
                    value_name="rate",
                )
                st.plotly_chart(
                    px.bar(melted, x="product", y="rate", color="metric", barmode="group", title="Fill Rate"),
                    use_container_width=True,
                )
            if parsed.order_intents.empty:
                st.warning(
                    "No structured order submission logs found. For exact limit-order fill rate, "
                    "print lines like: ORDER product=ASH_COATED_OSMIUM side=BUY price=9999 qty=10"
                )

    with tab_indicators:
        st.subheader("Logged Indicator Series")
        if indicators.empty:
            st.info(
                "No JSON numeric indicators found. Frankfurt-style logs such as "
                "`print(json.dumps({'OPTION': {'theo_diff': value}}))` will appear here."
            )
        else:
            indicator_labels = sorted(indicators["label"].unique())
            selected_indicators = st.multiselect(
                "Indicators",
                indicator_labels,
                default=indicator_labels[: min(5, len(indicator_labels))],
            )
            if selected_indicators:
                st.plotly_chart(plot_indicators(indicators, selected_indicators), use_container_width=True)
            st.dataframe(indicators, use_container_width=True, hide_index=True)

    with tab_logs:
        render_debug_log(parsed.debug_lines, int(selected_timestamp) if selected_timestamp is not None else None)
        if not parsed.sandbox.empty:
            sandbox_errors = parsed.sandbox[parsed.sandbox["sandboxLog"].astype(str).str.len() > 0]
            st.subheader("Sandbox Errors")
            st.dataframe(sandbox_errors, use_container_width=True, hide_index=True)

    with tab_raw:
        st.subheader("Activities")
        st.dataframe(activities[activities[PRODUCT_COLUMN].isin(selected_products)], use_container_width=True)
        st.subheader("Trade History")
        st.dataframe(trades[trades["symbol"].isin(selected_products)] if not trades.empty else trades)
        st.subheader("Parsed Order Intents")
        st.dataframe(parsed.order_intents, use_container_width=True, hide_index=True)
        st.subheader("Parsed Indicators")
        st.dataframe(indicators, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

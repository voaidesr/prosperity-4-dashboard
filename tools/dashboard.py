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
    """Create official product PnL chart from the activity log."""

    subset = activities[activities[PRODUCT_COLUMN].isin(products)]
    fig = px.line(
        subset,
        x="timestamp",
        y="profit_and_loss",
        color=PRODUCT_COLUMN,
        title="Official Per-Product Mark-To-Market PnL",
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

    tab_market, tab_pnl, tab_fills, tab_indicators, tab_logs, tab_raw = st.tabs(
        ["Market", "PnL", "Fill Rate", "Indicators", "Logs", "Raw Data"]
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

    with tab_pnl:
        st.plotly_chart(plot_pnl(activities, selected_products), use_container_width=True)
        pnl_table = decompose_pnl(activities[activities[PRODUCT_COLUMN].isin(selected_products)], trades)
        st.dataframe(pnl_table, use_container_width=True, hide_index=True)
        st.caption(
            "Market-making PnL is estimated from fill edge versus mid at fill time. "
            "Directional PnL is the residual against official mark-to-market PnL."
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

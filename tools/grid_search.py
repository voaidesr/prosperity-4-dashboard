"""Vectorized parameter grid search for Prosperity price CSVs.

The script is designed for fast research over historical ``data/prices`` files.
It does not try to reproduce the full matching engine. Instead, it evaluates a
parameterized signal model on mid-price returns with turnover costs estimated
from half the quoted spread. Use it to rank parameter regions before validating
final candidates in the exchange backtester.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from glob import glob
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRICE_GLOB = "data/prices/*.csv"
DEFAULT_WINDOWS = "50,100,200"
DEFAULT_ENTRIES = "1.0,1.5,2.0"
DEFAULT_EXITS = "0.0,0.5"
DEFAULT_LIMITS = "20,40,80"


@dataclass(frozen=True)
class SearchParams:
    """Hyperparameters evaluated by one grid-search run."""

    product: str
    strategy: str
    window: int
    entry_z: float
    exit_z: float
    position_limit: int
    cost_multiplier: float


@dataclass(frozen=True)
class SpreadDefinition:
    """Linear synthetic instrument definition."""

    name: str
    weights: dict[str, float]


def parse_csv_values(raw: str, cast: type) -> list:
    """Parse comma-separated CLI values into a typed list."""

    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(cast(token))
    return values


def parse_spread_definition(raw: str) -> SpreadDefinition:
    """Parse ``NAME=1*PRODUCT_A-2*PRODUCT_B`` into weights."""

    if "=" in raw:
        name, expression = raw.split("=", 1)
    elif ":" in raw:
        name, expression = raw.split(":", 1)
    else:
        raise ValueError(f"Spread definition needs NAME=EXPRESSION: {raw}")

    weights: dict[str, float] = {}
    normalized = expression.replace("-", "+-")
    for term in normalized.split("+"):
        term = term.strip()
        if not term:
            continue
        if "*" in term:
            raw_weight, product = term.split("*", 1)
            raw_weight = raw_weight.strip()
            weight = -1.0 if raw_weight == "-" else float(raw_weight)
        else:
            product = term
            weight = -1.0 if product.startswith("-") else 1.0
            product = product.lstrip("-")
        product = product.strip()
        if not product:
            continue
        weights[product] = weights.get(product, 0.0) + weight

    if not weights:
        raise ValueError(f"Spread definition has no products: {raw}")
    return SpreadDefinition(name=name.strip(), weights=weights)


def build_pair_spread_definitions(products: list[str]) -> list[SpreadDefinition]:
    """Build simple pair spreads for exploratory Round 3 relationship scans."""

    spreads: list[SpreadDefinition] = []
    for left_idx, left in enumerate(products):
        for right in products[left_idx + 1 :]:
            spreads.append(SpreadDefinition(name=f"{left}-{right}", weights={left: 1.0, right: -1.0}))
    return spreads


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    """Expand explicit files and glob patterns into sorted paths."""

    paths: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.exists():
            paths.add(candidate)
            continue
        for match in glob(pattern):
            match_path = Path(match)
            if match_path.is_file():
                paths.add(match_path)
    return sorted(paths)


def load_price_data(paths: list[Path]) -> pd.DataFrame:
    """Load and normalize Prosperity semicolon-delimited price files."""

    if not paths:
        raise FileNotFoundError(f"No price files found. Default pattern: {PRICE_GLOB}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep=";")
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    numeric_columns = [column for column in data.columns if column not in {"product", "source_file"}]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["best_bid"] = data["bid_price_1"]
    data["best_ask"] = data["ask_price_1"]
    data["spread"] = data["best_ask"] - data["best_bid"]
    data["half_spread"] = (data["spread"] / 2.0).clip(lower=0.0)
    data = data.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    data.loc[data["mid_price"] <= 0, "mid_price"] = np.nan
    data["mid_price"] = data.groupby(["source_file", "day", "product"])["mid_price"].transform(
        lambda series: series.ffill().bfill()
    )
    return data


def build_signal(mid: pd.Series, window: int, entry_z: float, exit_z: float, strategy: str) -> pd.Series:
    """Build a hysteresis z-score trading signal.

    Signal values are -1, 0, or 1. Mean reversion buys negative z-scores and
    sells positive z-scores. Trend following does the opposite.
    """

    min_periods = max(5, window // 5)
    rolling_mean = mid.rolling(window, min_periods=min_periods).mean()
    rolling_std = mid.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    z_score = (mid - rolling_mean) / rolling_std

    if strategy == "mean-reversion":
        raw = np.select([z_score <= -entry_z, z_score >= entry_z], [1.0, -1.0], default=np.nan)
    elif strategy == "trend":
        raw = np.select([z_score <= -entry_z, z_score >= entry_z], [-1.0, 1.0], default=np.nan)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    signal = pd.Series(raw, index=mid.index, dtype="float64")
    signal[z_score.abs() <= exit_z] = 0.0
    return signal.ffill().fillna(0.0)


def evaluate_product(product_data: pd.DataFrame, params: SearchParams) -> dict[str, float | int | str]:
    """Evaluate one product and one parameter set."""

    day_results = []
    for (_, day), day_frame in product_data.groupby(["source_file", "day"], sort=False):
        day_frame = day_frame.sort_values("timestamp").copy()
        mid = day_frame["mid_price"].astype(float)
        signal = build_signal(mid, params.window, params.entry_z, params.exit_z, params.strategy)

        # Use previous timestamp's position to avoid lookahead on the return.
        target_position = (signal * params.position_limit).round().astype(float)
        position = target_position.shift(1).fillna(0.0)
        price_change = mid.diff().fillna(0.0)

        turnover = target_position.diff().abs().fillna(target_position.abs())
        cost = turnover * day_frame["half_spread"].fillna(0.0) * params.cost_multiplier
        pnl = position * price_change - cost
        day_results.append(
            pd.DataFrame(
                {
                    "source_file": day_frame["source_file"],
                    "day": day_frame["day"],
                    "pnl": pnl,
                    "turnover": turnover,
                    "series_change": price_change,
                },
                index=day_frame.index,
            )
        )

    if not day_results:
        return {}

    result = pd.concat(day_results).sort_index()
    return summarize_result(result, params, kind="product")


def evaluate_spread(prices: pd.DataFrame, spread: SpreadDefinition, params: SearchParams) -> dict[str, float | int | str]:
    """Evaluate a linear spread as a synthetic Round 3 candidate."""

    required = set(spread.weights)
    if not required.issubset(set(prices["product"].unique())):
        return {}

    day_results = []
    for (_, day), day_frame in prices.groupby(["source_file", "day"], sort=False):
        pivot_mid = day_frame.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="last")
        pivot_cost = day_frame.pivot_table(index="timestamp", columns="product", values="half_spread", aggfunc="last")
        if not required.issubset(pivot_mid.columns):
            continue

        spread_series = sum(weight * pivot_mid[product] for product, weight in spread.weights.items())
        spread_cost = sum(abs(weight) * pivot_cost[product].fillna(0.0) for product, weight in spread.weights.items())
        valid = spread_series.dropna().index
        spread_series = spread_series.loc[valid]
        spread_cost = spread_cost.reindex(valid).fillna(0.0)
        if spread_series.empty:
            continue

        signal = build_signal(spread_series, params.window, params.entry_z, params.exit_z, params.strategy)
        target_position = (signal * params.position_limit).round().astype(float)
        position = target_position.shift(1).fillna(0.0)
        series_change = spread_series.diff().fillna(0.0)
        turnover = target_position.diff().abs().fillna(target_position.abs())
        pnl = position * series_change - turnover * spread_cost * params.cost_multiplier
        day_results.append(
            pd.DataFrame(
                {
                    "source_file": day_frame["source_file"].iloc[0],
                    "day": day,
                    "pnl": pnl,
                    "turnover": turnover,
                    "series_change": series_change,
                },
                index=spread_series.index,
            )
        )

    if not day_results:
        return {}

    result = pd.concat(day_results).sort_index()
    output = summarize_result(result, params, kind="spread")
    output["weights"] = ",".join(f"{product}:{weight:g}" for product, weight in spread.weights.items())
    return output


def summarize_result(
    result: pd.DataFrame,
    params: SearchParams,
    kind: str,
) -> dict[str, float | int | str]:
    """Summarize an evaluated product or spread with robustness diagnostics."""

    result = result.reset_index(drop=True)
    pnl = result["pnl"].astype(float)
    equity = pnl.cumsum()
    total_pnl = float(equity.iloc[-1]) if not equity.empty else 0.0
    max_drawdown = calculate_max_drawdown(equity)
    sharpe = calculate_sharpe(pnl)
    turnover = float(result["turnover"].sum())
    day_pnl = result.groupby(["source_file", "day"])["pnl"].sum()
    positive_day_ratio = float((day_pnl > 0).mean()) if not day_pnl.empty else 0.0
    worst_day_pnl = float(day_pnl.min()) if not day_pnl.empty else 0.0
    pnl_std_by_day = float(day_pnl.std(ddof=0)) if len(day_pnl) > 1 else 0.0
    risk_adjusted_return = total_pnl / max(max_drawdown, 1.0)
    stability_score = risk_adjusted_return * positive_day_ratio + worst_day_pnl / max(abs(total_pnl), 1.0)

    return {
        "instrument": params.product,
        "kind": kind,
        "strategy": params.strategy,
        "window": params.window,
        "entry_z": params.entry_z,
        "exit_z": params.exit_z,
        "position_limit": params.position_limit,
        "cost_multiplier": params.cost_multiplier,
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "positive_day_ratio": positive_day_ratio,
        "worst_day_pnl": worst_day_pnl,
        "pnl_std_by_day": pnl_std_by_day,
        "lag1_autocorr": calculate_lag_autocorr(result["series_change"]),
        "risk_adjusted_return": risk_adjusted_return,
        "stability_score": stability_score,
    }


def calculate_sharpe(pnl: pd.Series) -> float:
    """Calculate discrete-time Sharpe ratio on per-timestamp PnL."""

    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    if pnl.empty:
        return 0.0
    std = pnl.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(len(pnl)) * pnl.mean() / std)


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown."""

    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = running_max - equity
    return float(drawdown.max())


def calculate_lag_autocorr(series: pd.Series, lag: int = 1) -> float:
    """Calculate lag autocorrelation for exploratory mean-reversion checks."""

    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= lag + 1:
        return 0.0
    value = clean.autocorr(lag=lag)
    return 0.0 if pd.isna(value) else float(value)


def run_grid_search(
    prices: pd.DataFrame,
    products: list[str],
    strategies: list[str],
    windows: list[int],
    entry_values: list[float],
    exit_values: list[float],
    position_limits: list[int],
    cost_multipliers: list[float],
    spread_definitions: list[SpreadDefinition],
    rank_by: str,
) -> pd.DataFrame:
    """Run all parameter combinations and return ranked results."""

    rows: list[dict[str, float | int | str]] = []
    for product in products:
        product_data = prices[prices["product"] == product].copy()
        if product_data.empty:
            continue

        for strategy, window, entry_z, exit_z, limit, cost_multiplier in itertools.product(
            strategies,
            windows,
            entry_values,
            exit_values,
            position_limits,
            cost_multipliers,
        ):
            if exit_z >= entry_z:
                continue
            params = SearchParams(
                product=product,
                strategy=strategy,
                window=window,
                entry_z=entry_z,
                exit_z=exit_z,
                position_limit=limit,
                cost_multiplier=cost_multiplier,
            )
            result = evaluate_product(product_data, params)
            if result:
                rows.append(result)

    for spread in spread_definitions:
        for strategy, window, entry_z, exit_z, limit, cost_multiplier in itertools.product(
            strategies,
            windows,
            entry_values,
            exit_values,
            position_limits,
            cost_multipliers,
        ):
            if exit_z >= entry_z:
                continue
            params = SearchParams(
                product=spread.name,
                strategy=strategy,
                window=window,
                entry_z=entry_z,
                exit_z=exit_z,
                position_limit=limit,
                cost_multiplier=cost_multiplier,
            )
            result = evaluate_spread(prices, spread, params)
            if result:
                rows.append(result)

    if not rows:
        return pd.DataFrame()

    result_frame = pd.DataFrame(rows)
    sort_columns = [rank_by, "positive_day_ratio", "sharpe", "total_pnl"]
    sort_columns = [column for column in sort_columns if column in result_frame.columns]
    return result_frame.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run vectorized parameter grid search over Prosperity price CSVs.",
    )
    parser.add_argument(
        "--prices",
        nargs="+",
        default=[PRICE_GLOB],
        help="Price CSV paths or glob patterns. Default: data/prices/*.csv",
    )
    parser.add_argument(
        "--products",
        default="",
        help="Comma-separated products. Default: all products in the price files.",
    )
    parser.add_argument(
        "--strategies",
        default="mean-reversion",
        help="Comma-separated strategies: mean-reversion,trend",
    )
    parser.add_argument("--ma-windows", default=DEFAULT_WINDOWS, help="Comma-separated rolling windows.")
    parser.add_argument("--entry-z", default=DEFAULT_ENTRIES, help="Comma-separated z-score entry thresholds.")
    parser.add_argument("--exit-z", default=DEFAULT_EXITS, help="Comma-separated z-score exit thresholds.")
    parser.add_argument("--position-limits", default=DEFAULT_LIMITS, help="Comma-separated position limits.")
    parser.add_argument(
        "--cost-multipliers",
        default="1.0",
        help="Comma-separated turnover cost multipliers applied to half-spread.",
    )
    parser.add_argument(
        "--spread-def",
        action="append",
        default=[],
        help=(
            "Synthetic spread definition, e.g. "
            "'BASKET_EDGE=1*BASKET-6*CROISSANTS-3*JAMS-1*DJEMBES'. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--pair-spreads",
        action="store_true",
        help="Also evaluate every pairwise product spread as a quick relationship scan.",
    )
    parser.add_argument(
        "--rank-by",
        default="stability_score",
        choices=["stability_score", "risk_adjusted_return", "sharpe", "total_pnl"],
        help="Primary ranking metric. Default favors robust multi-day performance.",
    )
    parser.add_argument("--top", type=int, default=20, help="Rows to print.")
    parser.add_argument("--save-csv", default="", help="Optional output CSV for all grid results.")
    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    paths = expand_paths(args.prices)
    prices = load_price_data(paths)

    products = (
        parse_csv_values(args.products, str)
        if args.products.strip()
        else sorted(prices["product"].dropna().unique().tolist())
    )
    strategies = parse_csv_values(args.strategies, str)
    windows = parse_csv_values(args.ma_windows, int)
    entry_values = parse_csv_values(args.entry_z, float)
    exit_values = parse_csv_values(args.exit_z, float)
    position_limits = parse_csv_values(args.position_limits, int)
    cost_multipliers = parse_csv_values(args.cost_multipliers, float)
    spread_definitions = [parse_spread_definition(raw) for raw in args.spread_def]
    if args.pair_spreads:
        spread_definitions.extend(build_pair_spread_definitions(products))

    results = run_grid_search(
        prices=prices,
        products=products,
        strategies=strategies,
        windows=windows,
        entry_values=entry_values,
        exit_values=exit_values,
        position_limits=position_limits,
        cost_multipliers=cost_multipliers,
        spread_definitions=spread_definitions,
        rank_by=args.rank_by,
    )

    if results.empty:
        print("No grid-search results generated.")
        return

    display_columns = [
        "instrument",
        "kind",
        "strategy",
        "window",
        "entry_z",
        "exit_z",
        "position_limit",
        "total_pnl",
        "sharpe",
        "max_drawdown",
        "positive_day_ratio",
        "worst_day_pnl",
        "lag1_autocorr",
        "risk_adjusted_return",
        "stability_score",
        "turnover",
    ]
    display_columns = [column for column in display_columns if column in results.columns]
    print(results[display_columns].head(args.top).to_string(index=False))

    if args.save_csv:
        output_path = Path(args.save_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nSaved full grid results to {output_path}")


if __name__ == "__main__":
    main()

"""Deployable upload-first dashboard for Prosperity backtest logs."""

from __future__ import annotations

import hashlib
import hmac
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dashboard import (  # noqa: E402
    DEFAULT_POSITION_LIMIT,
    DEFAULT_RUNTIME_TIMEOUT_MS,
    PRODUCT_COLUMN,
    SMILE_FIT_METHODS,
    add_rolling_z_scores,
    build_basket_hedge_tracker,
    build_ai_markdown_report,
    build_conversion_report,
    build_fill_by_price_distance,
    build_option_pnl_attribution,
    build_options_analytics,
    build_fill_report,
    build_iv_residual_z_scores,
    build_normalized_mid_series,
    build_pair_spreads,
    build_position_limit_report,
    build_portfolio_greeks,
    build_rejected_order_report,
    build_runtime_report,
    build_single_spread_series,
    build_spread_stationarity,
    build_synthetic_basket,
    build_trader_flow,
    compare_product_pnl,
    decompose_pnl,
    default_option_expiry_day,
    downsample_by_timestamp,
    evenly_spaced_values,
    fit_volatility_surface,
    infer_option_chain,
    infer_underlying_product,
    nearest_value,
    parse_backtest_text,
    parse_basket_formula,
    parse_position_limits,
    plot_basket_spread,
    plot_bs_price_scatter,
    plot_conversion_pnl,
    plot_fill_by_distance,
    plot_greek_time_series,
    plot_hedge_error,
    plot_indicators,
    plot_iv_residuals,
    plot_iv_residual_z_scores,
    plot_iv_time_series,
    plot_normalized_mid,
    plot_orderbook,
    plot_option_pnl_attribution,
    plot_parameter_sweep_heatmap,
    plot_pair_spreads,
    plot_pnl_diff,
    plot_pnl,
    plot_position_limits,
    plot_portfolio_greeks,
    plot_runtime_report,
    plot_smile_drift,
    plot_smile_snapshot,
    plot_spreads,
    plot_spread_histogram,
    plot_stationarity_diagnostics,
    plot_synthetic_basket,
    plot_trader_flow,
    plot_z_scores,
    prepare_indicator_labels,
    render_debug_log,
)


DEFAULT_MAX_UPLOAD_MB = 25
ALLOWED_SUFFIXES = {".log", ".txt", ".json"}


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from environment or Streamlit secrets."""

    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name.lower(), default))
    except Exception:
        return default


def require_password() -> bool:
    """Optionally gate the app with a shared password."""

    password = get_secret("DASHBOARD_PASSWORD")
    if not password:
        return True

    st.title("Prosperity Log Dashboard")
    candidate = st.text_input("Dashboard password", type="password")
    if not candidate:
        st.info("Enter the shared dashboard password to continue.")
        return False
    if not hmac.compare_digest(candidate, password):
        st.error("Invalid password.")
        return False
    return True


def validate_upload(name: str, size: int, max_upload_mb: int) -> str | None:
    """Return an error message if an uploaded file should be rejected."""

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return "Only .log, .txt, and .json files are accepted."
    if size > max_upload_mb * 1024 * 1024:
        return f"File is too large. Limit is {max_upload_mb} MB."
    return None


@st.cache_data(show_spinner=False, max_entries=10, ttl=60 * 60)
def parse_uploaded_log(source_name: str, checksum: str, text: str):
    """Parse uploaded log text and cache by checksum."""

    del checksum
    return parse_backtest_text(text, source_name=source_name)


def decode_upload(uploaded_file) -> tuple[str, str]:
    """Read uploaded bytes as UTF-8 text without writing them to disk."""

    raw = uploaded_file.getvalue()
    checksum = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    return checksum, text


def timestamp_number_input(label: str, timestamps: list[int | float], key: str) -> int | float | None:
    """Render a cheap timestamp input and snap to the nearest available timestamp."""

    if not timestamps:
        return None
    minimum = int(min(timestamps))
    maximum = int(max(timestamps))
    selected = st.number_input(label, min_value=minimum, max_value=maximum, value=maximum, step=1, key=key)
    return nearest_value(timestamps, selected)


def parse_parameter_sweep_upload(uploaded_file) -> pd.DataFrame:
    """Parse grid-search CSV/JSON uploads."""

    if uploaded_file is None:
        return pd.DataFrame()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".json":
        return pd.read_json(uploaded_file)
    return pd.read_csv(uploaded_file)


def style_delta_table(frame: pd.DataFrame):
    """Color PnL deltas for submission diffs."""

    def color_delta(value):
        if pd.isna(value):
            return ""
        return "color: #15803d" if value >= 0 else "color: #b91c1c"

    return frame.style.map(color_delta, subset=["delta_pnl"])


def render_dashboard(parsed, comparison=None) -> None:
    """Render dashboard controls and visualizations for one parsed log."""

    with st.sidebar:
        st.header("Controls")
        z_window = st.slider("Rolling z-score window", min_value=10, max_value=500, value=100, step=10)
        entry_threshold = st.number_input("Entry z-score threshold", value=2.0, min_value=0.1, step=0.1)
        exit_threshold = st.number_input("Exit z-score threshold", value=0.5, min_value=0.0, step=0.1)
        z_score_source = st.selectbox("Z-score source", ["mid_price", "iv_residual"])
        max_plot_points = st.slider("Max plotted timestamps", min_value=500, max_value=20000, value=5000, step=500)

    activities = add_rolling_z_scores(parsed.activities, z_window)
    trades = parsed.trades
    indicators = prepare_indicator_labels(parsed.indicators)

    if activities.empty:
        st.error("The uploaded log has no parseable Activities log section.")
        return

    products = sorted(activities[PRODUCT_COLUMN].dropna().unique())
    timestamps = sorted(activities["timestamp"].dropna().unique())
    with st.sidebar:
        selected_products = st.multiselect("Products", products, default=products)
        selected_timestamp = timestamp_number_input("Debug timestamp", timestamps, "debug_timestamp")

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
            underlying_index = underlying_choices.index(inferred_underlying) if inferred_underlying in underlying_choices else 0
            underlying_product = st.selectbox("Underlying product", underlying_choices, index=underlying_index)
            option_expiry_day = st.number_input("Option expiry day", min_value=0.0, value=option_expiry_day, step=1.0)
            delta_rebalance_threshold = st.number_input(
                "Delta rebalance threshold",
                min_value=0.0,
                value=delta_rebalance_threshold,
                step=1.0,
            )

    with st.sidebar:
        st.subheader("AI Report")
        st.caption("Use the AI Report view for configurable basket, limits, and sweep inputs.")
        if st.button("Build AI report", key="build_ai_report_sidebar"):
            with st.spinner("Building AI markdown report..."):
                st.session_state["ai_report_markdown"] = build_ai_markdown_report(
                    parsed=parsed,
                    activities=activities,
                    indicators=indicators,
                    selected_products=selected_products,
                    option_products=option_products,
                    underlying_product=underlying_product,
                    option_expiry_day=option_expiry_day,
                    z_window=z_window,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    delta_rebalance_threshold=delta_rebalance_threshold,
                    runtime_threshold_ms=float(DEFAULT_RUNTIME_TIMEOUT_MS),
                    max_rows_per_table=50,
                )
                st.session_state["ai_report_filename"] = f"{Path(parsed.path.name).stem or 'prosperity'}_ai_report.md"

        if st.session_state.get("ai_report_markdown"):
            st.download_button(
                "Download AI markdown report",
                st.session_state["ai_report_markdown"],
                file_name=st.session_state.get("ai_report_filename", "prosperity_ai_report.md"),
                mime="text/markdown",
                key="download_ai_report_sidebar",
            )

    def current_options() -> pd.DataFrame:
        return build_options_analytics(activities, option_products, underlying_product, option_expiry_day)

    latest_total_pnl = activities.sort_values("timestamp").groupby(PRODUCT_COLUMN).tail(1)["profit_and_loss"].sum()
    own_fill_count = int(trades["is_own_trade"].sum()) if not trades.empty else 0

    metric_columns = st.columns(4)
    metric_columns[0].metric("Products", len(products))
    metric_columns[1].metric("Latest total PnL", f"{latest_total_pnl:,.1f}")
    metric_columns[2].metric("Own fills", f"{own_fill_count:,}")
    metric_columns[3].metric("Indicators", f"{len(indicators):,}")

    views = [
        "Market",
        "Volatility Surface",
        "Greeks",
        "PnL",
        "Stationarity",
        "Baskets",
        "Risk",
        "Fill Rate",
        "AI Report",
        "Parameter Sweep",
        "Submission Diff",
        "Runtime",
        "Trader Flow",
        "Conversions",
        "Indicators",
        "Logs",
        "Raw Data",
    ]
    view = st.radio("Dashboard view", views, horizontal=True)

    if view == "Market":
        st.plotly_chart(plot_orderbook(activities, trades, selected_products, max_plot_points), use_container_width=True)
        st.plotly_chart(plot_spreads(downsample_by_timestamp(activities, max_plot_points), selected_products), use_container_width=True)
        if z_score_source == "mid_price":
            st.plotly_chart(plot_z_scores(activities, selected_products, entry_threshold, exit_threshold), use_container_width=True)
        else:
            options_analytics = current_options()
            if options_analytics.empty:
                st.info("IV residual z-scores need voucher products and a selected underlying.")
            else:
                fit_method = st.selectbox("IV residual fit", SMILE_FIT_METHODS)
                fitted_options = fit_volatility_surface(options_analytics, fit_method)
                residual_z = build_iv_residual_z_scores(fitted_options, z_window)
                st.plotly_chart(plot_iv_residual_z_scores(residual_z, entry_threshold, exit_threshold), use_container_width=True)

        normalizer_options = ["none", "rolling_mean"]
        if not indicators.empty:
            normalizer_options.extend(sorted(indicators["label"].unique()))
        normalizer = st.selectbox("Normalize mid-price by", normalizer_options)
        normalized = build_normalized_mid_series(activities, selected_products, normalizer, indicators)
        if not normalized.empty:
            st.plotly_chart(plot_normalized_mid(downsample_by_timestamp(normalized, max_plot_points)), use_container_width=True)

        if len(selected_products) >= 2:
            left = st.selectbox("Pair left", selected_products, index=0)
            right_default = 1 if len(selected_products) > 1 else 0
            right = st.selectbox("Pair right", selected_products, index=right_default)
            pair_spread = build_single_spread_series(activities, left, right, z_window)
            if not pair_spread.empty:
                st.plotly_chart(plot_pair_spreads(pair_spread, [pair_spread["pair"].iloc[0]]), use_container_width=True)

    elif view == "Volatility Surface":
        if option_chain.empty:
            st.info("No voucher/option products detected. Expected names containing VOUCHER or OPTION with a trailing strike.")
        else:
            options_analytics = current_options()
            if options_analytics.empty:
                st.warning("Select voucher products and a matching underlying to build volatility analytics.")
            else:
                fit_method = st.selectbox("Smile fit", SMILE_FIT_METHODS)
                fitted_options = fit_volatility_surface(options_analytics, fit_method)
                fitted_plot = downsample_by_timestamp(fitted_options, max_plot_points)
                st.plotly_chart(plot_iv_time_series(fitted_plot), use_container_width=True)

                option_times = sorted(fitted_options["plot_time"].dropna().unique())
                if option_times:
                    smile_time = timestamp_number_input("Smile timestamp", option_times, "smile_timestamp")
                    if smile_time is not None:
                        st.plotly_chart(plot_smile_snapshot(fitted_options, smile_time, fit_method), use_container_width=True)
                    drift_defaults = evenly_spaced_values(option_times, 5)
                    drift_times = st.multiselect("Smile drift timestamps", option_times, default=drift_defaults)
                    if drift_times:
                        st.plotly_chart(plot_smile_drift(fitted_options, drift_times), use_container_width=True)

                st.plotly_chart(plot_bs_price_scatter(fitted_plot), use_container_width=True)
                st.plotly_chart(plot_iv_residuals(fitted_options), use_container_width=True)
                residual_z = build_iv_residual_z_scores(fitted_options, z_window)
                st.plotly_chart(plot_iv_residual_z_scores(residual_z, entry_threshold, exit_threshold), use_container_width=True)

    elif view == "Greeks":
        options_analytics = current_options()
        if options_analytics.empty:
            st.info("No option Greek series available for the current voucher/underlying selection.")
        else:
            options_plot = downsample_by_timestamp(options_analytics, max_plot_points)
            for greek in ["delta", "gamma", "vega", "theta", "rho"]:
                if greek in options_plot:
                    st.plotly_chart(plot_greek_time_series(options_plot, greek), use_container_width=True)
            portfolio_greeks = build_portfolio_greeks(activities, options_analytics, trades, underlying_product)
            if portfolio_greeks.empty:
                st.info("No portfolio Greek exposure available. Own-trade history may be empty.")
            else:
                latest_greeks = portfolio_greeks.tail(1).iloc[0]
                greek_columns = st.columns(4)
                greek_columns[0].metric("Net Delta", f"{latest_greeks['portfolio_delta']:,.2f}")
                greek_columns[1].metric("Net Gamma", f"{latest_greeks['portfolio_gamma']:,.2f}")
                greek_columns[2].metric("Net Vega", f"{latest_greeks['portfolio_vega']:,.2f}")
                greek_columns[3].metric("Net Theta", f"{latest_greeks['portfolio_theta']:,.2f}")
                portfolio_plot = downsample_by_timestamp(portfolio_greeks, max_plot_points)
                st.plotly_chart(plot_portfolio_greeks(portfolio_plot), use_container_width=True)
                st.plotly_chart(plot_hedge_error(portfolio_plot, delta_rebalance_threshold), use_container_width=True)
                st.dataframe(portfolio_greeks.tail(25), use_container_width=True, hide_index=True)

    elif view == "PnL":
        st.plotly_chart(plot_pnl(downsample_by_timestamp(activities, max_plot_points), selected_products), use_container_width=True)
        st.dataframe(
            decompose_pnl(activities[activities[PRODUCT_COLUMN].isin(selected_products)], trades),
            use_container_width=True,
            hide_index=True,
        )
        options_analytics = current_options()
        option_attribution = build_option_pnl_attribution(activities, trades, options_analytics, underlying_product)
        if not option_attribution.empty:
            st.subheader("Options PnL Attribution")
            st.plotly_chart(plot_option_pnl_attribution(option_attribution), use_container_width=True)
            st.dataframe(option_attribution, use_container_width=True, hide_index=True)
            st.caption(
                "Inventory PnL is mark-to-market on held positions. Delta/gamma/vega/theta columns decompose option "
                "inventory PnL; hedge PnL is allocated to the selected underlying leg."
            )

    elif view == "Stationarity":
        if len(selected_products) < 2:
            st.info("Select at least two products for spread diagnostics.")
        else:
            left = st.selectbox("Spread left", selected_products, index=0)
            right = st.selectbox("Spread right", selected_products, index=min(1, len(selected_products) - 1))
            stationarity_window = st.slider("Stationarity window", min_value=20, max_value=1000, value=max(100, z_window), step=20)
            spread_frame = build_single_spread_series(activities, left, right, z_window)
            if spread_frame.empty:
                st.info("No spread series available for this pair.")
            else:
                diagnostics = build_spread_stationarity(spread_frame, stationarity_window)
                st.plotly_chart(plot_pair_spreads(spread_frame, [spread_frame["pair"].iloc[0]]), use_container_width=True)
                st.plotly_chart(plot_stationarity_diagnostics(diagnostics), use_container_width=True)
                st.plotly_chart(plot_spread_histogram(spread_frame), use_container_width=True)
                st.caption("ADF p-values are a lightweight left-tail approximation because statsmodels is not a dependency.")

    elif view == "Baskets":
        formula = st.text_input("Basket formula", value="")
        basket_product, weights, error = parse_basket_formula(formula, products)
        if error:
            st.info(error)
        else:
            basket_frame = build_synthetic_basket(activities, basket_product, weights, z_window)
            if basket_frame.empty:
                st.warning("Could not build the synthetic basket from the selected formula.")
            else:
                st.plotly_chart(plot_synthetic_basket(downsample_by_timestamp(basket_frame, max_plot_points)), use_container_width=True)
                st.plotly_chart(plot_basket_spread(basket_frame, entry_threshold, exit_threshold), use_container_width=True)
                positions = build_basket_hedge_tracker(activities, trades, basket_product, weights)
                basket_position = st.number_input(
                    "Basket position for hedge tracker",
                    value=float(-positions["target_position"].iloc[0] / positions["coefficient"].iloc[0]) if not positions.empty and positions["coefficient"].iloc[0] else 0.0,
                    step=1.0,
                )
                st.dataframe(
                    build_basket_hedge_tracker(activities, trades, basket_product, weights, basket_position),
                    use_container_width=True,
                    hide_index=True,
                )

    elif view == "Risk":
        default_limit = st.number_input("Default product limit", min_value=1.0, value=float(DEFAULT_POSITION_LIMIT), step=1.0)
        overrides = st.text_area("Position limit overrides", value="", placeholder="AMETHYSTS=20, STARFRUIT=20")
        limits = parse_position_limits(overrides, selected_products, default_limit)
        limit_report = build_position_limit_report(activities, trades, selected_products, limits)
        st.plotly_chart(plot_position_limits(limit_report), use_container_width=True)
        st.dataframe(limit_report, use_container_width=True, hide_index=True)
        rejected = build_rejected_order_report(parsed.order_intents, trades, selected_products, limits)
        st.subheader("Orders That Would Breach Limits")
        if rejected.empty:
            st.info("No rejected-order candidates found from parsed order intent logs.")
        else:
            st.dataframe(rejected, use_container_width=True, hide_index=True)

    elif view == "Fill Rate":
        fill_report = build_fill_report(parsed, selected_products)
        if fill_report.empty:
            st.info("No fill diagnostics available.")
        else:
            st.dataframe(fill_report, use_container_width=True, hide_index=True)
        by_distance = build_fill_by_price_distance(parsed, selected_products)
        if by_distance.empty:
            st.info("Fill-rate by price distance needs structured order logs with product, side, price, and quantity.")
        else:
            st.plotly_chart(plot_fill_by_distance(by_distance), use_container_width=True)
            st.dataframe(by_distance, use_container_width=True, hide_index=True)

    elif view == "AI Report":
        st.subheader("AI Markdown Report")
        st.caption(
            "This report converts the dashboard data behind each visualization into markdown tables, "
            "diagnostics, and threshold warnings for an AI agent."
        )
        report_columns = st.columns(3)
        with report_columns[0]:
            report_default_limit = st.number_input(
                "Report default product limit",
                min_value=1.0,
                value=float(DEFAULT_POSITION_LIMIT),
                step=1.0,
                key="ai_report_default_limit",
            )
        with report_columns[1]:
            report_stationarity_window = st.slider(
                "Report stationarity window",
                min_value=20,
                max_value=1000,
                value=max(100, z_window),
                step=20,
                key="ai_report_stationarity_window",
            )
        with report_columns[2]:
            report_max_rows = st.slider(
                "Max rows per report table",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key="ai_report_max_rows",
            )

        report_pair_columns = st.columns(2)
        if len(selected_products) >= 2:
            with report_pair_columns[0]:
                report_stationarity_left = st.selectbox(
                    "Report spread left",
                    selected_products,
                    index=0,
                    key="ai_report_stationarity_left",
                )
            with report_pair_columns[1]:
                report_stationarity_right = st.selectbox(
                    "Report spread right",
                    selected_products,
                    index=min(1, len(selected_products) - 1),
                    key="ai_report_stationarity_right",
                )
        else:
            report_stationarity_left = selected_products[0] if selected_products else None
            report_stationarity_right = None

        report_basket_formula = st.text_input(
            "Optional basket formula for report",
            value="",
            placeholder="BASKET1 = 6A + 3B + 1*C",
            key="ai_report_basket_formula",
        )
        report_limit_overrides = st.text_area(
            "Optional position limit overrides for report",
            value="",
            placeholder="AMETHYSTS=20, STARFRUIT=20",
            key="ai_report_limit_overrides",
        )
        report_runtime_threshold = st.number_input(
            "Report runtime timeout threshold ms",
            min_value=1.0,
            value=float(DEFAULT_RUNTIME_TIMEOUT_MS),
            step=50.0,
            key="ai_report_runtime_threshold",
        )
        report_sweep_file = st.file_uploader(
            "Optional parameter sweep CSV/JSON for report",
            type=["csv", "json"],
            key="ai_report_parameter_sweep",
        )
        report_sweep = parse_parameter_sweep_upload(report_sweep_file)

        with st.spinner("Building AI markdown report..."):
            report_md = build_ai_markdown_report(
                parsed=parsed,
                activities=activities,
                indicators=indicators,
                selected_products=selected_products,
                option_products=option_products,
                underlying_product=underlying_product,
                option_expiry_day=option_expiry_day,
                z_window=z_window,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                delta_rebalance_threshold=delta_rebalance_threshold,
                default_position_limit=report_default_limit,
                position_limit_overrides=report_limit_overrides,
                basket_formula=report_basket_formula,
                stationarity_left=report_stationarity_left,
                stationarity_right=report_stationarity_right,
                stationarity_window=report_stationarity_window,
                comparison=comparison,
                parameter_sweep=report_sweep,
                runtime_threshold_ms=report_runtime_threshold,
                max_rows_per_table=report_max_rows,
            )

        report_name = f"{Path(parsed.path.name).stem or 'prosperity'}_ai_report.md"
        st.download_button(
            "Download AI markdown report",
            report_md,
            file_name=report_name,
            mime="text/markdown",
        )
        preview = report_md[:50000]
        st.text_area("Report preview", preview, height=500)
        if len(report_md) > len(preview):
            st.caption(f"Preview truncated to {len(preview):,} characters. The download contains the full report.")

    elif view == "Parameter Sweep":
        sweep_file = st.file_uploader("Upload grid-search CSV/JSON", type=["csv", "json"], key="parameter_sweep")
        sweep = parse_parameter_sweep_upload(sweep_file)
        required = {"entry_threshold", "exit_threshold", "window", "pnl", "sharpe"}
        if sweep.empty:
            st.info("Upload a CSV/JSON with entry_threshold, exit_threshold, window, pnl, sharpe.")
        elif not required.issubset(sweep.columns):
            st.error(f"Missing required columns: {', '.join(sorted(required - set(sweep.columns)))}")
        else:
            metric = st.selectbox("Heatmap metric", ["pnl", "sharpe"])
            windows = sorted(sweep["window"].dropna().unique())
            selected_window = st.selectbox("Window", windows)
            st.plotly_chart(plot_parameter_sweep_heatmap(sweep, metric, selected_window), use_container_width=True)
            st.dataframe(sweep, use_container_width=True, hide_index=True)

    elif view == "Submission Diff":
        if comparison is None:
            st.info("Upload an optional comparison log in the sidebar to see per-product PnL deltas.")
        else:
            diff = compare_product_pnl(comparison.activities, parsed.activities)
            st.plotly_chart(plot_pnl_diff(diff), use_container_width=True)
            st.dataframe(style_delta_table(diff), use_container_width=True, hide_index=True)
            st.caption("Delta is primary uploaded log minus comparison log.")

    elif view == "Runtime":
        threshold = st.number_input("IMC timeout threshold ms", min_value=1.0, value=DEFAULT_RUNTIME_TIMEOUT_MS, step=50.0)
        runtime = build_runtime_report(parsed.sandbox)
        if runtime.empty:
            st.info("No runtime or lambdaLog timestamp-gap data available.")
        else:
            st.plotly_chart(plot_runtime_report(runtime, threshold), use_container_width=True)
            st.dataframe(runtime.tail(50), use_container_width=True, hide_index=True)

    elif view == "Trader Flow":
        flow = build_trader_flow(trades, selected_products)
        if flow.empty:
            st.info("No counterparty flow available in the trade history.")
        else:
            counterparties = sorted(flow["counterparty"].dropna().unique())
            selected_counterparties = st.multiselect("Counterparties", counterparties, default=counterparties[: min(8, len(counterparties))])
            filtered = flow[flow["counterparty"].isin(selected_counterparties)] if selected_counterparties else flow
            st.plotly_chart(plot_trader_flow(filtered), use_container_width=True)
            summary = filtered.groupby(["counterparty", "product", "side"], as_index=False).agg(
                volume=("quantity", "sum"),
                trades=("quantity", "size"),
                avg_price=("price", "mean"),
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

    elif view == "Conversions":
        conversions = build_conversion_report(parsed.debug_lines)
        if conversions.empty:
            st.info("No conversion debug prints parsed. Expected lines like CONVERSION product=ORCHIDS qty=10 price=100 pnl=5.")
        else:
            if conversions["pnl"].notna().any():
                st.plotly_chart(plot_conversion_pnl(conversions), use_container_width=True)
            st.dataframe(conversions, use_container_width=True, hide_index=True)

    elif view == "Indicators":
        if indicators.empty:
            st.info("No JSON numeric indicators found in lambdaLog.")
        else:
            labels = sorted(indicators["label"].unique())
            selected = st.multiselect("Indicators", labels, default=labels[: min(5, len(labels))])
            if selected:
                st.plotly_chart(plot_indicators(indicators[indicators["label"].isin(selected)], selected), use_container_width=True)
            st.dataframe(indicators.drop(columns=["source_line"]), use_container_width=True, hide_index=True)

    elif view == "Logs":
        log_products = st.multiselect("Filter debug lines by product", products, default=[])
        render_debug_log(parsed.debug_lines, int(selected_timestamp) if selected_timestamp is not None else None, product_filter=log_products)
        sandbox_errors = parsed.sandbox[parsed.sandbox["sandboxLog"].astype(str).str.len() > 0]
        st.subheader("Sandbox Errors")
        st.dataframe(sandbox_errors, use_container_width=True, hide_index=True)

    elif view == "Raw Data":
        st.subheader("Activities")
        st.dataframe(activities[activities[PRODUCT_COLUMN].isin(selected_products)], use_container_width=True)
        st.subheader("Trade History")
        st.dataframe(trades[trades["symbol"].isin(selected_products)] if not trades.empty else trades)
        st.subheader("Parsed Order Intents")
        st.dataframe(parsed.order_intents, use_container_width=True, hide_index=True)
        csv = activities.to_csv(index=False).encode("utf-8")
        st.download_button("Download parsed activities CSV", csv, "parsed_activities.csv", "text/csv")


def main() -> None:
    """App entrypoint."""

    st.set_page_config(page_title="Prosperity Log Dashboard", layout="wide")
    if not require_password():
        st.stop()

    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB)))

    st.title("Prosperity Log Dashboard")
    st.caption("Upload a Prosperity backtest `.log` file. Files are parsed in memory and are not written to disk.")

    uploaded_file = st.file_uploader(
        "Upload primary backtest log",
        type=["log", "txt", "json"],
        accept_multiple_files=False,
        help=f"Maximum accepted size: {max_upload_mb} MB.",
    )
    comparison_file = st.file_uploader(
        "Optional comparison log",
        type=["log", "txt", "json"],
        accept_multiple_files=False,
        help="Used by the Submission Diff view.",
    )

    if uploaded_file is None:
        st.info("Upload a log to start.")
        return

    error = validate_upload(uploaded_file.name, uploaded_file.size, max_upload_mb)
    if error:
        st.error(error)
        return

    checksum, text = decode_upload(uploaded_file)
    with st.spinner("Parsing uploaded log..."):
        parsed = parse_uploaded_log(uploaded_file.name, checksum, text)

    comparison = None
    if comparison_file is not None:
        comparison_error = validate_upload(comparison_file.name, comparison_file.size, max_upload_mb)
        if comparison_error:
            st.error(comparison_error)
            return
        comparison_checksum, comparison_text = decode_upload(comparison_file)
        with st.spinner("Parsing comparison log..."):
            comparison = parse_uploaded_log(comparison_file.name, comparison_checksum, comparison_text)

    render_dashboard(parsed, comparison)


if __name__ == "__main__":
    main()

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
    PRODUCT_COLUMN,
    add_rolling_z_scores,
    build_fill_report,
    build_normalized_mid_series,
    build_pair_spreads,
    decompose_pnl,
    parse_backtest_text,
    plot_indicators,
    plot_normalized_mid,
    plot_orderbook,
    plot_pair_spreads,
    plot_pnl,
    plot_spreads,
    plot_z_scores,
    prepare_indicator_labels,
    render_debug_log,
)


DEFAULT_MAX_UPLOAD_MB = 25
ALLOWED_SUFFIXES = {".log", ".txt"}


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
        return "Only .log and .txt files are accepted."
    if size > max_upload_mb * 1024 * 1024:
        return f"File is too large. Limit is {max_upload_mb} MB."
    return None


@st.cache_data(show_spinner=False, max_entries=4, ttl=60 * 60)
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


def render_dashboard(parsed) -> None:
    """Render dashboard controls and visualizations for one parsed log."""

    with st.sidebar:
        st.header("Controls")
        z_window = st.slider("Rolling z-score window", min_value=10, max_value=500, value=100, step=10)
        entry_threshold = st.number_input("Entry z-score threshold", value=2.0, min_value=0.1, step=0.1)
        exit_threshold = st.number_input("Exit z-score threshold", value=0.5, min_value=0.0, step=0.1)
        max_plot_points = st.slider("Max plotted timestamps", min_value=500, max_value=20000, value=5000, step=500)

    activities = add_rolling_z_scores(parsed.activities, z_window)
    trades = parsed.trades
    indicators = prepare_indicator_labels(parsed.indicators)

    if activities.empty:
        st.error("The uploaded log has no parseable Activities log section.")
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

    latest_total_pnl = activities.sort_values("timestamp").groupby(PRODUCT_COLUMN).tail(1)["profit_and_loss"].sum()
    own_fill_count = int(trades["is_own_trade"].sum()) if not trades.empty else 0

    metric_columns = st.columns(4)
    metric_columns[0].metric("Products", len(products))
    metric_columns[1].metric("Latest total PnL", f"{latest_total_pnl:,.1f}")
    metric_columns[2].metric("Own fills", f"{own_fill_count:,}")
    metric_columns[3].metric("Indicators", f"{len(indicators):,}")

    tab_market, tab_pnl, tab_fills, tab_indicators, tab_logs, tab_raw = st.tabs(
        ["Market", "PnL", "Fill Rate", "Indicators", "Logs", "Raw Data"]
    )

    with tab_market:
        st.plotly_chart(plot_orderbook(activities, trades, selected_products, max_plot_points), use_container_width=True)
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
        st.dataframe(
            decompose_pnl(activities[activities[PRODUCT_COLUMN].isin(selected_products)], trades),
            use_container_width=True,
            hide_index=True,
        )

    with tab_fills:
        fill_report = build_fill_report(parsed, selected_products)
        if fill_report.empty:
            st.info("No fill diagnostics available.")
        else:
            st.dataframe(fill_report, use_container_width=True, hide_index=True)
        if parsed.order_intents.empty:
            st.warning(
                "No structured order submission logs found. For exact fill rate, print lines like "
                "`ORDER product=ASH_COATED_OSMIUM side=BUY price=9999 qty=10`."
            )

    with tab_indicators:
        if indicators.empty:
            st.info("No JSON numeric indicators found in lambdaLog.")
        else:
            labels = sorted(indicators["label"].unique())
            selected = st.multiselect("Indicators", labels, default=labels[: min(5, len(labels))])
            if selected:
                st.plotly_chart(plot_indicators(indicators, selected), use_container_width=True)
            st.dataframe(indicators.drop(columns=["source_line"]), use_container_width=True, hide_index=True)

    with tab_logs:
        render_debug_log(parsed.debug_lines, int(selected_timestamp) if selected_timestamp is not None else None)
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
        if not activities.empty:
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
        "Upload backtest log",
        type=["log", "txt"],
        accept_multiple_files=False,
        help=f"Maximum accepted size: {max_upload_mb} MB.",
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

    render_dashboard(parsed)


if __name__ == "__main__":
    main()

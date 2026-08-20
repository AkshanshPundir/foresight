import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent


# ============================================================
# FILE PATHS
# ============================================================

SALES_FILE = ROOT / "sales_sample.csv"
RISK_FILE = ROOT / "risk_scores.csv"
FORECAST_FILE = ROOT / "forecast_6_weeks.csv"
METRICS_FILE = ROOT / "metrics.json"
SUMMARY_FILE = ROOT / "summary.json"


# ============================================================
# LOAD DATA
# ============================================================

sales = pd.read_csv(SALES_FILE)
risk = pd.read_csv(RISK_FILE)
forecast = pd.read_csv(FORECAST_FILE)


# ============================================================
# LOAD JSON FILES
# ============================================================

with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

with open(SUMMARY_FILE, "r") as f:
    summary = json.load(f)


# ============================================================
# DATA PREPARATION
# ============================================================

sales["date"] = pd.to_datetime(
    sales["date"],
    errors="coerce"
)

sales["total_value"] = pd.to_numeric(
    sales["total_value"],
    errors="coerce"
)

sales["quantity"] = pd.to_numeric(
    sales["quantity"],
    errors="coerce"
)

sales = sales.dropna(
    subset=["date", "total_value"]
)


# Risk data
if "category" in risk.columns:
    risk["category"] = risk["category"].fillna("Unknown")

if "action" in risk.columns:
    risk["action"] = risk["action"].fillna("Unknown")


# ============================================================
# HEADER
# ============================================================

st.title("📦 Project FORESIGHT")

st.caption(
    "Demand & Inventory Intelligence — NorthBay Living"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")


# Category filter
if "category" in risk.columns:

    categories = (
        ["All"]
        + sorted(
            risk["category"]
            .astype(str)
            .unique()
            .tolist()
        )
    )

    category = st.sidebar.selectbox(
        "Category",
        categories,
        key="main_category_filter"
    )

else:

    category = "All"


# Apply category filter
if category != "All" and "category" in risk.columns:

    risk_view = risk[
        risk["category"] == category
    ].copy()

    selected_skus = risk_view["sku_id"].tolist()

    fc_view = forecast[
        forecast["sku_id"].isin(selected_skus)
    ].copy()

else:

    risk_view = risk.copy()
    fc_view = forecast.copy()


# ============================================================
# KPI SECTION
# ============================================================

c1, c2, c3, c4 = st.columns(4)


revenue = summary.get("revenue", 0)
units = summary.get("units", 0)
stockout_risk = summary.get(
    "stockout_risk_skus",
    0
)
capital_locked = summary.get(
    "capital_locked",
    0
)


c1.metric(
    "Revenue",
    f"₹{revenue:,.0f}"
)

c2.metric(
    "Units Sold",
    f"{units:,.0f}"
)

c3.metric(
    "Reorder Now",
    f"{stockout_risk:,.0f}"
)

c4.metric(
    "Capital Locked",
    f"₹{capital_locked:,.0f}"
)


st.divider()


# ============================================================
# REVENUE + RISK ACTIONS
# ============================================================

left, right = st.columns(2)


# ---------------- REVENUE TREND ----------------

with left:

    st.subheader("Revenue Trend")

    monthly_sales = (
        sales
        .set_index("date")
        .resample("ME")["total_value"]
        .sum()
        .reset_index()
    )

    revenue_fig = px.line(
        monthly_sales,
        x="date",
        y="total_value",
        markers=True,
        labels={
            "date": "Date",
            "total_value": "Revenue"
        },
        title="Monthly Revenue"
    )

    st.plotly_chart(
        revenue_fig,
        use_container_width=True
    )


# ---------------- RISK ACTIONS ----------------

with right:

    st.subheader("Risk Actions")

    if "action" in risk_view.columns:

        action_counts = (
            risk_view["action"]
            .value_counts()
            .reset_index()
        )

        action_counts.columns = [
            "action",
            "count"
        ]

        risk_fig = px.bar(
            action_counts,
            x="action",
            y="count",
            labels={
                "action": "Recommended Action",
                "count": "SKUs"
            },
            title="Inventory Risk Actions"
        )

        st.plotly_chart(
            risk_fig,
            use_container_width=True
        )

    else:

        st.info(
            "Risk action information is unavailable."
        )


# ============================================================
# INVENTORY ACTION TABLE
# ============================================================

st.subheader(
    "Prioritised Inventory Actions"
)


display_columns = [
    "sku_id",
    "sku_name",
    "category",
    "stock_on_hand",
    "avg_weekly_demand",
    "weeks_cover",
    "stockout_risk",
    "overstock_risk",
    "action",
    "sales_at_risk",
    "capital_locked"
]


available_columns = [
    col
    for col in display_columns
    if col in risk_view.columns
]


st.dataframe(
    risk_view[available_columns].head(100),
    use_container_width=True
)


# ============================================================
# DEMAND FORECAST
# ============================================================

st.subheader(
    "Demand Forecast"
)


if not fc_view.empty:

    sku_list = sorted(
        fc_view["sku_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_sku = st.selectbox(
        "Select SKU",
        sku_list[:500],
        key="forecast_sku_selector"
    )


    # Historical demand
    historical = (
        sales[
            sales["sku_id"] == selected_sku
        ]
        .groupby(
            pd.Grouper(
                key="date",
                freq="W-SUN"
            )
        )["quantity"]
        .sum()
        .reset_index()
    )


    historical = historical.rename(
        columns={
            "date": "week",
            "quantity": "units"
        }
    )


    # Forecast data
    future = fc_view[
        fc_view["sku_id"] == selected_sku
    ].copy()


    if "forecast_week" in future.columns:

        future = future.rename(
            columns={
                "forecast_week": "week",
                "forecast_units": "units"
            }
        )


    # Combine
    historical_plot = historical[
        ["week", "units"]
    ]

    future_plot = future[
        ["week", "units"]
    ]


    chart_data = pd.concat(
        [
            historical_plot,
            future_plot
        ],
        ignore_index=True
    )


    chart_data["week"] = pd.to_datetime(
        chart_data["week"],
        errors="coerce"
    )


    chart_data = chart_data.sort_values(
        "week"
    )


    forecast_fig = px.line(
        chart_data,
        x="week",
        y="units",
        markers=True,
        labels={
            "week": "Week",
            "units": "Units"
        },
        title=f"Demand Forecast — SKU {selected_sku}"
    )


    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

else:

    st.warning(
        "No forecast data available for the selected category."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader(
    "Model Performance"
)


baseline_wape = metrics.get(
    "baseline_wape",
    0
)

model_wape = metrics.get(
    "model_wape",
    0
)

improvement = metrics.get(
    "improvement_pct",
    0
)


m1, m2, m3 = st.columns(3)


m1.metric(
    "Seasonal-Naive WAPE",
    f"{baseline_wape:.2f}%"
)

m2.metric(
    "Model WAPE",
    f"{model_wape:.2f}%"
)

m3.metric(
    "Improvement",
    f"{improvement:.1f}%"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Project FORESIGHT | Demand & Inventory Intelligence"
)

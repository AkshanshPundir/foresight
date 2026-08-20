import streamlit as st
import pandas as pd, json
from pathlib import Path
import pandas as pd
import json
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent

SALES_FILE = ROOT / "sales_sample.csv"
RISK_FILE = ROOT / "risk_scores.csv"
FORECAST_FILE = ROOT / "forecast_6_weeks.csv"
METRICS_FILE = ROOT / "metrics.json"
SUMMARY_FILE = ROOT / "summary.json"

sales = pd.read_csv(SALES_FILE)
risk = pd.read_csv(RISK_FILE)
forecast = pd.read_csv(FORECAST_FILE)

sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
sales["total_value"] = pd.to_numeric(
    sales["total_value"],
    errors="coerce"
)
sales = sales.dropna(subset=["date", "total_value"])

categories = ["All"] + sorted(
    risk["category"].dropna().astype(str).unique().tolist()
)

cat = st.sidebar.selectbox(
    "Category",
    categories,
    key="category_filter"
)

if cat != "All":
    skus = risk.loc[
        risk["category"] == cat,
        "sku_id"
    ].tolist()

    risk_view = risk[risk["category"] == cat]

    fc_view = forecast[
        forecast["sku_id"].isin(skus)
    ]
else:
    risk_view = risk
    fc_view = forecast
with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

with open(SUMMARY_FILE, "r") as f:
    summary = json.load(f)
    cat = st.sidebar.selectbox(
    "Category"
)

if cat != "All":
    skus = risk.loc[
        risk["category"] == cat,
        "sku_id"
    ].tolist()

    risk_view = risk[risk["category"] == cat]

    fc_view = forecast[
        forecast["sku_id"].isin(skus)
    ]
else:
    risk_view = risk
    fc_view = forecast

st.set_page_config(page_title="FORESIGHT",page_icon="📦",layout="wide")
st.title("📦 Project FORESIGHT")
st.caption("Demand & Inventory Intelligence — NorthBay Living")
if "category" in risk.columns:
    risk["category"] = risk["category"].fillna("Unknown")
cat = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(risk["category"].dropna().unique().tolist())
)

if cat != "All":
    skus = risk.loc[risk["category"] == cat, "sku_id"].tolist()
    risk_view = risk[risk["category"] == cat]
    fc_view = forecast[forecast["sku_id"].isin(skus)]
else:
    risk_view = risk
    fc_view = forecast

c1,c2,c3,c4=st.columns(4)
c1.metric("Revenue",f"₹{summary['revenue']:,.0f}")
c2.metric("Units Sold",f"{summary['units']:,}")
c3.metric("Reorder Now",f"{summary['stockout_risk_skus']:,}")
c4.metric("Capital Locked",f"₹{summary['capital_locked']:,.0f}")

st.divider()
left,right=st.columns(2)
with left:
   st.subheader("Revenue Trend")

monthly_sales = (
    sales
    .set_index("date")
    .resample("ME")["total_value"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_sales,
    x="date",
    y="total_value",
    labels={"total_value": "Revenue"}
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Risk Actions")

risk_view = risk.copy()

action_counts = (
    risk_view["action"]
    .value_counts()
    .reset_index()
)

action_counts.columns = ["action", "count"]

fig_risk = px.bar(
    action_counts,
    x="action",
    y="count",
    labels={
        "action": "Recommended Action",
        "count": "SKUs"
    },
    title="SKU Risk Actions"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)
st.subheader("Prioritised Inventory Actions")
st.dataframe(risk_view[['sku_id','sku_name','category','stock_on_hand','avg_weekly_demand','weeks_cover','stockout_risk','overstock_risk','action','sales_at_risk','capital_locked']].head(100),use_container_width=True)

st.subheader("Demand Forecast")
sel=st.selectbox("SKU",sorted(fc_view.sku_id.unique().tolist())[:500])
fh=fc_view[fc_view.sku_id==sel]
h=sales[sales.sku_id==sel].groupby(pd.Grouper(key='date',freq='W-SUN')).quantity.sum().reset_index()
chart=pd.concat([h.rename(columns={'date':'week','quantity':'units'})[['week','units']],fh.rename(columns={'forecast_week':'week','forecast_units':'units'})[['week','units']]]).sort_values('week')
fig=px.line(chart,x='week',y='units',markers=True); st.plotly_chart(fig,use_container_width=True)

st.subheader("Model Performance")
st.write(f"Seasonal-naive WAPE: **{metrics['baseline_wape']:.2f}%** | Model WAPE: **{metrics['model_wape']:.2f}%** | Improvement: **{metrics['improvement_pct']:.1f}%**")

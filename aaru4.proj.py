"""
Analytics Report & Insights Documentation
-----------------------------------------
Turns raw sales data into a professional, stakeholder-ready analytics report:

    Executive summary  ->  KPIs  ->  Findings with charts  ->  Insights
    ->  Prioritised, actionable recommendations  ->  Methodology & data quality

By default the script generates a reproducible, synthetic two-year retail sales
dataset (so it runs anywhere with no downloads). To analyse your own data, pass
a CSV with the required columns (see README):

    python analytics_report.py
    python analytics_report.py --csv my_sales.csv --company "Acme Ltd"
    python analytics_report.py --output reports --seed 7 --discount-cap 0.15

Outputs (in the output folder):
    analytics_report.html   self-contained report (open in a browser, print to PDF)
    insights.json           KPIs, insights and recommendations in machine-readable form
    cleaned_data.csv        the cleaned dataset used for the analysis
    figures/*.png           all charts as separate images
"""

import argparse
import base64
import calendar
import html
import io
import json
import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # render charts without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BLUE, GREEN, RED, ORANGE, GREY = "#2f5d8a", "#2e8b57", "#c0392b", "#e08e2b", "#8a8f98"

REQUIRED_COLUMNS = [
    "order_id", "order_date", "customer_id", "region", "category", "product",
    "segment", "channel", "units", "unit_price", "unit_cost", "discount", "returned",
]


# ======================================================================
# 1. DATA: generation, loading, cleaning
# ======================================================================
def generate_data(n_orders=12000, seed=42):
    """Create a realistic synthetic retail dataset with seasonality, growth,
    regional differences, channel shift, discounting and returns."""
    rng = np.random.default_rng(seed)

    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    month_weight = np.array([0.80, 0.80, 0.90, 0.90, 1.00, 0.95,
                             0.90, 0.95, 1.00, 1.10, 1.40, 1.60])
    t_day = np.linspace(0, 1, len(days))
    weights = month_weight[days.month - 1] * (1 + 0.25 * t_day)
    order_dates = rng.choice(days, size=n_orders, p=weights / weights.sum())
    t = np.asarray((pd.to_datetime(order_dates) - days[0]).days) / (len(days) - 1)

    catalog = [  # category, product, list price, cost ratio, popularity
        ("Technology", "Laptop", 900, 0.78, 0.06),
        ("Technology", "Monitor", 250, 0.70, 0.08),
        ("Technology", "Headset", 80, 0.55, 0.10),
        ("Technology", "Webcam", 60, 0.60, 0.07),
        ("Furniture", "Desk", 400, 0.72, 0.05),
        ("Furniture", "Chair", 220, 0.68, 0.08),
        ("Furniture", "Bookcase", 150, 0.70, 0.05),
        ("Office Supplies", "Paper", 20, 0.55, 0.12),
        ("Office Supplies", "Binders", 15, 0.50, 0.12),
        ("Office Supplies", "Pens", 8, 0.45, 0.12),
        ("Office Supplies", "Labels", 10, 0.48, 0.10),
    ]
    pop = np.array([c[4] for c in catalog])
    idx = rng.choice(len(catalog), size=n_orders, p=pop / pop.sum())
    category = np.array([catalog[i][0] for i in idx])
    product = np.array([catalog[i][1] for i in idx])
    base_price = np.array([catalog[i][2] for i in idx], dtype=float)
    cost_ratio = np.array([catalog[i][3] for i in idx])

    region = rng.choice(["East", "West", "North", "South"], size=n_orders,
                        p=[0.35, 0.25, 0.22, 0.18])
    p_online = 0.35 + 0.20 * t
    u = rng.random(n_orders)
    channel = np.where(u < 0.15, "Wholesale",
                       np.where(u < 0.15 + p_online, "Online", "Retail"))
    segment = rng.choice(["Consumer", "Corporate", "Small Business"], size=n_orders,
                         p=[0.5, 0.3, 0.2])

    # Discounts driven by a latent propensity score (wholesale, South, Nov/Dec higher)
    month = pd.to_datetime(order_dates).month.to_numpy()
    score = (rng.normal(0, 1, n_orders) + 0.5 * (channel == "Wholesale")
             + 0.4 * (region == "South") + 0.3 * np.isin(month, [11, 12]))
    thresholds = [-0.385, 0.126, 0.674, 1.126, 1.751]
    discount = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])[np.digitize(score, thresholds)]

    units = rng.integers(1, 6, n_orders) * np.where(channel == "Wholesale",
                                                    rng.integers(2, 5, n_orders), 1)
    unit_price = base_price * rng.normal(1, 0.04, n_orders)
    unit_cost = base_price * cost_ratio * rng.normal(1, 0.02, n_orders)

    return_prob = np.select([category == "Furniture", category == "Technology"], [0.09, 0.05], 0.02)
    returned = (rng.random(n_orders) < return_prob).astype(int)

    n_customers = 6000
    customer_weights = rng.pareto(1.6, n_customers) + 1
    customer_id = rng.choice([f"C{i:05d}" for i in range(1, n_customers + 1)], size=n_orders,
                             p=customer_weights / customer_weights.sum())

    df = pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(1, n_orders + 1)],
        "order_date": pd.to_datetime(order_dates), "customer_id": customer_id,
        "region": region, "category": category, "product": product,
        "segment": segment, "channel": channel, "units": units,
        "unit_price": unit_price.round(2), "unit_cost": unit_cost.round(2),
        "discount": discount, "returned": returned,
    })

    # Inject realistic data-quality problems so the cleaning step has work to do
    df.loc[rng.choice(n_orders, 120, replace=False), "discount"] = np.nan
    df.loc[rng.choice(n_orders, 60, replace=False), "region"] = np.nan
    df = pd.concat([df, df.sample(30, random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def clean_data(raw):
    """Validate, clean and enrich the data. Returns (dataframe, quality_log)."""
    log = []
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    log.append(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    bad = int(df["order_date"].isna().sum())
    if bad:
        df = df.dropna(subset=["order_date"])
        log.append(f"Removed {bad} rows with unreadable dates.")

    dups = int(df.duplicated(subset="order_id").sum())
    df = df.drop_duplicates(subset="order_id")
    log.append(f"Removed {dups} duplicate orders (same order_id).")

    for col in ["units", "unit_price", "unit_cost", "discount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid = int((df[["units", "unit_price", "unit_cost"]].isna().any(axis=1)
                   | (df["units"] <= 0) | (df["unit_price"] <= 0)).sum())
    if invalid:
        df = df[~(df[["units", "unit_price", "unit_cost"]].isna().any(axis=1)
                  | (df["units"] <= 0) | (df["unit_price"] <= 0))]
        log.append(f"Removed {invalid} rows with missing or non-positive units/prices/costs.")

    if df["discount"].max() > 1:  # discounts supplied as percentages
        df["discount"] = df["discount"] / 100
        log.append("Discount values looked like percentages; converted to fractions.")
    disc_missing = int(df["discount"].isna().sum())
    if disc_missing:
        df["discount"] = df["discount"].fillna(df.groupby("category")["discount"].transform("median"))
        df["discount"] = df["discount"].fillna(0)
        log.append(f"Filled {disc_missing} missing discounts with the category median.")

    for col in ["region", "category", "product", "segment", "channel"]:
        n = int(df[col].isna().sum())
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        if n:
            log.append(f"Filled {n} missing '{col}' values with 'Unknown'.")

    df["returned"] = (df["returned"].astype(str).str.strip().str.lower()
                      .isin(["1", "1.0", "true", "yes", "y"]).astype(int))

    # Financial measures. Returned orders are excluded from net revenue and net profit.
    df["gross_revenue"] = df["units"] * df["unit_price"] * (1 - df["discount"])
    df["cost"] = df["units"] * df["unit_cost"]
    df["profit"] = df["gross_revenue"] - df["cost"]
    df["net_revenue"] = df["gross_revenue"] * (1 - df["returned"])
    df["net_profit"] = df["profit"] * (1 - df["returned"])
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month

    q1, q3 = df["gross_revenue"].quantile([0.25, 0.75])
    n_out = int((df["gross_revenue"] > q3 + 1.5 * (q3 - q1)).sum())
    log.append(f"Flagged {n_out} high-value orders as statistical outliers (IQR rule); kept as valid large orders.")
    log.append(f"Final dataset: {len(df):,} orders from {df['order_date'].min():%d %b %Y} "
               f"to {df['order_date'].max():%d %b %Y}.")
    return df.reset_index(drop=True), log


# ======================================================================
# 2. HELPERS
# ======================================================================
def money(x):
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.1f}K"
    return f"{sign}${x:,.0f}"


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def safe_div(a, b):
    return a / b if b else 0.0


def known(g):
    """Drop the 'Unknown' bucket (filled-in missing labels) before ranking best/worst."""
    k = g.drop(index="Unknown", errors="ignore")
    return k if len(k) else g


def finish_fig(fig, name, outdir):
    """Save a figure to disk and return it as a base64 string for embedding."""
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name), dpi=140)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ======================================================================
# 3. ANALYSIS
# ======================================================================
def group_summary(df, col):
    g = df.groupby(col).agg(
        revenue=("net_revenue", "sum"), profit=("net_profit", "sum"),
        orders=("order_id", "count"), avg_discount=("discount", "mean"),
        return_rate=("returned", "mean"),
    )
    g["margin"] = g["profit"] / g["revenue"]
    g["share"] = g["revenue"] / g["revenue"].sum()
    return g.sort_values("revenue", ascending=False)


def compute_kpis(df):
    years = sorted(df["year"].unique())
    cur, prev = years[-1], (years[-2] if len(years) > 1 else None)

    def year_stats(y):
        d = df[df["year"] == y]
        rev, prof = d["net_revenue"].sum(), d["net_profit"].sum()
        return {"revenue": rev, "profit": prof, "margin": safe_div(prof, rev),
                "orders": len(d), "customers": d["customer_id"].nunique(),
                "aov": d["gross_revenue"].mean(), "return_rate": d["returned"].mean()}

    k = {"current_year": int(cur), "previous_year": int(prev) if prev else None,
         "current": year_stats(cur), "previous": year_stats(prev) if prev else None}
    if prev:
        c, p = k["current"], k["previous"]
        k["growth"] = {
            "revenue_pct": safe_div(c["revenue"] - p["revenue"], p["revenue"]),
            "profit_pct": safe_div(c["profit"] - p["profit"], p["profit"]),
            "margin_pp": (c["margin"] - p["margin"]) * 100,
            "orders_pct": safe_div(c["orders"] - p["orders"], p["orders"]),
            "aov_pct": safe_div(c["aov"] - p["aov"], p["aov"]),
            "return_rate_pp": (c["return_rate"] - p["return_rate"]) * 100,
        }
    return k


def run_analysis(df, discount_cap):
    A = {}
    A["kpis"] = compute_kpis(df)
    A["category"] = group_summary(df, "category")
    A["region"] = group_summary(df, "region")
    A["channel"] = group_summary(df, "channel")
    A["segment"] = group_summary(df, "segment")
    A["product"] = group_summary(df, "product")

    # Seasonality
    A["monthly"] = df.pivot_table(index="month", columns="year", values="net_revenue", aggfunc="sum")

    # Channel mix by year
    mix = df.pivot_table(index="year", columns="channel", values="net_revenue", aggfunc="sum")
    A["channel_mix"] = mix.div(mix.sum(axis=1), axis=0)

    # Discount bands
    bands = pd.cut(df["discount"], [-0.001, 0, 0.05, 0.10, 0.15, 0.20, 1.0],
                   labels=["No discount", "1-5%", "6-10%", "11-15%", "16-20%", "Over 20%"])
    dd = df.assign(band=bands)
    db = dd.groupby("band", observed=True).agg(
        orders=("order_id", "count"), revenue=("net_revenue", "sum"),
        profit=("net_profit", "sum"), loss_orders=("profit", lambda s: (s < 0).mean()))
    db["margin"] = db["profit"] / db["revenue"]
    db["order_share"] = db["orders"] / db["orders"].sum()
    A["discount_bands"] = db
    over = df[df["discount"] > discount_cap]
    A["discount_cap"] = discount_cap
    A["excess_discount_cost"] = float((over["units"] * over["unit_price"]
                                       * (over["discount"] - discount_cap)
                                       * (1 - over["returned"])).sum())
    A["over_cap_order_share"] = len(over) / len(df)

    # Product Pareto and loss-making products
    prod = A["product"].sort_values("revenue", ascending=False)
    prod["cum_share"] = prod["revenue"].cumsum() / prod["revenue"].sum()
    A["pareto"] = prod
    top_n = max(1, int(np.ceil(0.2 * len(prod))))
    A["top20_product_share"] = float(prod["revenue"].head(top_n).sum() / prod["revenue"].sum())
    A["top20_product_n"] = top_n
    A["loss_products"] = prod[prod["profit"] < 0]

    # Customers
    cust = df.groupby("customer_id").agg(revenue=("net_revenue", "sum"), orders=("order_id", "count"))
    cust = cust.sort_values("revenue", ascending=False)
    top10 = max(1, int(np.ceil(0.10 * len(cust))))
    A["customers"] = cust
    A["top10_customer_share"] = float(cust["revenue"].head(top10).sum() / cust["revenue"].sum())
    A["repeat_rate"] = float((cust["orders"] >= 2).mean())

    # Returns
    returned_rev = df[df["returned"] == 1].groupby("category")["gross_revenue"].sum()
    A["returned_revenue_by_category"] = returned_rev
    A["total_returned_revenue"] = float(returned_rev.sum())
    return A


# ======================================================================
# 4. CHARTS
# ======================================================================
def make_charts(df, A, outdir):
    charts = {}

    # 1. Monthly trend by year
    fig, ax = plt.subplots(figsize=(9, 4.2))
    palette = [GREY, BLUE, ORANGE, GREEN]
    for i, y in enumerate(A["monthly"].columns):
        ax.plot(A["monthly"].index, A["monthly"][y] / 1e3, marker="o", lw=2.2,
                color=palette[i % len(palette)], label=str(y))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(calendar.month_abbr[1:13])
    ax.set_ylabel("Net revenue ($K)")
    ax.set_title("Monthly Net Revenue by Year")
    ax.legend(title="Year")
    charts["trend"] = finish_fig(fig, "01_monthly_trend.png", outdir)

    # 2. Category revenue and margin
    cat = A["category"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(cat.index, cat["revenue"] / 1e6, color=BLUE)
    ax[0].set_title("Net Revenue by Category ($M)")
    ax[0].tick_params(axis="x", rotation=15)
    ax[1].bar(cat.index, cat["margin"] * 100, color=[GREEN if m >= cat["margin"].mean() else ORANGE for m in cat["margin"]])
    ax[1].set_title("Profit Margin by Category (%)")
    ax[1].tick_params(axis="x", rotation=15)
    for i, m in enumerate(cat["margin"]):
        ax[1].text(i, m * 100 + 0.3, f"{m * 100:.1f}%", ha="center", fontsize=9)
    charts["category"] = finish_fig(fig, "02_category.png", outdir)

    # 3. Region revenue and margin
    reg = A["region"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(reg.index, reg["revenue"] / 1e6, color=BLUE)
    ax[0].set_title("Net Revenue by Region ($M)")
    ax[1].bar(reg.index, reg["margin"] * 100, color=[GREEN if m >= reg["margin"].mean() else ORANGE for m in reg["margin"]])
    ax[1].set_title("Profit Margin by Region (%)")
    for i, m in enumerate(reg["margin"]):
        ax[1].text(i, m * 100 + 0.3, f"{m * 100:.1f}%", ha="center", fontsize=9)
    charts["region"] = finish_fig(fig, "03_region.png", outdir)

    # 4. Channel mix by year and margin by channel
    mix, ch = A["channel_mix"], A["channel"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    (mix * 100).plot(kind="bar", stacked=True, ax=ax[0], colormap="Blues", rot=0, edgecolor="white")
    ax[0].set_title("Channel Mix by Year (% of net revenue)")
    ax[0].set_xlabel("")
    ax[0].legend(title="", fontsize=8)
    ax[1].bar(ch.index, ch["margin"] * 100, color=[GREEN if m >= ch["margin"].mean() else ORANGE for m in ch["margin"]])
    ax[1].set_title("Profit Margin by Channel (%)")
    for i, m in enumerate(ch["margin"]):
        ax[1].text(i, m * 100 + 0.3, f"{m * 100:.1f}%", ha="center", fontsize=9)
    charts["channel"] = finish_fig(fig, "04_channel.png", outdir)

    # 5. Discount bands
    db = A["discount_bands"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(db.index.astype(str), db["margin"] * 100, color=[RED if m < 0 else (GREEN if m > 0.15 else ORANGE) for m in db["margin"]])
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_title("Profit Margin by Discount Level (%)")
    ax[0].tick_params(axis="x", rotation=25)
    ax[1].bar(db.index.astype(str), db["loss_orders"] * 100, color=RED)
    ax[1].set_title("Orders Sold at a Loss (%)")
    ax[1].tick_params(axis="x", rotation=25)
    charts["discount"] = finish_fig(fig, "05_discounts.png", outdir)

    # 6. Pareto of products
    par = A["pareto"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(par.index, par["revenue"] / 1e3, color=BLUE)
    ax.set_ylabel("Net revenue ($K)")
    ax.tick_params(axis="x", rotation=40)
    ax2 = ax.twinx()
    ax2.plot(par.index, par["cum_share"] * 100, color=ORANGE, marker="o")
    ax2.set_ylabel("Cumulative share (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(False)
    ax.set_title("Product Revenue and Cumulative Share (Pareto)")
    charts["pareto"] = finish_fig(fig, "06_product_pareto.png", outdir)

    # 7. Customer concentration and returns
    cust = A["customers"]["revenue"].sort_values(ascending=False).to_numpy()
    cum = np.cumsum(cust) / cust.sum()
    x = np.arange(1, len(cust) + 1) / len(cust) * 100
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, cum * 100, color=BLUE, lw=2)
    ax[0].plot([0, 100], [0, 100], "--", color=GREY)
    ax[0].set_xlabel("Customers ranked by revenue (%)")
    ax[0].set_ylabel("Cumulative revenue (%)")
    ax[0].set_title("Customer Revenue Concentration")
    cat = A["category"]
    ax[1].bar(cat.index, cat["return_rate"] * 100, color=[RED if r == cat["return_rate"].max() else GREY for r in cat["return_rate"]])
    ax[1].set_title("Return Rate by Category (%)")
    ax[1].tick_params(axis="x", rotation=15)
    for i, r in enumerate(cat["return_rate"]):
        ax[1].text(i, r * 100 + 0.1, f"{r * 100:.1f}%", ha="center", fontsize=9)
    charts["customers_returns"] = finish_fig(fig, "07_customers_returns.png", outdir)
    return charts


# ======================================================================
# 5. INSIGHTS AND RECOMMENDATIONS (generated from the numbers)
# ======================================================================
def build_insights(df, A):
    k = A["kpis"]
    cur_year, c = k["current_year"], k["current"]
    cat, reg, ch, db = A["category"], A["region"], A["channel"], A["discount_bands"]
    kc, kr, kch = known(cat), known(reg), known(ch)
    insights = []

    # 1. Growth and seasonality
    cur_monthly = A["monthly"][cur_year].dropna()
    q4_share = safe_div(cur_monthly[cur_monthly.index >= 10].sum(), cur_monthly.sum())
    peak_month = calendar.month_name[int(A["monthly"].mean(axis=1).idxmax())]
    if k["previous_year"]:
        g = k["growth"]["revenue_pct"]
        title = f"Revenue {'grew' if g >= 0 else 'declined'} year on year" + (", with a pronounced Q4 peak" if q4_share > 0.30 else "")
        growth_txt = (f"Net revenue reached {money(c['revenue'])} in {cur_year}, "
                      f"{'up' if g >= 0 else 'down'} {abs(g) * 100:.1f}% on {k['previous_year']}. ")
    else:
        title = "Revenue trend and seasonality"
        growth_txt = f"Net revenue reached {money(c['revenue'])} in {cur_year}. "
    insights.append({
        "key": "growth", "title": title, "chart": "trend",
        "finding": growth_txt + f"Sales peak in {peak_month}, and the final quarter delivers {q4_share * 100:.0f}% of annual revenue.",
        "so_what": "Staffing, stock and marketing spend should be planned around the busiest months. A stock-out or delay in the peak quarter has an outsized effect on the full-year result."})

    # 2. Category
    top_cat, low_cat, best_cat = kc["revenue"].idxmax(), kc["margin"].idxmin(), kc["margin"].idxmax()
    if top_cat == low_cat:
        cat_title = "The largest category is also the least profitable"
        cat_text = (f"{top_cat} is the largest category ({pct(cat.loc[top_cat, 'share'], 0)} of revenue) but also has the "
                    f"thinnest margin ({pct(cat.loc[top_cat, 'margin'])}). {best_cat} has the best margin ({pct(cat.loc[best_cat, 'margin'])}).")
        cat_sowhat = f"Growth in {top_cat} adds a lot of sales but comparatively little profit, so pricing and discount discipline there matter more than extra volume."
    else:
        cat_title = "Revenue and profitability come from different categories"
        cat_text = (f"{top_cat} is the largest category at {pct(cat.loc[top_cat, 'share'], 0)} of revenue "
                    f"({pct(cat.loc[top_cat, 'margin'])} margin). {best_cat} has the best margin "
                    f"({pct(cat.loc[best_cat, 'margin'])}), while {low_cat} has the thinnest "
                    f"({pct(cat.loc[low_cat, 'margin'])}) on {pct(cat.loc[low_cat, 'share'], 0)} of revenue.")
        cat_sowhat = f"Growth in {low_cat} adds sales but little profit, so pricing and discount discipline there matter more than volume."
    insights.append({"key": "category", "title": cat_title, "chart": "category",
                     "finding": cat_text, "so_what": cat_sowhat})

    # 3. Region
    best_reg, worst_reg = kr["margin"].idxmax(), kr["margin"].idxmin()
    gap = (reg.loc[best_reg, "margin"] - reg.loc[worst_reg, "margin"]) * 100
    overall_disc = df["discount"].mean()
    disc_note = ""
    if reg.loc[worst_reg, "avg_discount"] > overall_disc + 0.01:
        disc_note = (f" {worst_reg}'s average discount ({pct(reg.loc[worst_reg, 'avg_discount'])}) is above the company average "
                     f"({pct(overall_disc)}), a likely contributor.")
    insights.append({
        "key": "region", "title": "Regional margins vary noticeably", "chart": "region",
        "finding": (f"{kr['revenue'].idxmax()} is the largest region ({pct(kr['share'].max(), 0)} of revenue). "
                    f"Margin ranges from {pct(reg.loc[worst_reg, 'margin'])} in {worst_reg} to "
                    f"{pct(reg.loc[best_reg, 'margin'])} in {best_reg}, a gap of {gap:.1f} percentage points." + disc_note),
        "so_what": f"Closing even part of the gap in {worst_reg} is a direct profit opportunity that does not need extra sales."})

    # 4. Channel
    mix = A["channel_mix"]
    first_y, last_y = mix.index.min(), mix.index.max()
    best_ch, worst_ch = kch["margin"].idxmax(), kch["margin"].idxmin()
    shift = ""
    if "Online" in mix.columns and first_y != last_y:
        shift = (f"Online's share of revenue moved from {pct(mix.loc[first_y, 'Online'], 0)} in {first_y} to "
                 f"{pct(mix.loc[last_y, 'Online'], 0)} in {last_y}. ")
    insights.append({
        "key": "channel", "title": "The sales channel mix is shifting" if shift else "Channel profitability differs", "chart": "channel",
        "finding": shift + f"{best_ch} earns the highest margin ({pct(ch.loc[best_ch, 'margin'])}) and {worst_ch} the lowest ({pct(ch.loc[worst_ch, 'margin'])}).",
        "so_what": f"Channel strategy should weigh profitability, not just volume. Growth that comes mainly through {worst_ch} dilutes overall margin."})

    # 5. Discounts
    first_band, last_band = db.index[0], db.index[-1]
    erodes = db.loc[last_band, "margin"] < db.loc[first_band, "margin"]
    overall_loss = (df["profit"] < 0).mean()
    insights.append({
        "key": "discount", "title": "Heavier discounting erodes margin" if erodes else "Discounting shows little margin impact", "chart": "discount",
        "finding": (f"Margin moves from {pct(db.loc[first_band, 'margin'])} on '{first_band}' orders to "
                    f"{pct(db.loc[last_band, 'margin'])} on '{last_band}' orders, where "
                    f"{pct(db.loc[last_band, 'loss_orders'], 0)} of orders lose money. Overall, {pct(overall_loss, 1)} of orders are sold at a loss. "
                    f"{pct(A['over_cap_order_share'], 0)} of orders carry a discount above {pct(A['discount_cap'], 0)}."),
        "so_what": (f"Discount spend above a {pct(A['discount_cap'], 0)} cap totals about {money(A['excess_discount_cost'])}, "
                    "making discount discipline one of the most direct levers for improving profit.")})

    # 6. Concentration and returns
    top_ret = kc["return_rate"].idxmax()
    one_time = 1 - A["repeat_rate"]
    insights.append({
        "key": "concentration", "title": "Revenue is concentrated, and returns are a hidden cost", "chart": "customers_returns",
        "finding": (f"The top 10% of customers generate {pct(A['top10_customer_share'], 0)} of revenue, and "
                    f"{pct(one_time, 0)} of customers have ordered only once. The overall return rate is {pct(c['return_rate'])}, "
                    f"highest in {top_ret} ({pct(cat.loc[top_ret, 'return_rate'])}). Returned orders removed {money(A['total_returned_revenue'])} of revenue."),
        "so_what": "Protecting top accounts and converting one-time buyers into repeat customers lifts revenue reliably, while reducing returns recovers revenue already won."})

    # 7. Product concentration
    insights.append({
        "key": "products", "title": "A small set of products drives most revenue", "chart": "pareto",
        "finding": (f"The top {A['top20_product_n']} products ({pct(A['top20_product_n'] / len(A['pareto']), 0)} of the range) "
                    f"account for {pct(A['top20_product_share'], 0)} of revenue. "
                    + (f"{len(A['loss_products'])} product(s) lose money overall: {', '.join(html.escape(p) for p in A['loss_products'].index)}."
                       if len(A["loss_products"]) else "No product loses money overall.")),
        "so_what": "Range decisions should focus on the few products that carry the business, and any loss-making items need repricing, cost reduction or removal."})
    return insights


def build_recommendations(df, A, discount_cap):
    cat, ch, reg = A["category"], A["channel"], A["region"]
    kc, kr, kch = known(cat), known(reg), known(ch)
    recs = []
    low_cat = kc["margin"].idxmin()
    recs.append({
        "priority": "High", "action": f"Cap discounts at {pct(discount_cap, 0)} and require approval above it",
        "rationale": f"{pct(A['over_cap_order_share'], 0)} of orders exceed the cap, and the deepest discounts push margin toward or below zero.",
        "impact": f"Up to {money(A['excess_discount_cost'])} of discount spend is above the cap (upper bound, before any volume loss).",
        "kpi": "Average discount, share of loss-making orders, profit margin"})

    if len(A["loss_products"]):
        loss_total = A["loss_products"]["profit"].sum()
        names = ", ".join(html.escape(p) for p in A["loss_products"].index)
        recs.append({
            "priority": "High", "action": f"Reprice or review loss-making products ({names})",
            "rationale": "These products lose money after discounts and costs.",
            "impact": f"Combined net loss of {money(abs(loss_total))} to recover.",
            "kpi": "Product-level margin"})
    else:
        recs.append({
            "priority": "High", "action": f"Protect margin in {low_cat}",
            "rationale": f"{low_cat} has the thinnest category margin ({pct(cat.loc[low_cat, 'margin'])}) on {pct(cat.loc[low_cat, 'share'], 0)} of revenue.",
            "impact": f"Each 1 point of margin in {low_cat} is worth about {money(cat.loc[low_cat, 'revenue'] * 0.01)}.",
            "kpi": f"{low_cat} margin"})

    best_ch, worst_ch = kch["margin"].idxmax(), kch["margin"].idxmin()
    if best_ch != worst_ch:
        total_rev = df["net_revenue"].sum()
        delta = 0.05 * total_rev * (ch.loc[best_ch, "margin"] - ch.loc[worst_ch, "margin"])
        recs.append({
            "priority": "Medium", "action": f"Steer growth toward {best_ch} and review {worst_ch} pricing",
            "rationale": f"{best_ch} margin is {pct(ch.loc[best_ch, 'margin'])} versus {pct(ch.loc[worst_ch, 'margin'])} for {worst_ch}.",
            "impact": f"Illustrative: moving 5% of revenue from {worst_ch} to {best_ch} adds about {money(delta)} of profit.",
            "kpi": "Channel mix, margin by channel"})

    worst_reg = kr["margin"].idxmin()
    recs.append({
        "priority": "Medium", "action": f"Run a margin improvement plan for {worst_reg}",
        "rationale": f"{worst_reg} has the lowest regional margin ({pct(reg.loc[worst_reg, 'margin'])}).",
        "impact": f"Each 1 point of margin in {worst_reg} is worth about {money(reg.loc[worst_reg, 'revenue'] * 0.01)}.",
        "kpi": f"{worst_reg} margin and average discount"})

    cur_monthly = A["monthly"][A["kpis"]["current_year"]].dropna()
    q4 = safe_div(cur_monthly[cur_monthly.index >= 10].sum(), cur_monthly.sum())
    recs.append({
        "priority": "Medium", "action": "Plan inventory, staffing and campaigns ahead of the peak quarter",
        "rationale": f"The final quarter delivers {pct(q4, 0)} of annual revenue.",
        "impact": "Avoids lost sales from stock-outs in the most valuable quarter.",
        "kpi": "Stock availability, peak-quarter revenue vs plan"})

    top_ret = kc["return_rate"].idxmax()
    ret_rev = A["returned_revenue_by_category"].get(top_ret, 0)
    recs.append({
        "priority": "Medium", "action": f"Reduce returns in {top_ret}",
        "rationale": f"{top_ret} has the highest return rate ({pct(cat.loc[top_ret, 'return_rate'])}).",
        "impact": f"{money(ret_rev)} of {top_ret} revenue was returned; better product information, packaging and quality checks could recover part of it.",
        "kpi": "Return rate by category"})

    one_time = 1 - A["repeat_rate"]
    recs.append({
        "priority": "Low", "action": "Launch a retention programme for top customers and one-time buyers",
        "rationale": f"The top 10% of customers drive {pct(A['top10_customer_share'], 0)} of revenue, and {pct(one_time, 0)} of customers ordered only once.",
        "impact": "Higher repeat purchase rate and lower dependence on a few accounts.",
        "kpi": "Repeat customer rate, top-10% revenue share"})
    return recs


def executive_summary(df, A, recs):
    k = A["kpis"]
    c, cur = k["current"], k["current_year"]
    bullets = []
    if k["previous_year"]:
        g = k["growth"]
        bullets.append(f"{cur} net revenue was {money(c['revenue'])} ({g['revenue_pct'] * 100:+.1f}% vs {k['previous_year']}) "
                       f"and net profit {money(c['profit'])} ({g['profit_pct'] * 100:+.1f}%), at a {pct(c['margin'])} margin "
                       f"({g['margin_pp']:+.1f} pts).")
    else:
        bullets.append(f"{cur} net revenue was {money(c['revenue'])} and net profit {money(c['profit'])}, at a {pct(c['margin'])} margin.")
    kc = known(A["category"])
    top_cat, low_cat = kc["revenue"].idxmax(), kc["margin"].idxmin()
    if top_cat == low_cat:
        bullets.append(f"{top_cat} is the largest category but also has the thinnest margin ({pct(kc['margin'].min())}).")
    else:
        bullets.append(f"{top_cat} leads on revenue, but {low_cat} has the thinnest margin ({pct(kc['margin'].min())}).")
    bullets.append(f"Discounting is a major profit leak: about {money(A['excess_discount_cost'])} of discount spend over the period analysed sits above a {pct(A['discount_cap'], 0)} cap.")
    bullets.append(f"Revenue is concentrated: the top 10% of customers drive {pct(A['top10_customer_share'], 0)} of sales.")
    top_actions = [r["action"] for r in recs[:3]]
    return bullets, top_actions


# ======================================================================
# 6. HTML REPORT
# ======================================================================
CSS = """
:root{--ink:#1d2733;--muted:#5d6b7a;--line:#e3e8ee;--accent:#2f5d8a;--bg:#f6f8fb;--card:#ffffff}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);margin:0;line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding:32px 20px 60px}
header{background:var(--accent);color:#fff;padding:34px 20px}
header .wrap{padding:0 20px}
header h1{margin:0 0 6px;font-size:28px}
header p{margin:0;opacity:.9}
h2{margin:38px 0 12px;font-size:22px;border-bottom:2px solid var(--line);padding-bottom:6px}
h3{margin:0 0 8px;font-size:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.card .value{font-size:24px;font-weight:700;margin:2px 0}
.card .delta{font-size:13px}
.up{color:#1e7a46}.down{color:#b83227}.flat{color:var(--muted)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0}
.finding{margin:14px 0 0;padding:14px 16px;background:#eef4fa;border-left:4px solid var(--accent);border-radius:6px}
.sowhat{margin:10px 0 0;padding:12px 16px;background:#f3f9f4;border-left:4px solid #2e8b57;border-radius:6px}
.tag{font-weight:700}
img{max-width:100%;height:auto;display:block;margin:10px auto;border:1px solid var(--line);border-radius:8px;background:#fff}
table{width:100%;border-collapse:collapse;font-size:14px;background:var(--card)}
th,td{border:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}
th{background:#eef2f7}
.pri-High{color:#b83227;font-weight:700}.pri-Medium{color:#b9770e;font-weight:700}.pri-Low{color:#5d6b7a;font-weight:700}
ul{padding-left:20px}
footer{color:var(--muted);font-size:13px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px}
@media print{body{background:#fff}.panel,.card{break-inside:avoid}header{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def kpi_card(label, value, delta=None, good_when_up=True, fmt="{:+.1f}%"):
    delta_html = ""
    if delta is not None:
        cls = "flat" if abs(delta) < 1e-9 else ("up" if (delta > 0) == good_when_up else "down")
        arrow = "&#9650;" if delta > 0 else ("&#9660;" if delta < 0 else "&#8211;")
        delta_html = f'<div class="delta {cls}">{arrow} {fmt.format(delta)}</div>'
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div>{delta_html}</div>'


def build_html(df, A, insights, recs, charts, quality_log, company, data_source):
    k = A["kpis"]
    c, cur = k["current"], k["current_year"]
    g = k.get("growth")
    bullets, top_actions = executive_summary(df, A, recs)
    prev_label = f"vs {k['previous_year']}" if k["previous_year"] else ""

    cards = [
        kpi_card(f"Net revenue {cur}", money(c["revenue"]), g["revenue_pct"] * 100 if g else None),
        kpi_card(f"Net profit {cur}", money(c["profit"]), g["profit_pct"] * 100 if g else None),
        kpi_card("Profit margin", pct(c["margin"]), g["margin_pp"] if g else None, fmt="{:+.1f} pts"),
        kpi_card("Orders", f"{c['orders']:,}", g["orders_pct"] * 100 if g else None),
        kpi_card("Avg order value", money(c["aov"]), g["aov_pct"] * 100 if g else None),
        kpi_card("Return rate", pct(c["return_rate"]), g["return_rate_pp"] if g else None,
                 good_when_up=False, fmt="{:+.1f} pts"),
    ]

    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>{html.escape(company)} Analytics Report</title><style>{CSS}</style></head><body>"]
    parts.append(f"<header><div class='wrap'><h1>{html.escape(company)}: Sales Analytics Report</h1>"
                 f"<p>Period: {df['order_date'].min():%d %b %Y} to {df['order_date'].max():%d %b %Y} &nbsp;|&nbsp; "
                 f"Prepared {datetime.now():%d %b %Y} &nbsp;|&nbsp; Audience: business stakeholders</p></div></header>")
    parts.append("<div class='wrap'>")

    # Executive summary
    parts.append("<h2>1. Executive Summary</h2>")
    parts.append(f"<div class='cards'>{''.join(cards)}</div>")
    if prev_label:
        parts.append(f"<p style='color:var(--muted);font-size:13px;margin-top:-6px'>Changes shown {prev_label}.</p>")
    parts.append("<div class='panel'><h3>Headlines</h3><ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
    parts.append("<h3>Top recommended actions</h3><ol>" + "".join(f"<li>{a}</li>" for a in top_actions) + "</ol></div>")

    # Findings
    parts.append("<h2>2. Key Findings and Insights</h2>")
    for i, ins in enumerate(insights, 1):
        parts.append(f"<div class='panel'><h3>2.{i} {html.escape(ins['title'])}</h3>"
                     f"<img alt='{html.escape(ins['title'])}' src='data:image/png;base64,{charts[ins['chart']]}'>"
                     f"<div class='finding'><span class='tag'>Finding:</span> {ins['finding']}</div>"
                     f"<div class='sowhat'><span class='tag'>Why it matters:</span> {ins['so_what']}</div></div>")

    # Recommendations
    parts.append("<h2>3. Recommendations</h2>")
    parts.append("<table><tr><th>#</th><th>Priority</th><th>Action</th><th>Rationale</th><th>Expected impact</th><th>KPI to track</th></tr>")
    for i, r in enumerate(recs, 1):
        parts.append(f"<tr><td>{i}</td><td class='pri-{r['priority']}'>{r['priority']}</td><td>{r['action']}</td>"
                     f"<td>{r['rationale']}</td><td>{r['impact']}</td><td>{r['kpi']}</td></tr>")
    parts.append("</table>")
    parts.append(f"<p style='color:var(--muted);font-size:13px'>Impact figures cover the whole period analysed ({df['order_date'].min():%b %Y} to {df['order_date'].max():%b %Y}), "
                 "are estimates from historical data and assume other factors stay constant.</p>")

    # Methodology
    parts.append("<h2>4. Methodology and Data Quality</h2><div class='panel'>")
    parts.append(f"<p><b>Data source:</b> {html.escape(data_source)}</p><p><b>Definitions:</b></p><ul>"
                 "<li><b>Net revenue</b> = units x list price x (1 - discount), excluding returned orders.</li>"
                 "<li><b>Net profit</b> = net revenue minus product cost, excluding returned orders.</li>"
                 "<li><b>Margin</b> = net profit / net revenue.</li>"
                 "<li><b>Year-on-year</b> compares the latest full calendar year in the data with the one before it.</li></ul>")
    parts.append("<p><b>Data preparation log:</b></p><ul>" + "".join(f"<li>{html.escape(l)}</li>" for l in quality_log) + "</ul></div>")

    parts.append("<h2>5. Limitations</h2><div class='panel'><ul>"
                 "<li>Findings show association, not proven cause; for example, discounts may be higher in situations that were already low-margin.</li>"
                 "<li>Impact estimates are upper bounds or illustrations and ignore changes in customer behaviour.</li>"
                 "<li>Returned orders are assumed to lose all revenue and profit; restocking recovery is not modelled.</li>"
                 "<li>Results depend on the quality and completeness of the underlying data.</li></ul></div>")
    parts.append(f"<footer>Generated automatically by analytics_report.py &middot; {datetime.now():%d %b %Y %H:%M}</footer></div></body></html>")
    return "".join(parts)


# ======================================================================
# MAIN
# ======================================================================
def to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    return o


def main():
    parser = argparse.ArgumentParser(description="Generate a stakeholder analytics report")
    parser.add_argument("--csv", help="Path to your sales CSV (default: synthetic demo data)")
    parser.add_argument("--output", default="output", help="Output folder (default: output)")
    parser.add_argument("--company", default="Northwind Retail", help="Company name shown in the report")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the synthetic data")
    parser.add_argument("--discount-cap", type=float, default=0.20, help="Discount cap used in the scenario (default 0.20)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.csv:
        print(f"Loading data from {args.csv} ...")
        raw = pd.read_csv(args.csv)
        source = f"User-supplied file: {os.path.basename(args.csv)}"
    else:
        print("Generating synthetic demo data ...")
        raw = generate_data(seed=args.seed)
        source = "Synthetic, reproducible demo dataset (retail orders, 2024 to 2025). Replace with real data using --csv."

    df, log = clean_data(raw)
    for line in log:
        print("  -", line)

    A = run_analysis(df, args.discount_cap)
    charts = make_charts(df, A, args.output)
    insights = build_insights(df, A)
    recs = build_recommendations(df, A, args.discount_cap)

    report_html = build_html(df, A, insights, recs, charts, log, args.company, source)
    report_path = os.path.join(args.output, "analytics_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    df.to_csv(os.path.join(args.output, "cleaned_data.csv"), index=False)
    bullets, top_actions = executive_summary(df, A, recs)
    payload = {"kpis": A["kpis"], "executive_summary": bullets, "top_actions": top_actions,
               "insights": [{k: v for k, v in i.items() if k != "chart"} for i in insights],
               "recommendations": recs, "data_preparation": log}
    with open(os.path.join(args.output, "insights.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)

    print("\nEXECUTIVE SUMMARY")
    for b in bullets:
        print("  *", b)
    print("\nTOP RECOMMENDATIONS")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. [{r['priority']}] {r['action']}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

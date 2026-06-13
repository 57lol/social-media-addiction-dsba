# -*- coding: utf-8 -*-
"""
Web version of the Social Media Addiction report (Streamlit), pink-beige pastel theme.
Equivalent to the Jupyter notebook: all sections, plots and findings, plus interactivity.

Run:  streamlit run app/streamlit_app.py      (from the project root)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from scipy import stats
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data import load_clean_data

# ── pastel theme ──
PASTEL = {"blush": "#E7A6B0", "rose": "#D98AA0", "mauve": "#C98BAE", "plum": "#9B6A8F",
          "beige": "#E4D2B8", "taupe": "#CBB293", "cream": "#FBF3EC", "ink": "#6E4B5E"}
pastel_seq = ["#E7A6B0", "#D98AA0", "#C98BAE", "#E4D2B8", "#CBB293", "#9B6A8F"]
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.prop_cycle": cycler(color=pastel_seq),
    "figure.facecolor": PASTEL["cream"], "axes.facecolor": PASTEL["cream"],
    "savefig.facecolor": PASTEL["cream"], "axes.edgecolor": PASTEL["taupe"],
    "axes.titlecolor": PASTEL["ink"], "axes.labelcolor": PASTEL["ink"],
    "xtick.color": PASTEL["ink"], "ytick.color": PASTEL["ink"],
    "axes.titleweight": "bold", "grid.color": "#EAD9CC", "figure.dpi": 110,
})

st.set_page_config(page_title="Digital Addiction — report", page_icon="🌸", layout="wide")
st.markdown(f"""
<style>
.stApp {{ background-color: {PASTEL['cream']}; }}
section[data-testid="stSidebar"] {{ background-color: #F5DCE4; }}
h1, h2, h3 {{ color: {PASTEL['ink']}; }}
[data-testid="stMetricValue"] {{ color: {PASTEL['plum']}; }}
.stButton>button, .stFormSubmitButton>button {{
    background-color: {PASTEL['rose']}; color: white; border: none; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data…")
def get_data():
    return load_clean_data()


data = get_data()


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


st.sidebar.title("🌸 Digital Addiction")
st.sidebar.caption("Final project · Python for Data Science DSBA")
SECTION = st.sidebar.radio("Report sections", [
    "Abstract",
    "1. Dataset description",
    "2. Descriptive statistics",
    "3. Data cleanup",
    "4. Feature engineering",
    "5. Single-variable plots",
    "6. Detailed comparisons",
    "7. Hypothesis testing",
    "8. Conclusions",
    "🔎 Explore",
    "➕ Predict a user",
])

# ─────────────────────────────────────────────────────────────────────────
if SECTION == "Abstract":
    st.title("🌸 Digital Addiction: What Really Drives a Social-Media Addiction Score?")
    st.markdown("""
**Course:** Python for Data Science (DSBA), 2025/2026 — final project
**Dataset:** Social-media usage & digital addiction (10,000 users, 23 fields)

> **Authors:** _Anya & Eva_ — put your names and group here.

### Abstract
We use a dataset of **10,000 social-media users** to ask what actually drives a person's
**addiction score** — time on TikTok and Instagram, night-time scrolling, age, or sleep. We run a
full pipeline: data description and quality, descriptive statistics, a (short) cleanup, feature
engineering, visualisations and **three hypotheses tested with `scipy`**.

The headline: **behaviour beats demographics**. Daily social-media time is by far the strongest
correlate of addiction (TikTok more than Instagram), while age and sleep duration are *unrelated*
to the addiction score. **Contribution:** Anya — data & statistics; Eva — visualisation & testing.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", f"{len(data):,}")
    c2.metric("Avg addiction score", f"{data['addiction_score'].mean():.1f}")
    c3.metric("Avg daily time", f"{data['daily_hours'].mean():.1f} h")

elif SECTION == "1. Dataset description":
    st.header("1. Dataset description")
    st.markdown("""
**Domain:** digital well-being / social-media behaviour. Each row is one user: daily TikTok and
Instagram minutes, behavioural indices (night usage, scroll velocity, dopamine dependency,
attention span, impulsivity), sleep, demographics, and a final **`addiction_score`** plus a
categorical **`addiction_level`** (Low / Medium / High / Severe).
""")
    st.dataframe(data.head(20), width="stretch")
    st.subheader("Data quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Missing values", int(data.isna().sum().sum()))
    c3.metric("Duplicate rows", int(data.duplicated().sum()))
    st.success("The dataset is clean: no missing values, no duplicates, correct types.")

elif SECTION == "2. Descriptive statistics":
    st.header("2. Descriptive statistics")
    fields = ["tiktok_minutes_daily", "instagram_minutes_daily", "sleep_hours",
              "addiction_score", "night_usage_ratio", "attention_span_score"]
    desc = data[fields].describe().T
    desc["median"] = data[fields].median()
    st.dataframe(desc[["count", "mean", "median", "std", "min", "25%", "75%", "max"]].round(2),
                 width="stretch")
    st.markdown("""
Users average ~121 min/day on TikTok and ~100 min/day on Instagram. Sleep centres at ~7 h, the
addiction score around 58, and night usage is spread uniformly between 0 and 1.
""")

elif SECTION == "3. Data cleanup":
    st.header("3. Data cleanup")
    st.markdown("""
The data is already clean, so we *show* it is clean, set `addiction_level` as an ordered category
and drop the `user_id` identifier (implemented in `app/data.py`, shared with the notebook).
""")
    st.success(f"0 missing · 0 duplicates · {len(data):,} rows × {data.shape[1]} columns.")

elif SECTION == "4. Feature engineering":
    st.header("4. Feature engineering (new columns)")
    st.markdown("""
- **`total_minutes`** = TikTok + Instagram minutes
- **`daily_hours`** = total_minutes / 60
- **`tiktok_share`** = TikTok / total (how TikTok-dominated)
- **`night_minutes`** = total_minutes × night_usage_ratio
- **`age_group`** = `<25` / `25–35` / `35+`
- **`heavy_user`** = above-median total time
- **`level_num`** = ordinal Low=1 … Severe=4
""")
    st.dataframe(data[["tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes",
                       "daily_hours", "tiktok_share", "night_minutes", "age_group",
                       "heavy_user", "level_num"]].head(15), width="stretch")

elif SECTION == "5. Single-variable plots":
    st.header("5. Single-variable plots (5 fields, 3 types)")
    st.subheader("Histograms")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    specs = [("tiktok_minutes_daily", "TikTok minutes/day", PASTEL["rose"]),
             ("instagram_minutes_daily", "Instagram minutes/day", PASTEL["mauve"]),
             ("addiction_score", "Addiction score", PASTEL["plum"]),
             ("sleep_hours", "Sleep hours", PASTEL["taupe"])]
    for ax, (col, title, color) in zip(axes.flat, specs):
        sns.histplot(data[col], bins=40, kde=True, color=color, ax=ax,
                     edgecolor="white", linewidth=0.4)
        ax.axvline(data[col].mean(), color=PASTEL["ink"], ls="--", lw=1.2)
        ax.set_title(title); ax.set_xlabel("")
    show(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Box plot")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=data, x="addiction_level", y="total_minutes",
                    order=["Low", "Medium", "High", "Severe"], palette=pastel_seq, ax=ax)
        ax.set_xlabel("Addiction level"); ax.set_ylabel("Total minutes/day")
        show(fig)
    with c2:
        st.subheader("Scatter + regression")
        s = data.sample(3000, random_state=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.regplot(data=s, x="total_minutes", y="addiction_score", ax=ax,
                    scatter_kws=dict(alpha=0.35, s=14, color=PASTEL["rose"]),
                    line_kws=dict(color=PASTEL["plum"], lw=2))
        ax.set_xlabel("Total minutes/day"); ax.set_ylabel("Addiction score")
        show(fig)

elif SECTION == "6. Detailed comparisons":
    st.header("6. Detailed comparisons & relationships")
    st.subheader("Correlation matrix")
    corr_cols = ["tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes",
                 "night_usage_ratio", "night_minutes", "sleep_hours", "age", "addiction_score"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(data[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdPu",
                vmin=-1, vmax=1, linewidths=0.5, linecolor="white", ax=ax)
    show(fig)

    st.subheader("TikTok vs Instagram effect on addiction")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, col, title, color in zip(axes,
            ["tiktok_minutes_daily", "instagram_minutes_daily"], ["TikTok", "Instagram"],
            [PASTEL["rose"], PASTEL["mauve"]]):
        s = data.sample(3000, random_state=2)
        sns.regplot(data=s, x=col, y="addiction_score", ax=ax,
                    scatter_kws=dict(alpha=0.3, s=14, color=color),
                    line_kws=dict(color=PASTEL["plum"], lw=2))
        r, p = stats.pearsonr(data[col], data["addiction_score"])
        ax.set_title(f"{title}: r = {r:.3f}")
        ax.set_xlabel(f"{title} minutes/day")
    axes[0].set_ylabel("Addiction score")
    show(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Addiction by age group")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=data, x="age_group", y="addiction_score",
                    order=["<25", "25-35", "35+"], palette=pastel_seq, ax=ax)
        ax.set_xlabel("Age group"); ax.set_ylabel("Addiction score")
        show(fig)
    with c2:
        st.subheader("Minutes vs addiction by level")
        s = data.sample(4000, random_state=3)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(data=s, x="total_minutes", y="addiction_score",
                        hue="addiction_level", hue_order=["Low", "Medium", "High", "Severe"],
                        palette=pastel_seq[:4], alpha=0.6, s=18, ax=ax)
        ax.set_xlabel("Total minutes/day"); ax.set_ylabel("Addiction score")
        show(fig)

    st.subheader("Behavioural profile by addiction level")
    profile = data.groupby("addiction_level", observed=True).agg(
        n=("addiction_level", "size"), total_minutes=("total_minutes", "mean"),
        night_minutes=("night_minutes", "mean"), sleep_hours=("sleep_hours", "mean"),
        age=("age", "mean")).round(1)
    st.dataframe(profile, width="stretch")

elif SECTION == "7. Hypothesis testing":
    st.header("7. Hypothesis testing")
    st.caption("α = 0.05, tests via scipy.stats")

    st.subheader("H1 — Time drives addiction; TikTok > Instagram")
    r_tt = stats.pearsonr(data["tiktok_minutes_daily"], data["addiction_score"])[0]
    r_ig = stats.pearsonr(data["instagram_minutes_daily"], data["addiction_score"])[0]
    r_tot = stats.pearsonr(data["total_minutes"], data["addiction_score"])[0]
    z = lambda x: (x - x.mean()) / x.std()
    X = np.column_stack([np.ones(len(data)), z(data["tiktok_minutes_daily"]),
                         z(data["instagram_minutes_daily"])])
    beta, *_ = np.linalg.lstsq(X, z(data["addiction_score"]).values, rcond=None)
    c1, c2, c3 = st.columns(3)
    c1.metric("r TikTok", f"{r_tt:.3f}")
    c2.metric("r Instagram", f"{r_ig:.3f}")
    c3.metric("r Total", f"{r_tot:.3f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=["TikTok", "Instagram", "Total"], y=[r_tt, r_ig, r_tot], palette=pastel_seq, ax=ax)
    ax.set_ylabel("Pearson r"); ax.set_ylim(0, 1)
    show(fig)
    st.success(f"H1 confirmed: both significant (p≪0.05); standardised β — TikTok {beta[1]:.2f} > "
               f"Instagram {beta[2]:.2f}. TikTok matters more.")

    st.subheader("H2 — Heavy users more addicted, in every age group")
    rows = []
    for g, sub in data.groupby("age_group", observed=True):
        heavy = sub[sub["heavy_user"] == 1]["addiction_score"]
        light = sub[sub["heavy_user"] == 0]["addiction_score"]
        t, p = stats.ttest_ind(heavy, light, equal_var=False)
        rows.append([g, round(light.mean(), 1), round(heavy.mean(), 1),
                     round(heavy.mean() - light.mean(), 1), p])
    h2 = pd.DataFrame(rows, columns=["age_group", "light_mean", "heavy_mean", "gap", "p_value"])
    st.dataframe(h2, width="stretch")
    plot_df = data.groupby(["age_group", "heavy_user"], observed=True)["addiction_score"].mean().reset_index()
    plot_df["user"] = plot_df["heavy_user"].map({0: "Light", 1: "Heavy"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=plot_df, x="age_group", y="addiction_score", hue="user",
                order=["<25", "25-35", "35+"], palette=[PASTEL["beige"], PASTEL["rose"]], ax=ax)
    ax.set_ylabel("Mean addiction score"); ax.set_ylim(0, 75)
    show(fig)
    st.success("H2 confirmed: heavy users score ~11 points higher in every age group (p≪0.05).")

    st.subheader("H3 — Age and sleep are NOT related to addiction")
    r_age, p_age = stats.pearsonr(data["age"], data["addiction_score"])
    r_sleep, p_sleep = stats.pearsonr(data["sleep_hours"], data["addiction_score"])
    c1, c2 = st.columns(2)
    c1.metric("r age ↔ addiction", f"{r_age:+.3f}", f"p={p_age:.2f}")
    c2.metric("r sleep ↔ addiction", f"{r_sleep:+.3f}", f"p={p_sleep:.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    s = data.sample(3000, random_state=4)
    for ax, col, title in zip(axes, ["age", "sleep_hours"], ["Age", "Sleep hours"]):
        sns.regplot(data=s, x=col, y="addiction_score", ax=ax,
                    scatter_kws=dict(alpha=0.25, s=12, color=PASTEL["taupe"]),
                    line_kws=dict(color=PASTEL["plum"], lw=2))
        ax.set_xlabel(title); ax.set_ylabel("Addiction score")
    show(fig)
    st.warning("H3 confirmed (null result): |r| ≈ 0.01, p ≫ 0.05 — age and sleep do not explain "
               "addiction. This debunks the common assumption.")

elif SECTION == "8. Conclusions":
    st.header("8. Conclusions")
    st.markdown("""
1. **Total social-media time is the dominant driver of addiction** (r ≈ 0.85).
2. **TikTok matters more than Instagram** (H1).
3. **Behaviour beats demographics** (H2): heavy users are more addicted in *every* age group.
4. **Age and sleep are unrelated to addiction** (H3) — a clean null result.
5. Usage-based behaviour (minutes, night scrolling) moves with addiction; sleep, age, country
   and GDP do not.

**Takeaway:** to predict digital addiction here, look at *how much* (and how late) someone scrolls —
especially on TikTok — not at their age or sleep.
""")

elif SECTION == "🔎 Explore":
    st.header("🔎 Interactive explorer")
    c1, c2, c3 = st.columns(3)
    levels = c1.multiselect("Addiction level", ["Low", "Medium", "High", "Severe"],
                            default=["Low", "Medium", "High", "Severe"])
    ages = c2.multiselect("Age group", ["<25", "25-35", "35+"], default=["<25", "25-35", "35+"])
    countries = c3.multiselect("Country (optional)", sorted(data["country"].unique()))
    mins = st.slider("Total daily minutes", 0, int(data["total_minutes"].max()),
                     (0, int(data["total_minutes"].max())), step=10)
    f = data[data["addiction_level"].isin(levels) & data["age_group"].isin(ages)
             & data["total_minutes"].between(mins[0], mins[1])]
    if countries:
        f = f[f["country"].isin(countries)]
    if len(f) == 0:
        st.warning("No users match the filter.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Users", f"{len(f):,}")
        m2.metric("Avg addiction", f"{f['addiction_score'].mean():.1f}")
        m3.metric("Avg minutes", f"{f['total_minutes'].mean():.0f}")
        m4.metric("Avg sleep", f"{f['sleep_hours'].mean():.1f} h")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(f["addiction_score"], bins=40, kde=True, color=PASTEL["rose"], ax=ax)
        ax.set_title("Addiction score in the selection"); ax.set_xlabel("Addiction score")
        show(fig)
        st.dataframe(f.head(50), width="stretch")

elif SECTION == "➕ Predict a user":
    st.header("➕ Predict a user's addiction score")
    st.caption("Same idea as POST /users: a simple linear model on total daily minutes.")
    slope, intercept = np.polyfit(data["total_minutes"], data["addiction_score"], 1)
    with st.form("new_user"):
        c1, c2, c3 = st.columns(3)
        tt = c1.number_input("TikTok minutes/day", 0, 600, 150, step=10)
        ig = c2.number_input("Instagram minutes/day", 0, 600, 90, step=10)
        age = c3.number_input("Age", 10, 100, 21)
        c4, c5 = st.columns(2)
        night = c4.slider("Night usage ratio", 0.0, 1.0, 0.5, step=0.05)
        sleep = c5.number_input("Sleep hours", 0.0, 24.0, 6.5, step=0.5)
        submitted = st.form_submit_button("Predict")
    if submitted:
        total = tt + ig
        pred = float(np.clip(intercept + slope * total, 0, 100))
        band = "Low" if pred < 40 else "Medium" if pred < 58 else "High" if pred < 70 else "Severe"
        c1, c2, c3 = st.columns(3)
        c1.metric("Total minutes/day", f"{total}")
        c2.metric("Predicted addiction score", f"{pred:.1f}")
        c3.metric("Predicted level", band)
        st.info(f"Daily time ≈ {total/60:.1f} h. Prediction is a simple linear estimate "
                f"from total minutes (the strongest correlate of addiction).")

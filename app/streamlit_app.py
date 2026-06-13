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

pink = "#D98AA0"
mauve = "#C98BAE"
plum = "#9B6A8F"
taupe = "#CBB293"
cream = "#FBF3EC"
ink = "#6E4B5E"
palette = ["#E7A6B0", "#D98AA0", "#C98BAE", "#E4D2B8", "#CBB293", "#9B6A8F"]

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = cycler(color=palette)
plt.rcParams["figure.facecolor"] = cream
plt.rcParams["axes.facecolor"] = cream
plt.rcParams["savefig.facecolor"] = cream

st.set_page_config(page_title="Digital addiction report", page_icon="🌸", layout="centered")
st.markdown("<style>.stApp{background-color:" + cream + ";}h1,h2,h3{color:" + ink + ";}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data...")
def get_data():
    return load_clean_data()


data = get_data()


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


st.title("Digital addiction: what is linked to a social media addiction score?")
st.write("Course: Python for Data Science (DSBA), 2025/2026. Authors: Mitricheva Anna and Romanovskaia Eva, group 251.")

st.header("Abstract")
st.write("This project looks at a dataset of 10000 social media users and tries to understand what is connected to a person's addiction score: the time they spend on TikTok and Instagram, their night usage, their age and their sleep. We describe the data, check its quality, compute basic statistics, clean and normalize it, add a few new columns, draw plots and finally test two hypotheses with scipy.")
st.write("What we found is that the time spent on social media is the main thing linked to the addiction score, and TikTok is connected to it a bit more than Instagram. Age and sleep are not related to the score at all. Contribution: Anna prepared and cleaned the data and wrote the statistics part, Eva made the plots and ran the hypothesis tests, and we wrote the text and conclusions together.")
c1, c2, c3 = st.columns(3)
c1.metric("Users", str(len(data)))
c2.metric("Average score", str(round(data["addiction_score"].mean(), 1)))
c3.metric("Average daily time", str(round(data["daily_hours"].mean(), 1)) + " h")

st.header("1. The dataset")
st.write("The data is about social media use and digital addiction. Each row is one user with their daily TikTok and Instagram minutes, some behaviour indices, sleep, age and country, and a final addiction score with a level from Low to Severe.")
st.dataframe(data.head(20), width="stretch")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", str(len(data)))
c2.metric("Missing values", int(data.isna().sum().sum()))
c3.metric("Duplicates", int(data.duplicated().sum()))
st.write("The dataset is clean: no missing values, no duplicates and correct types.")

st.header("2. Descriptive statistics")
fields = ["tiktok_minutes_daily", "instagram_minutes_daily", "sleep_hours", "addiction_score", "night_usage_ratio", "attention_span_score"]
desc = data[fields].describe().T
desc["median"] = data[fields].median()
st.dataframe(desc[["count", "mean", "median", "std", "min", "25%", "75%", "max"]].round(2), width="stretch")
st.write("On average users spend about 121 minutes a day on TikTok and about 100 on Instagram. Sleep is around 7 hours and the addiction score around 58.")

st.header("3. Data cleanup")
st.write("The data is already clean, so we mainly show that it is clean, turn addiction_level into an ordered category and drop the user_id column. This is done in app/data.py, which the notebook and the site share.")
st.success("0 missing values, 0 duplicates, " + str(len(data)) + " rows and " + str(data.shape[1]) + " columns.")

st.header("4. New columns")
st.write("We add total_minutes, daily_hours, tiktok_share, night_minutes, age_group, heavy_user and level_num, all built from the existing columns.")
st.dataframe(data[["total_minutes", "daily_hours", "tiktok_share", "night_minutes", "age_group", "heavy_user", "level_num"]].head(15), width="stretch")

st.header("5. Normalization")
st.write("The numeric fields are on very different scales, so we standardize the main numeric columns to mean 0 and standard deviation 1 and keep the original columns for the readable plots.")
zcols = [c + "_z" for c in ["tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes", "night_minutes", "sleep_hours", "addiction_score"]]
st.dataframe(data[zcols].describe().round(2).T[["mean", "std", "min", "max"]], width="stretch")

st.header("6. Plots of single fields")
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
specs = [("tiktok_minutes_daily", "TikTok minutes per day", pink), ("instagram_minutes_daily", "Instagram minutes per day", mauve), ("addiction_score", "Addiction score", plum), ("sleep_hours", "Sleep hours", taupe)]
for ax, (col, title, color) in zip(axes.flat, specs):
    sns.histplot(data[col], bins=40, kde=True, color=color, ax=ax, edgecolor="white", linewidth=0.4)
    ax.axvline(data[col].mean(), color=ink, ls="--", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("")
show(fig)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=data, x="addiction_level", y="total_minutes", order=["Low", "Medium", "High", "Severe"], palette=palette, ax=ax)
ax.set_xlabel("Addiction level")
ax.set_ylabel("Total minutes per day")
show(fig)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.regplot(data=data.sample(3000, random_state=1), x="total_minutes", y="addiction_score", scatter_kws={"alpha": 0.35, "s": 14, "color": pink}, line_kws={"color": plum, "lw": 2}, ax=ax)
ax.set_xlabel("Total minutes per day")
ax.set_ylabel("Addiction score")
show(fig)
st.write("TikTok and Instagram minutes are roughly bell shaped, the addiction score sits around 55 to 65 and sleep is symmetric around 7 hours. Total minutes rise sharply with the addiction level and the scatter shows a clear upward trend.")

st.header("7. Comparisons and relationships")
corr_cols = ["tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes", "night_usage_ratio", "night_minutes", "sleep_hours", "age", "addiction_score"]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(data[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdPu", vmin=-1, vmax=1, linewidths=0.5, linecolor="white", ax=ax)
show(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, col, title, color in zip(axes, ["tiktok_minutes_daily", "instagram_minutes_daily"], ["TikTok", "Instagram"], [pink, mauve]):
    sns.regplot(data=data.sample(3000, random_state=2), x=col, y="addiction_score", ax=ax, scatter_kws={"alpha": 0.3, "s": 14, "color": color}, line_kws={"color": plum, "lw": 2})
    r = stats.pearsonr(data[col], data["addiction_score"])[0]
    ax.set_title(title + ": r = " + str(round(r, 3)))
    ax.set_xlabel(title + " minutes per day")
axes[0].set_ylabel("Addiction score")
show(fig)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=data, x="age_group", y="addiction_score", order=["<25", "25-35", "35+"], palette=palette, ax=ax)
ax.set_xlabel("Age group")
ax.set_ylabel("Addiction score")
show(fig)
profile = data.groupby("addiction_level", observed=True).agg(n=("addiction_level", "size"), total_minutes=("total_minutes", "mean"), night_minutes=("night_minutes", "mean"), sleep_hours=("sleep_hours", "mean"), age=("age", "mean")).round(1)
st.dataframe(profile, width="stretch")
st.write("The addiction score is strongly linked to total minutes and to the TikTok and Instagram minutes, while sleep and age sit close to zero. TikTok has a higher correlation than Instagram, the age groups have almost the same average score, and higher addiction levels mostly mean a lot more daily and night minutes.")

st.header("8. Hypotheses")
st.write("We test two hypotheses with a plot and a statistical test from scipy, using a significance level of 0.05.")

st.subheader("Hypothesis 1: time drives addiction, and TikTok matters more than Instagram")
r_tt = stats.pearsonr(data["tiktok_minutes_daily"], data["addiction_score"])[0]
r_ig = stats.pearsonr(data["instagram_minutes_daily"], data["addiction_score"])[0]
r_tot = stats.pearsonr(data["total_minutes"], data["addiction_score"])[0]
X = np.column_stack([np.ones(len(data)), data["tiktok_minutes_daily_z"], data["instagram_minutes_daily_z"]])
beta = np.linalg.lstsq(X, data["addiction_score_z"].values, rcond=None)[0]
c1, c2, c3 = st.columns(3)
c1.metric("r TikTok", str(round(r_tt, 3)))
c2.metric("r Instagram", str(round(r_ig, 3)))
c3.metric("r Total", str(round(r_tot, 3)))
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=["TikTok", "Instagram", "Total"], y=[r_tt, r_ig, r_tot], palette=palette, ax=ax)
ax.set_ylabel("Pearson r")
ax.set_ylim(0, 1)
show(fig)
st.write("The hypothesis holds. Both apps have a strong and significant correlation with the score (p well below 0.05), and the regression gives TikTok a larger coefficient (" + str(round(beta[1], 2)) + ") than Instagram (" + str(round(beta[2], 2)) + "). So time drives the score and TikTok matters a bit more.")

st.subheader("Hypothesis 2: age and sleep are not related to the addiction score")
r_age, p_age = stats.pearsonr(data["age"], data["addiction_score"])
r_sleep, p_sleep = stats.pearsonr(data["sleep_hours"], data["addiction_score"])
c1, c2 = st.columns(2)
c1.metric("r age", str(round(r_age, 3)), "p = " + str(round(p_age, 2)))
c2.metric("r sleep", str(round(r_sleep, 3)), "p = " + str(round(p_sleep, 2)))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
s = data.sample(3000, random_state=4)
for ax, col, title in zip(axes, ["age", "sleep_hours"], ["Age", "Sleep hours"]):
    sns.regplot(data=s, x=col, y="addiction_score", ax=ax, scatter_kws={"alpha": 0.25, "s": 12, "color": taupe}, line_kws={"color": plum, "lw": 2})
    ax.set_xlabel(title)
    ax.set_ylabel("Addiction score")
show(fig)
st.write("The hypothesis holds. Both correlations are basically zero and not significant, and the regression lines are almost flat. So in this data age and sleep do not explain the addiction score.")

st.header("9. Conclusions")
st.write("1. The time spent on social media is the main thing linked to the addiction score, with the total daily minutes reaching a correlation of about 0.85.")
st.write("2. TikTok is connected to the score a little more than Instagram.")
st.write("3. Age and sleep are not related to the addiction score at all.")
st.write("4. Higher addiction levels mostly mean a lot more daily and night minutes, while sleep, age and country do not really matter here.")
st.write("So the useful signal for a person's addiction score is how much and how late they scroll, especially on TikTok, and not their age or how long they sleep.")

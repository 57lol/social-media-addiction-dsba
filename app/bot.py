# -*- coding: utf-8 -*-
"""
Telegram bot for the Social Media Addiction project (aiogram 3.x).

A menu with several pages giving access to any part of the project: dataset description,
descriptive statistics, plots (as images), stats by addiction level, hypotheses, conclusions,
and a form that predicts a new user's addiction score.

Token:
  • environment variable  ADDICTION_BOT_TOKEN, or
  • file  app/bot_token.txt  (one line with the @BotFather token).

Run:  python -m app.bot          (from the project root, on the Python that has aiogram)
"""
import os
import sys
import io
import asyncio
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from cycler import cycler
from scipy import stats

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (Message, CallbackQuery, BufferedInputFile,
                           InlineKeyboardButton, InlineKeyboardMarkup)
from aiogram.utils.keyboard import InlineKeyboardBuilder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data import load_clean_data

logging.basicConfig(level=logging.INFO)

PASTEL = {"rose": "#D98AA0", "mauve": "#C98BAE", "plum": "#9B6A8F",
          "beige": "#E4D2B8", "taupe": "#CBB293", "cream": "#FBF3EC", "ink": "#6E4B5E"}
pastel_seq = ["#E7A6B0", "#D98AA0", "#C98BAE", "#E4D2B8", "#CBB293", "#9B6A8F"]
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.prop_cycle": cycler(color=pastel_seq),
                     "figure.facecolor": PASTEL["cream"], "axes.facecolor": PASTEL["cream"],
                     "savefig.facecolor": PASTEL["cream"], "axes.titlecolor": PASTEL["ink"]})

DATA = load_clean_data()
_SLOPE, _INTERCEPT = np.polyfit(DATA["total_minutes"], DATA["addiction_score"], 1)
dp = Dispatcher(storage=MemoryStorage())


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def menu_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📋 About the dataset", callback_data="about")
    kb.button(text="📊 Descriptive statistics", callback_data="stats")
    kb.button(text="📈 Plots", callback_data="plots")
    kb.button(text="🏆 Stats by addiction level", callback_data="levels")
    kb.button(text="🔬 Hypotheses", callback_data="hypo")
    kb.button(text="📝 Conclusions", callback_data="conclusions")
    kb.button(text="➕ Predict a user", callback_data="predict")
    kb.adjust(1)
    return kb.as_markup()


def back_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Menu", callback_data="menu")]])


WELCOME = (
    "🌸 <b>Digital Addiction — data report</b>\n\n"
    "Final project · Python for Data Science (DSBA).\n"
    f"Dataset: <b>{len(DATA):,}</b> social-media users. "
    f"Average addiction score: <b>{DATA['addiction_score'].mean():.1f}</b>.\n\n"
    "Pick a section:"
)


@dp.message(Command("start", "menu", "help"))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(WELCOME, reply_markup=menu_kb())


@dp.callback_query(F.data == "menu")
async def cb_menu(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.answer(WELCOME, reply_markup=menu_kb())
    await cb.answer()


@dp.callback_query(F.data == "about")
async def cb_about(cb: CallbackQuery):
    txt = (
        "📋 <b>About the dataset</b>\n\n"
        "Domain: digital well-being / social-media behaviour. Each row is one user: daily TikTok and "
        "Instagram minutes, behavioural indices (night usage, dopamine dependency, attention span), "
        "sleep, demographics, and a final <b>addiction score</b> + level (Low/Medium/High/Severe).\n\n"
        f"<b>{len(DATA):,}</b> users · clean data (no missing values, no duplicates)."
    )
    await cb.message.answer(txt, reply_markup=back_kb())
    await cb.answer()


@dp.callback_query(F.data == "stats")
async def cb_stats(cb: CallbackQuery):
    fields = ["tiktok_minutes_daily", "instagram_minutes_daily", "sleep_hours",
              "addiction_score", "night_usage_ratio", "attention_span_score"]
    lines = ["📊 <b>Descriptive statistics</b>\n", "<pre>field            mean  median   std</pre>"]
    short = {"tiktok_minutes_daily": "tiktok_min", "instagram_minutes_daily": "instagram_min",
             "sleep_hours": "sleep_h", "addiction_score": "addiction", "night_usage_ratio": "night_ratio",
             "attention_span_score": "attention"}
    for c in fields:
        s = DATA[c]
        lines.append(f"<pre>{short[c]:13} {s.mean():7.1f} {s.median():7.1f} {s.std():6.1f}</pre>")
    await cb.message.answer("\n".join(lines), reply_markup=back_kb())
    await cb.answer()


@dp.callback_query(F.data == "plots")
async def cb_plots(cb: CallbackQuery):
    await cb.answer("Rendering plots…")

    # 1) addiction score distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(DATA["addiction_score"], bins=40, kde=True, color=PASTEL["rose"],
                 edgecolor="white", ax=ax)
    ax.set_title("Addiction score distribution"); ax.set_xlabel("Addiction score")
    await cb.message.answer_photo(BufferedInputFile(fig_to_png(fig), "p1.png"),
                                  caption="Addiction score is concentrated around 55–65.")

    # 2) tiktok vs instagram effect
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = DATA.sample(3000, random_state=2)
    sns.regplot(data=s, x="total_minutes", y="addiction_score",
                scatter_kws=dict(alpha=0.3, s=12, color=PASTEL["mauve"]),
                line_kws=dict(color=PASTEL["plum"], lw=2), ax=ax)
    r = stats.pearsonr(DATA["total_minutes"], DATA["addiction_score"])[0]
    ax.set_title(f"Total minutes vs addiction (r = {r:.2f})")
    ax.set_xlabel("Total minutes/day"); ax.set_ylabel("Addiction score")
    await cb.message.answer_photo(BufferedInputFile(fig_to_png(fig), "p2.png"),
                                  caption="More daily minutes → higher addiction (strong link).")

    # 3) correlation heatmap
    cols = ["tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes",
            "night_minutes", "sleep_hours", "age", "addiction_score"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(DATA[cols].corr(), annot=True, fmt=".2f", cmap="RdPu", vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Correlations")
    await cb.message.answer_photo(BufferedInputFile(fig_to_png(fig), "p3.png"),
                                  caption="Sleep and age sit near zero — unrelated to addiction.",
                                  reply_markup=back_kb())


@dp.callback_query(F.data == "levels")
async def cb_levels(cb: CallbackQuery):
    g = DATA.groupby("addiction_level", observed=True).agg(
        n=("addiction_level", "size"), minutes=("total_minutes", "mean"),
        sleep=("sleep_hours", "mean"))
    lines = ["🏆 <b>By addiction level</b>\n", "<pre>level    n     minutes  sleep</pre>"]
    for lvl, row in g.iterrows():
        lines.append(f"<pre>{str(lvl):7} {int(row['n']):5d}  {row['minutes']:7.0f}  {row['sleep']:4.1f}</pre>")
    await cb.message.answer("\n".join(lines), reply_markup=back_kb())
    await cb.answer()


@dp.callback_query(F.data == "hypo")
async def cb_hypo(cb: CallbackQuery):
    await cb.answer("Computing tests…")
    r_tt = stats.pearsonr(DATA["tiktok_minutes_daily"], DATA["addiction_score"])[0]
    r_ig = stats.pearsonr(DATA["instagram_minutes_daily"], DATA["addiction_score"])[0]
    r_age, p_age = stats.pearsonr(DATA["age"], DATA["addiction_score"])
    txt = (
        "🔬 <b>Hypotheses</b> (α = 0.05)\n\n"
        f"<b>H1.</b> Time drives addiction; TikTok &gt; Instagram.\n"
        f"→ r(TikTok)={r_tt:.2f} &gt; r(Instagram)={r_ig:.2f}. <b>Confirmed.</b>\n\n"
        "<b>H2.</b> Heavy users are more addicted in every age group.\n"
        "→ ~11-point gap, p≪0.05 in all groups. <b>Confirmed.</b>\n\n"
        "<b>H3.</b> Age and sleep are NOT related to addiction.\n"
        f"→ r(age)={r_age:+.2f}, p={p_age:.2f}. <b>Confirmed (null result).</b>"
    )
    await cb.message.answer(txt, reply_markup=back_kb())


@dp.callback_query(F.data == "conclusions")
async def cb_conclusions(cb: CallbackQuery):
    txt = (
        "📝 <b>Conclusions</b>\n\n"
        "1. Total social-media time is the dominant driver of addiction (r ≈ 0.85).\n"
        "2. TikTok matters more than Instagram (H1).\n"
        "3. Behaviour beats demographics (H2): heavy users are more addicted at every age.\n"
        "4. Age and sleep are unrelated to addiction (H3).\n\n"
        "<i>To predict digital addiction here, look at how much (and how late) someone scrolls — "
        "especially on TikTok — not at age or sleep.</i>"
    )
    await cb.message.answer(txt, reply_markup=back_kb())
    await cb.answer()


class NewUser(StatesGroup):
    tiktok = State()
    instagram = State()
    age = State()


@dp.callback_query(F.data == "predict")
async def cb_predict(cb: CallbackQuery, state: FSMContext):
    await state.set_state(NewUser.tiktok)
    await cb.message.answer("➕ <b>Predict a user</b>\n\nEnter TikTok minutes per day (e.g. 150):")
    await cb.answer()


async def _num(message: Message, lo, hi):
    try:
        v = float(message.text.replace(",", ".").strip())
    except ValueError:
        await message.answer("Please send a number:")
        return None
    if not (lo <= v <= hi):
        await message.answer(f"Value must be between {lo} and {hi}. Try again:")
        return None
    return v


@dp.message(NewUser.tiktok)
async def u_tiktok(message: Message, state: FSMContext):
    v = await _num(message, 0, 600)
    if v is None:
        return
    await state.update_data(tiktok=v)
    await state.set_state(NewUser.instagram)
    await message.answer("Enter Instagram minutes per day (e.g. 90):")


@dp.message(NewUser.instagram)
async def u_instagram(message: Message, state: FSMContext):
    v = await _num(message, 0, 600)
    if v is None:
        return
    await state.update_data(instagram=v)
    await state.set_state(NewUser.age)
    await message.answer("Enter age (e.g. 21):")


@dp.message(NewUser.age)
async def u_age(message: Message, state: FSMContext):
    v = await _num(message, 10, 100)
    if v is None:
        return
    d = await state.get_data()
    await state.clear()
    total = d["tiktok"] + d["instagram"]
    pred = float(np.clip(_INTERCEPT + _SLOPE * total, 0, 100))
    band = "Low" if pred < 40 else "Medium" if pred < 58 else "High" if pred < 70 else "Severe"
    txt = (
        "✅ <b>Prediction</b>\n\n"
        f"TikTok: <b>{d['tiktok']:.0f}</b> min · Instagram: <b>{d['instagram']:.0f}</b> min\n"
        f"Total: <b>{total:.0f}</b> min/day ({total/60:.1f} h) · Age: <b>{v:.0f}</b>\n\n"
        f"🌸 Predicted addiction score: <b>{pred:.1f}</b>\n"
        f"📊 Predicted level: <b>{band}</b>\n\n"
        "<i>Simple linear estimate from total minutes — the strongest correlate.</i>"
    )
    await message.answer(txt, reply_markup=back_kb())


def get_token() -> str:
    token = os.getenv("ADDICTION_BOT_TOKEN", "").strip()
    if not token:
        path = os.path.join(os.path.dirname(__file__), "bot_token.txt")
        if os.path.exists(path):
            token = open(path, encoding="utf-8").read().strip()
    return token


async def main():
    token = get_token()
    if not token:
        print("\n[!] No bot token found.\n"
              "    1) Get a token from @BotFather in Telegram.\n"
              "    2) Either:  export ADDICTION_BOT_TOKEN=YOUR_TOKEN\n"
              "       or create  app/bot_token.txt  with the token inside.\n"
              "    3) Run again:  python -m app.bot\n")
        sys.exit(1)
    bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    print("Bot is running. Open it in Telegram and press /start. Ctrl+C to stop.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass

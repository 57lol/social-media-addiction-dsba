# 🌸 Digital Addiction — Data Science Project

Final project for **Python for Data Science (DSBA), 2025/2026**.
Dataset: **social-media usage & digital addiction** (10,000 users, 23 fields).

We investigate what really drives a user's **addiction score** — time on TikTok/Instagram,
night-time scrolling, age, or sleep. Full pipeline: data description & quality, descriptive
statistics, cleanup, feature engineering, visualisations and **three hypotheses tested with scipy**.

> ⚠️ **Before submitting:** open the notebook and write your **names and group** in the first
> cell (Abstract). Do the same in the Streamlit app (Abstract section).

---

## What's inside

```
Social_Media_Addiction_Project/
├── Social_Media_Addiction.ipynb   ← MAIN report notebook (executed, with plots)
├── build_notebook.py              ← regenerates the notebook
├── data/data.csv                  ← dataset (10,000 rows × 23 columns)
├── app/
│   ├── data.py                    ← shared loading + feature engineering
│   ├── streamlit_app.py           ← web report (pink-beige theme, = the notebook)
│   ├── api.py                     ← REST API (FastAPI): GET filters + POST predict
│   └── bot.py                     ← Telegram bot (menu over the project)
├── run_web.command                ← launch the web report (double-click)
├── run_api.command                ← launch the REST API (double-click)
├── run_bot.command                ← launch the Telegram bot (double-click)
├── requirements.txt               ← web app deps (also used by Streamlit Cloud)
├── requirements-notebook.txt      ← deps to re-run the notebook
└── requirements-bot.txt           ← deps for the Telegram bot
```

Course requirements covered (all in the notebook): abstract + contribution · dataset description &
quality · descriptive statistics (mean/median/std for 6 fields) · cleanup · 5 numeric fields in 3
plot types · 6+ comparison outputs · feature engineering (7 new columns) · **3 hypotheses tested
with scipy** · discussion at every step.

---

## 1. The notebook (main deliverable)

It is already executed — just open it:

```bash
jupyter lab Social_Media_Addiction.ipynb     # or: jupyter notebook
```

To re-run it from scratch: `python3 build_notebook.py`.

## 2. Web report + REST API

Just **double-click** the launchers (they set up everything on first run):
- `run_web.command` → web report at **http://localhost:8501**
- `run_api.command` → REST API + docs at **http://localhost:8000/docs**

From a terminal instead:
```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app/streamlit_app.py     # web
./.venv/bin/python -m uvicorn app.api:app --port 8000        # API
```

REST API examples:
```bash
curl "http://localhost:8000/users?level=High&min_age=18&max_age=25&limit=5"
curl "http://localhost:8000/stats?group_by=addiction_level&metric=avg_total_minutes"
curl -X POST "http://localhost:8000/users" -H "Content-Type: application/json" \
     -d '{"age":21,"tiktok_minutes_daily":150,"instagram_minutes_daily":90}'
```

## 3. Telegram bot

A menu with pages over the whole project + a "predict a user" form.

1. Get a token from **@BotFather** (`/newbot`).
2. Save it: `echo "YOUR_TOKEN" > app/bot_token.txt`
3. Double-click `run_bot.command`, then open the bot and press **/start**.

## 4. Publish the web report online (optional, for the bonus)

Push this folder to a public GitHub repo, then on **https://share.streamlit.io** sign in with
GitHub → *Create app* → pick the repo, branch `main`, main file `app/streamlit_app.py` → **Deploy**.
Streamlit Cloud installs from `requirements.txt` automatically.

---

*Built with pandas, matplotlib/seaborn and scipy. Pastel theme 🌸.*

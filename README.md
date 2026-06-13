# Digital addiction project

Final project for Python for Data Science (DSBA), 2025/2026.
Authors: Mitricheva Anna and Romanovskaia Eva, group 251.

Dataset: social media usage and digital addiction, 10000 users and 23 fields.
We study what is linked to a user's addiction score: time on TikTok and Instagram, night usage, age and sleep.

## Files

```
Social_Media_Addiction_Project/
  Social_Media_Addiction.ipynb   the main report notebook (already run, with plots)
  build_notebook.py              the script that builds the notebook
  data/data.csv                  the dataset
  app/data.py                    loading, cleaning and new columns
  app/streamlit_app.py           the web version of the report
  app/api.py                     a small REST API (FastAPI)
  run_web.command                start the web report
  run_api.command                start the REST API
  requirements.txt               packages for the web app
  requirements-notebook.txt      packages to re-run the notebook
```

## How to open the notebook

It is already run, so just open it:

```
jupyter lab Social_Media_Addiction.ipynb
```

To build it again from scratch: `python3 build_notebook.py`.

## How to run the web report

Double click `run_web.command`. On the first run it sets up everything by itself, then opens at http://localhost:8501.

From a terminal instead:

```
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app/streamlit_app.py
```

## REST API

Double click `run_api.command`, then open http://localhost:8000/docs. Examples:

```
curl "http://localhost:8000/users?level=High&min_age=18&max_age=25&limit=5"
curl "http://localhost:8000/stats?group_by=addiction_level&metric=avg_total_minutes"
curl -X POST "http://localhost:8000/users" -H "Content-Type: application/json" -d '{"age":21,"tiktok_minutes_daily":150,"instagram_minutes_daily":90}'
```

## Put the web report online

Push this folder to a public GitHub repo, then on https://share.streamlit.io sign in with GitHub, pick the repo, branch main and main file app/streamlit_app.py, and deploy. Streamlit installs the packages from requirements.txt by itself.

#!/bin/bash
# Double-click to launch the Telegram bot.
# Needs a token from @BotFather: put it in app/bot_token.txt (see README).
cd "$(dirname "$0")" || exit 1

# pick a Python that already has aiogram, otherwise create a dedicated bot env
if python3 -c "import aiogram" 2>/dev/null; then
  PYBIN="python3"
elif [ -x ".venv_bot/bin/python" ]; then
  PYBIN=".venv_bot/bin/python"
else
  echo "🤖 First run: setting up the bot environment (a few minutes)…"
  python3 -m venv .venv_bot \
    && ./.venv_bot/bin/python -m pip install -q --upgrade pip \
    && ./.venv_bot/bin/python -m pip install -q -r requirements-bot.txt
  PYBIN=".venv_bot/bin/python"
fi
echo "🤖 Starting Telegram bot…  (Ctrl+C to stop)"
exec "$PYBIN" -m app.bot

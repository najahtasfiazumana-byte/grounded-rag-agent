# Setup (Windows, Anaconda, VS Code)

Same steps as the other projects. Commands are for the VS Code terminal (PowerShell).

## 1. Unzip and open

Unzip `grounded-rag-agent.zip` into `Downloads`, then in VS Code: **File → Open Folder** and pick the
inner `grounded-rag-agent` folder (the one that contains `README.md`). Open a terminal with
**Terminal → New Terminal**, then check you're in the right place:

```powershell
dir README.md
```

## 2. Create the environment

```powershell
conda create -n rag python=3.11 -y
conda activate rag
pip install -e ".[dev,ui,pdf]"
```

Your prompt should now start with `(rag)`.

## 3. Check it works offline (no key)

```powershell
pytest
rag-agent demo
python scripts/guardrail_demo.py
```

`pytest` should end with `47 passed`.

## 4. Add your Gemini key

```powershell
copy .env.example .env
code .env
```

Paste your key after `GEMINI_API_KEY=` and save. Never paste the key into a chat, screenshot or commit.
`.env` is git-ignored.

Check the key and model:

```powershell
python scripts/check_gemini.py gemini-3.6-flash
```

## 5. Ask questions

```powershell
rag-agent ask "When should the M365 renewal quote be sent?" --trace
rag-agent ask "When did Windows 10 reach end of support?"
rag-agent ask "What discount did we give Syarikat Maju Jaya last March?"
```

The second one needs internet: it uses DuckDuckGo search (no key). If DuckDuckGo rate-limits you, try
`--web wikipedia`.

## 6. The app

```powershell
streamlit run app.py
```

It opens in your browser. Press `Ctrl + C` in the terminal to stop it.

## 7. Real-model evaluation (for your CV numbers)

```powershell
rag-agent eval --provider gemini --web ddgs
```

This makes about 60 Gemini calls with a 4-second pause between them, to stay under the free tier's
per-minute limit. Expect **10-15 minutes**. It prints one line per question as it goes:

```
[ 3/25] kb03    naive: correct         grounded: correct
```

At the end it writes `outputs/eval/eval_report.md`. Before you quote the numbers, check the top
of the report:

- **No warning lines:** every question completed, so the numbers are good to use.
- **"N question(s) hit model errors":** those were rate-limited and left out. Run it again, slower: `--delay 8`.
- **"Stopped early ... Daily free-tier quota ... used up":** today's quota is gone. Run it again after the
  reset (about 3-4 pm Malaysia time), or set `GEMINI_MODEL` in `.env` to another model from
  `python scripts/check_gemini.py --list`. Each model has its own quota.

## Troubleshooting

| Problem | Fix |
|---|---|
| `rag-agent` is not recognized | `conda activate rag`, then `pip install -e ".[dev,ui,pdf]"` again |
| `FileNotFoundError: README.md` or `config.yaml` | You're in the outer folder: `cd grounded-rag-agent` |
| `GEMINI_API_KEY is not set` | Check `.env` is in the same folder as `README.md` and the line has no spaces around `=` |
| Gemini `404` | Model name changed: `python scripts/check_gemini.py --list` and set `GEMINI_MODEL` in `.env` |
| Gemini `429` / `503` | Free-tier limit or busy: the agent waits and retries by itself; if it still fails, wait a minute |
| Eval says "hit model errors" or "Stopped early" | See step 7: re-run with `--delay 8`, or after the daily reset |
| `web search: failed: ... Ratelimit` in `--trace` | DuckDuckGo throttled you: use `--web wikipedia` or wait |
| Answers look stale after editing `data/kb` | The index rebuilds automatically; to force it, delete `outputs/index.json` |

## Deploy the app (optional)

1. Push the repo to GitHub (see the README).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch
   `main`, file `app.py`.
3. To use Gemini there, add `GEMINI_API_KEY = "..."` under **Advanced settings → Secrets**. Without
   it the app runs in offline mode, which is fine for a public demo. Only use the fictional playbook
   on a public app.

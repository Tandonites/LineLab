# Setup

## Prereqs

- Python 3.11+
- Node.js 18+ (for the TS frontend)
- `make`, `curl`, `unzip`

## First-time setup

```bash
make setup
```

This does everything in one shot:

- creates a Python venv at `.venv/`
- installs all `requirements.txt` deps
- scaffolds a Vite + React + TS frontend in `frontend/`
- creates `data/`, `models/`, `src/`, `notebooks/` dirs
- writes a `.env` template

Run `make help` anytime to see all targets.

## Pull data

```bash
make data
```

Downloads:

- MTA subway GTFS schedule → `data/raw/gtfs_subway/`
- MTA ridership CSV → `data/raw/mta_ridership.csv`

The ridership file is big — give it a minute.

## Train

```bash
make preprocess   # clean + feature-engineer (needs src/preprocess.py)
make train        # fit ridership + cost models (needs src/train.py)
```

## Run the app

Two terminals:

```bash
# terminal 1 — Python API
make backend      # FastAPI on http://localhost:8000

# terminal 2 — TS frontend
make frontend     # Vite dev server on http://localhost:5173
```

The frontend posts drawn subway lines to the backend's `/predict` endpoint and renders the predicted ridership + cost changes.

## Other useful targets

| Target         | What it does                                            |
| -------------- | ------------------------------------------------------- |
| `make jupyter` | launch notebook server in `notebooks/`                  |
| `make freeze`  | pin current deps to `requirements.lock`                 |
| `make clean`   | remove `__pycache__`, checkpoints, etc.                 |
| `make nuke`    | clean + delete `.venv`, `data/`, `models/`, `frontend/` |

## API keys

Edit `.env` after `make setup`:

```
NYC_OPEN_DATA_APP_TOKEN=    # optional, raises rate limit on sodapy
API_HOST=0.0.0.0
API_PORT=8000
```

The NYC Open Data token is optional — sodapy works without it but you'll hit rate limits faster.

## Troubleshooting

**`make: command not found`** — install make (`xcode-select --install` on macOS, `apt install make` on Ubuntu).

**Vite scaffold prompts interactively** — if `npm create vite` asks questions, just hit enter through them; the `--template react-ts` flag should handle it.

**Torch install is slow** — that's expected, it's a few hundred MB. Run `make setup` on good wifi.

**Port 8000 already in use** — change `API_PORT` in `.env` and pass it to uvicorn, or kill whatever's using it (`lsof -i :8000`).

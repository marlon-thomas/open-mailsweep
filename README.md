# 🧹 OpenMailSweep

**A self-hosted mail cleaner that learns *you* — Gmail & Yahoo, fully local, millisecond-fast.**

OpenMailSweep continuously scans your inbox, applies hard safety rules, routes mail you have already taught it about instantly, and classifies everything else with a **fast local personalised model** that learns from every decision you make. Uncertain mail goes to a Pending queue — you stay the authority. It runs entirely on your machine (Docker or Python), stores nothing off-box, and needs no AI API keys.

```text
Gmail / Yahoo ─► hard safety rules ─► learned rules ─► local classifier ─┬─ high confidence ─► Action queue
   (keep, starred, gov,     (list / sender /      (trains on email      ├─ low confidence ─► Pending (you decide)
    replied, allow-list)     domain, learned       CONTENT, not who       └─ every answer trains rules + model online
                             online)                sent it)
```

## Why it exists

General-purpose local LLM classifiers (the "Laya" era of this project) cost **20–35 seconds per email** on CPU — a permanent backlog. OpenMailSweep replaces that with a sparse linear model (hashed word + character n-grams, log-loss SGD):

| | Laya (v0.9.x) | OpenMailSweep (v1.0) |
|---|---|---|
| Prediction | 20–35 s / email | **~5–40 ms / email** |
| Model download | ~850 MB | none |
| Learns from your answers | no | **yes, online + periodic retrain** |
| GPU | optional heavy stack | automatic if compatible (CPU fallback identical) |

Benchmark on a CPU-only machine: `openmailsweep benchmark-classifier`.

## Features

- **Providers:** Gmail (REST + OAuth) and Yahoo (IMAP + SMTP, app password) behind one interface.
- **Content-first classifier:** learns from subject, body signals, structure, unsubscribe headers and mailbox categories — never memorising the exact sender address or List-ID. New senders are classified by *what their mail looks like*. Identity routing stays with learned rules.
- **Online learning:** every Pending answer trains the model immediately; corrections are double-weighted. Full retrain every 6 h / 100 new examples / on rule changes — atomically swapped, previous model retained.
- **Bootstrap:** existing learned rules and confirmed decisions rebuild the model before any unknown mail is classified.
- **Safety hierarchy:** starred/important/replied/official-domain/transactional/allow-list protections outrank everything; low confidence always goes to Pending; destructive classes require very high confidence; unsubscribe only ever runs its own verified HTTPS-one-click → authenticated-mailto fallback chain with duplicate suppression.
- **Continuous intake:** streams the whole matching mailbox page by page (no message cap) while classification and a concurrent action pool run independently; shared weighted Gmail quota pacing keeps Google happy.
- **Human-in-the-loop web UI** on `http://localhost:8787`: dashboard, Pending queue, action queue, learned-rules editor, classifier health.
- **Persistent local state:** everything in SQLite under `./data`. No cloud, no telemetry, no keys.

## Install (one line)

```bash
curl -fsSL https://raw.githubusercontent.com/marlon-thomas/open-mailsweep/main/install.sh | bash
```

Then open **http://localhost:8787**. The script needs Docker; it clones the repo to `~/open-mailsweep`, creates `.env`, and starts the stack. Re-running it upgrades (`git pull` + rebuild).

<details>
<summary>Manual install / without Docker</summary>

```bash
git clone https://github.com/marlon-thomas/open-mailsweep && cd open-mailsweep
cp .env.example .env          # then edit
docker compose up -d --build  # start
docker compose logs -f sweeper
# Stop: docker compose down
```

Native Python (3.11+): `pip install -e . && openmailsweep auth && openmailsweep serve`
</details>

## First-time provider setup

### Gmail
1. Create a Google Cloud project → enable the Gmail API → OAuth **Desktop app** client → download the client JSON to `secrets/credentials.json`.
2. Authenticate once (browser):
   ```bash
   docker compose run --rm -p 127.0.0.1:8765:8765 sweeper auth
   ```
   Scopes: `gmail.modify` + `gmail.send` (send is only for authenticated `List-Unsubscribe: mailto:` fallbacks).

### Yahoo
1. Generate an **app password**: Yahoo Mail → Settings → Accounts → *Manage app passwords* (Yahoo rejects normal passwords on IMAP).
2. In `.env`:
   ```dotenv
   MAIL_PROVIDER=yahoo
   YAHOO_EMAIL=you@yahoo.com
   YAHOO_APP_PASSWORD=xxxx xxxx xxxx xxxx
   ```
3. Verify: `docker compose run --rm sweeper auth`

Yahoo notes: message ids are folder-scoped internally (`Mailbox:uid`); Gmail categories map to the closest flags (`\Flagged` → starred protection); archive/read-later/trash are folder moves with copy-verified deletes.

## How decisions are made

1. **Hard protection** (never automated): starred, Gmail-important, threads you replied to, official domains (`*.gov.uk`, `*.nhs.uk`, `*.police.uk`…), security/invoice/delivery/transactional subjects, SPF/DMARC failures, configured allow-lists.
2. **Learned rules** you taught it — mailing list / sender / domain — route immediately (still re-checked against hard safety before any mutation).
3. **Local classifier** acts only above per-action confidence thresholds once mature (`20` examples → recommendations only, `100` → auto-action enabled, per-class thresholds up to `0.98` + verified unsubscribe method for Unsubscribe + Clean).
4. **You** decide everything else from the Pending queue; each answer feeds rules + model.

UI actions: **Keep in Inbox · Read Later · Archive · Clean · Unsubscribe + Clean** (the app picks HTTPS-one-click or authenticated mailto automatically; ordinary unsubscribe web pages are never browsed).

## Configuration

Everything has sensible defaults — see [`.env.example`](.env.example) for the annotated list: classifier thresholds, retrain cadence, provider choice, Gmail quota pacing (`GMAIL_QUOTA_UNITS_PER_MINUTE`), worker pool size, scan query/interval.

## Legacy CLI

`audit`, `review`, `sweep`, `bulk-clean`, `auth`, `serve`, `benchmark-classifier` — e.g.

```bash
docker compose run --rm sweeper audit --query "in:inbox newer_than:30d" --limit 50
docker compose run --rm sweeper benchmark-classifier --samples 2000
```

## Data & privacy

| File | Contents |
|---|---|
| `data/openmailsweep.db` | queues, learned rules, training examples (sanitised features + ≤600-char body excerpts) |
| `data/audit.db` | decision history |
| `data/classifier/` | model + metadata (versioned, atomic swaps, previous kept) |
| `secrets/` | OAuth client/token (Gmail) — never commit |

Mail text never leaves your machine. No analytics, no external calls beyond the email providers.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE). Not affiliated with, or endorsed by, Google LLC or Yahoo. You are responsible for what automated cleanup does to your mailbox; safety defaults are deliberately conservative.

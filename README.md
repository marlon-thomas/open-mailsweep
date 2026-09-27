# 🧹 OpenMailSweep

**A self-hosted mail cleaner that learns from your decisions — Gmail & Yahoo, fully local.**

OpenMailSweep continuously scans your inbox, applies hard safety rules, instantly routes mail you have already taught it about, and classifies everything else with a fast local model that learns from every decision you make. Uncertain mail goes to a Pending queue — you stay the authority. No AI keys, no cloud, nothing leaves your machine.

```bash
curl -fsSL https://raw.githubusercontent.com/marlon-thomas/open-mailsweep/main/install.sh | bash
```

Then open **http://localhost:8787**. Needs Docker; clones to `~/open-mailsweep`, creates `.env`, starts the stack. Re-run any time to upgrade.

## How mail is decided

1. **Hard protection** — never automated: starred, flagged important, threads you replied to, `*.gov.uk`/`*.nhs.uk`/`*.police.uk`, security/invoice/delivery/transactional subjects, SPF/DMARC failures, your allow-lists.
2. **Learned rules** — mailing list / sender / domain rules you taught it route instantly (hard safety is still re-checked before any mutation).
3. **Local classifier** — predicts from the email's *content* (subject, body signals, structure, unsubscribe headers, mailbox category — deliberately never the exact sender or List-ID, which is the rules table's job). Acts only above per-action confidence gates; every answer you make trains it online, and destructive gates are calibrated from measured holdout precision (`LOCAL_CLASSIFIER_TARGET_PRECISION`, default 97%) rather than fixed guesses.
4. **Pending** — everything uncertain waits for you; one answer can release all matching mail.

UI actions: **Keep in Inbox · Read Later · Archive · Clean · Unsubscribe + Clean** — the app picks authenticated one-click HTTPS or an authenticated mailto fallback automatically and suppresses duplicate unsubscribes per list; ordinary unsubscribe web pages are never browsed.

## Providers

| | Gmail | Yahoo |
|---|---|---|
| Protocol | Gmail API (OAuth) | IMAP + SMTP |
| Setup | OAuth desktop client JSON at `secrets/credentials.json`, then `openmailsweep auth` | `YAHOO_EMAIL` + app password in `.env` (Mail → Settings → Accounts → *Manage app passwords*) |
| Protection mapping | native | starred from `\Flagged`; replied-thread via Sent header search; folder moves are copy-verified (source deleted only after the destination copy is confirmed) |

Set `MAIL_PROVIDER=gmail|yahoo` in `.env`; both clients ship in one image.

## Classifier in one paragraph

A hashed linear model (word + character n-grams, log-loss SGD, scikit-learn). Predictions run in single-digit-to-tens of milliseconds on CPU, with automatic GPU acceleration when a compatible one is present. It bootstraps from your existing confirmed rules/decisions before classifying anything, learns incrementally from every Pending answer (corrections double-weighted), and fully retrains every 6 h / 100 examples / on rule changes with atomic model swap and previous-model retention. Below 20 examples everything goes to Pending; below 100, only high-confidence non-destructive actions fire.

## Configuration

Sensible defaults for everything; the annotated [`.env.example`](.env.example) covers thresholds, retrain cadence, scan query/interval, action-worker pool, and Gmail quota pacing (`GMAIL_QUOTA_UNITS_PER_MINUTE=3600` default, weighted and shared by all workers).

## Legacy CLI

```bash
openmailsweep audit  --query "in:inbox newer_than:30d" --limit 50   # read-only review
openmailsweep review | sweep | bulk-clean                            # gated by ALLOW_SWEEP / ALLOW_UNSUBSCRIBE
openmailsweep serve | auth | benchmark-classifier
```

## Data & privacy

| Path | Contents |
|---|---|
| `data/openmailsweep.db` | queues, learned rules, training examples (sanitised features; body excerpts ≤600 chars) |
| `data/audit.db` | decision history |
| `data/classifier/` | versioned model + metadata (atomic swaps, previous kept) |
| `secrets/` | provider credentials — never commit |

SQLite with WAL; interrupted work self-heals at startup. No analytics, no telemetry, no traffic other than your mail provider.

## Development

```bash
pip install -e ".[dev]" && pytest      # 117 tests
```

## License

MIT — see [LICENSE](LICENSE). Not affiliated with Google or Yahoo. Automation can trash and unsubscribe mail irreversibly; the safety hierarchy and gates are deliberately conservative, and you remain responsible for what you approve.

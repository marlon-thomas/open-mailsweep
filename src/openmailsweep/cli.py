from __future__ import annotations

import typer

from .audit import AuditLog
from .config import Settings, load_policy
from .gmail_client import GmailClient
from .provider import create_mail_provider, provider_is_ready
from .jev_client import JevClient
from .runner import Runner

app = typer.Typer(help="Audit-first mail cleaner with a fast local personalised classifier (or hosted Jev)")


def _echo(message: str) -> None:
    typer.echo(message)


def _components():
    settings = Settings.from_env()
    policy = load_policy(settings.policy_file)
    gmail = create_mail_provider(settings, interactive=True)

    classifier = JevClient(
        settings.jev_api_key,
        settings.jev_model,
        settings.jev_base_url,
        max_body_chars=policy.classification.max_body_chars,
    )
    return settings, policy, gmail, classifier


@app.command()
def auth():
    """Authorize the configured provider (Gmail OAuth flow / Yahoo credential check)."""
    settings = Settings.from_env()
    if settings.mail_provider == "yahoo":
        from .yahoo_client import YahooClient

        client = YahooClient(settings)
        client.list_folders()
        typer.echo(f"Authenticated as {settings.yahoo_email} (IMAP folders reachable)")
        return
    gmail = GmailClient(settings.gmail_credentials, settings.gmail_token)
    profile = gmail.service.users().getProfile(userId="me").execute()
    typer.echo(f"Authenticated as {profile.get('emailAddress')}")


def _run(mode: str, query: str, limit: int):
    typer.echo(f"[start] mode={mode} query={query!r} limit={limit}")
    settings, policy, gmail, classifier = _components()
    runner = Runner(
        gmail,
        classifier,
        policy,
        AuditLog(settings.audit_db),
        report=_echo,
        batch_size=settings.classify_batch_size if settings.classifier == "jev" else 1,
    )
    counts = runner.run(
        mode,
        query,
        limit,
        allow_sweep=settings.allow_sweep,
        allow_unsubscribe=settings.allow_unsubscribe,
    )
    typer.echo(f"Classifier: {settings.classifier}")
    typer.echo(f"Mode: {mode}")
    typer.echo(f"Query: {query}")
    typer.echo(f"Processed: {counts.get('processed', 0)}")
    for key in sorted(counts):
        if key != "processed":
            typer.echo(f"  {key}: {counts[key]}")


@app.command()
def audit(query: str = "in:inbox newer_than:1y", limit: int = 100):
    """Read-only classification, unsubscribe detection, and audit logging."""
    _run("audit", query, limit)


@app.command()
def review(query: str = "in:inbox newer_than:1y", limit: int = 100):
    """Apply OpenMailSweep review/category labels; does not trash messages."""
    _run("review", query, limit)


@app.command()
def sweep(query: str = "label:OpenMailSweep/Junk", limit: int = 100):
    """Trash only high-confidence junk. Never follows unsubscribe links."""
    _run("sweep", query, limit)




@app.command()
def serve(host: str = "0.0.0.0", port: int | None = None):
    """Run the local OpenMailSweep web UI plus continuous scanner/action workers."""
    import uvicorn

    from .web import create_app

    settings = Settings.from_env()
    listen_port = port or settings.ui_port
    typer.echo(f"OpenMailSweep UI: http://localhost:{listen_port}")
    uvicorn.run(create_app(), host=host, port=listen_port, log_level="info")

@app.command("bulk-clean")
def bulk_clean(
    query: str = "{label:OpenMailSweep/Promotion label:OpenMailSweep/Newsletter}",
    limit: int = 100,
):
    """Unsubscribe safe one-click bulk mail where possible, then trash it."""
    _run("bulk-clean", query, limit)


@app.command("benchmark-classifier")
def benchmark_classifier(samples: int = 2000, rounds: int = 5):
    """Benchmark the local classifier latency/throughput on synthetic data.

    Generates ``samples`` distinct-sender examples across the five action
    classes, trains, and reports prediction latency percentiles plus memory
    and model size.
    """
    import random as _random
    import time as _time

    from .config import load_policy as _load_policy
    from .local_classifier import LocalClassifier as _LocalClassifier
    from .training_data import TrainingExample as _Example

    settings = Settings.from_env()
    policy = _load_policy(settings.policy_file)
    rng = _random.Random(7)
    templates = {
        "unsubscribe_trash": ("Flash sale {pct}% off everything shop now {word}", "limited time offer promo discount code clearance coupon save {pct}"),
        "trash": ("You have won a free {word} claim now", "click here winner congratulations urgent prize free"),
        "archive": ("Your {period} usage summary", "summary report generated automatically for your records"),
        "read_later": ("Job alert: {n} roles matching your saved search", "new vacancies recommended jobs career opportunities profile"),
        "keep": ("Re: are we still on for {day}", "hey looking forward to seeing you let me know"),
    }
    words = ["gadget", "widget", "sneakers", "laptop", "camera", "jacket", "speaker"]
    days = ["monday", "tuesday", "friday", "saturday", "sunday"]
    periods = ["weekly", "monthly", "quarterly"]
    labels = list(templates)
    examples = []
    per_class = max(10, samples // len(labels))
    for index, label in enumerate(labels):
        subject_tpl, snippet_tpl = templates[label]
        for i in range(per_class):
            sender = f"user{i}.{index}@sender-{index}-{i % 997}.example-{i}.com"
            examples.append(
                _Example(
                    message_id=f"bench-{label}-{i}",
                    label=label,
                    sender=f"Bench Person {i} <{sender}>",
                    sender_address=sender,
                    subject=subject_tpl.format(pct=rng.randint(10, 90), word=rng.choice(words), n=rng.randint(2, 40), day=rng.choice(days), period=rng.choice(periods)),
                    snippet=snippet_tpl.format(pct=rng.randint(10, 90), word=rng.choice(words), day=rng.choice(days)),
                    list_id=f"news@list-{i}.example.com" if label in {"unsubscribe_trash", "archive"} else "",
                    features={"labels": ["CATEGORY_PROMOTIONS"] if "sale" in label else ["UNREAD"], "one_click": label == "unsubscribe_trash", "has_mailto": label == "unsubscribe_trash", "has_list_unsubscribe": label == "unsubscribe_trash", "precedence_bulk": label == "unsubscribe_trash", "auto_submitted": False, "body_excerpt": snippet_tpl[:120]},
                )
            )
    typer.echo(f"[benchmark] examples={len(examples)} rounds={rounds}")
    started = _time.perf_counter()
    model = _LocalClassifier(settings, policy)
    state = model.bootstrap(examples, "bench")
    train_seconds = _time.perf_counter() - started
    typer.echo(f"[benchmark] {state} full-train seconds={train_seconds:.2f}")
    latencies: list[float] = []
    correct = 0
    holdout = examples[::4]
    for _round in range(rounds):
        for ex in holdout:
            pred = model.predict(ex)
            latencies.append(pred.latency_ms)
            correct += pred.action == ex.label
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    size_mb = sum(f.stat().st_size for f in settings.classifier_dir.glob("*")) / 1e6 if settings.classifier_dir.exists() else 0
    typer.echo(
        f"[benchmark] holdout={len(holdout) * rounds} accuracy={correct / (len(holdout) * rounds):.3f} "
        f"p50={p50:.1f}ms p95={p95:.1f}ms throughput={1000 / max(p50, 0.001):.0f} msg/s model={size_mb:.1f}MB"
    )
    if p95 > 100:
        typer.echo("[benchmark] WARNING: p95 exceeds the 100 ms target")
    typer.echo(f"[benchmark] backend={model.backend.name} device={model.backend.device}")


if __name__ == "__main__":
    app()

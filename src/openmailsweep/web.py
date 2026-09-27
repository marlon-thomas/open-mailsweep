from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .audit import AuditLog
from .config import Settings, load_policy
from .state_store import ACTIONS, StateStore
from .worker import OpenMailSweepService

ActionName = Literal["keep", "archive", "read_later", "trash", "unsubscribe_trash"]
ScopeName = Literal["message", "list", "sender", "domain"]


class PendingAnswer(BaseModel):
    action: ActionName
    scope_type: ScopeName


class RuleUpdate(BaseModel):
    action: ActionName


def _signals(item: dict) -> list[str]:
    try:
        value = json.loads(item.get("safety_signals_json") or "[]")
        return value if isinstance(value, list) else []
    except Exception:
        return []


def _public_item(item: dict) -> dict:
    value = dict(item)
    value.pop("safety_signals_json", None)
    value["safety_signals"] = _signals(item)
    value["gmail_url"] = f"https://mail.google.com/mail/u/0/#all/{item.get('message_id','')}"
    scopes = ["message"]
    if item.get("list_id"):
        scopes.append("list")
    if item.get("sender_address"):
        scopes.append("sender")
    if item.get("sender_domain"):
        scopes.append("domain")
    value["scope_options"] = scopes
    return value


def _static_version() -> str:
    """Cache-buster for static assets so browsers pick up UI changes."""
    from . import __version__
    try:
        js = Path(__file__).resolve().parent / "static" / "app.js"
        return f"{__version__}-{int(js.stat().st_mtime)}"
    except Exception:
        return __version__


def create_app() -> FastAPI:
    settings = Settings.from_env()
    policy = load_policy(settings.policy_file)
    store = StateStore(settings.state_db)
    audit = AuditLog(settings.audit_db)
    service = OpenMailSweepService(settings, policy, store, audit)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(title="OpenMailSweep", version="0.9.6", lifespan=lifespan)
    app.state.settings = settings
    app.state.policy = policy
    app.state.store = store
    app.state.service = service

    package_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(package_dir / "templates"))
    app.mount("/static", StaticFiles(directory=str(package_dir / "static")), name="static")

    def page_context(request: Request, title: str) -> dict:
        return {
            "request": request,
            "app_version": _static_version(),
            "title": title,
            "stats": store.stats(),
            "worker": service.status.snapshot(),
            "paused": service.is_paused(),
            "scan_query": settings.scan_query,
            "scan_interval": settings.scan_interval_seconds,
            "fetch_batch_size": settings.fetch_batch_size,
            "action_workers": settings.action_workers,
            "mail_provider": settings.mail_provider,
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        ctx = page_context(request, "Dashboard")
        ctx["recent"] = [_public_item(x) for x in store.list_recent(12)]
        return templates.TemplateResponse(request, "dashboard.html", ctx)

    @app.get("/pending", response_class=HTMLResponse)
    def pending_page(request: Request):
        return templates.TemplateResponse(request, "pending.html", page_context(request, "Pending decisions"))

    @app.get("/queue", response_class=HTMLResponse)
    def queue_page(request: Request):
        return templates.TemplateResponse(request, "queue.html", page_context(request, "Action queue"))

    @app.get("/rules", response_class=HTMLResponse)
    def rules_page(request: Request):
        ctx = page_context(request, "Learned rules")
        ctx["rules"] = store.list_rules()
        ctx["history"] = store.list_rule_history(20)
        return templates.TemplateResponse(request, "rules.html", ctx)

    @app.get("/health")
    def health():
        return {"ok": True, "worker": service.status.snapshot(), "stats": store.stats()}

    @app.get("/api/classifier")
    def api_classifier():
        state = service.local.public_state() if service.local is not None else None
        return {"enabled": service.uses_local_classifier, "classifier": state}

    @app.get("/api/stats")
    def api_stats():
        return {
            "stats": store.stats(),
            "worker": service.status.snapshot(),
            "paused": service.is_paused(),
        }

    @app.get("/api/pending")
    def api_pending(limit: int = 20):
        limit = max(1, min(100, limit))
        items = [_public_item(x) for x in store.list_pending(limit)]
        return {"items": items, "count": store.stats().get("pending", 0)}

    @app.post("/api/pending/{item_id}/answer")
    def api_answer_pending(item_id: int, body: PendingAnswer):
        try:
            result = store.answer_pending(item_id, body.action, body.scope_type)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        service.handle_user_answer(item_id, body.action, body.scope_type)
        service.wake_actions()
        service.request_scan()  # Newly learned rules may save work on the current inbox.
        return {"ok": True, **result, "stats": store.stats()}

    @app.get("/api/queue")
    def api_queue(limit: int = 60):
        limit = max(1, min(200, limit))
        return {
            "discovered": [_public_item(x) for x in store.list_queue(("discovered",), limit)],
            "classifying": [_public_item(x) for x in store.list_queue(("classifying",), limit)],
            "actionable": [_public_item(x) for x in store.list_queue(("actionable",), limit)],
            "processing": [_public_item(x) for x in store.list_queue(("processing",), limit)],
            "failed": [_public_item(x) for x in store.list_queue(("failed",), limit)],
            "recent": [_public_item(x) for x in store.list_recent(limit)],
        }

    @app.get("/api/rules")
    def api_rules():
        return {"rules": store.list_rules(), "history": store.list_rule_history(50)}

    @app.put("/api/rules/{rule_id}")
    def api_update_rule(rule_id: int, body: RuleUpdate):
        try:
            rule = store.update_rule(rule_id, body.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        source_message_id = rule.get("source_message_id")
        if source_message_id:
            row = store.get_item_by_message(source_message_id)
            if row is not None:
                service.handle_user_answer(int(row["id"]), body.action, rule.get("scope_type", "message"))
        service.handle_rule_change()
        service.wake_actions()
        return {"ok": True, "rule": rule}

    @app.delete("/api/rules/{rule_id}")
    def api_delete_rule(rule_id: int):
        try:
            returned = store.delete_rule(rule_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        service.handle_rule_change()
        return {"ok": True, "returned_to_pending": returned}

    @app.post("/api/worker/scan-now")
    def api_scan_now():
        service.request_scan()
        return {"ok": True}

    @app.post("/api/worker/pause")
    def api_pause():
        service.pause()
        return {"ok": True, "paused": True}

    @app.post("/api/worker/resume")
    def api_resume():
        service.resume()
        return {"ok": True, "paused": False}

    return app

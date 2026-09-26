#!/usr/bin/env bash
#
# OpenMailSweep installer — Docker-based one-liner:
#   curl -fsSL https://raw.githubusercontent.com/marlon-thomas/open-mailsweep/main/install.sh | bash
#
# Re-running upgrades in place (git pull + rebuild). All data lives in ./data
# and secrets in ./secrets; the installer never touches anything else.
#
set -euo pipefail

REPO_URL="${OPENMAILSWEEP_REPO:-https://github.com/marlon-thomas/open-mailsweep.git}"
BRANCH="${OPENMAILSWEEP_BRANCH:-main}"
INSTALL_DIR="${OPENMAILSWEEP_DIR:-$HOME/open-mailsweep}"

log()  { printf '\033[1;32m[open-mailsweep]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[open-mailsweep]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[open-mailsweep]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,12p' "$0"
  exit 0
fi

command -v docker >/dev/null 2>&1 || die "Docker is required. Install it from https://docs.docker.com/go/docker-install/"
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  die "Docker Compose (v2 plugin or docker-compose) is required."
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  log "Updating existing install at $INSTALL_DIR"
  git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
elif [[ -d "$INSTALL_DIR" ]]; then
  warn "$INSTALL_DIR exists but is not a git checkout; leaving it untouched."
  die "Remove or relocate it, or set OPENMAILSWEEP_DIR to a different path."
else
  log "Cloning $REPO_URL -> $INSTALL_DIR"
  mkdir -p "$(dirname "$INSTALL_DIR")"
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
  else
    log "git not found; downloading release tarball instead (updates need re-running this script)"
    mkdir -p "$INSTALL_DIR"
    curl -fsSL "https://codeload.github.com/marlon-thomas/open-mailsweep/tar.gz/refs/heads/$BRANCH" \
      | tar --strip-components=1 -xz -C "$INSTALL_DIR"
  fi
fi

cd "$INSTALL_DIR"
mkdir -p secrets data

if [[ ! -f .env ]]; then
  cp .env.example .env
  log "Created .env from .env.example"
fi

PROVIDER="$(sed -n 's/^MAIL_PROVIDER=//p' .env | tail -n1)"
PROVIDER="${PROVIDER:-gmail}"

if [[ "$PROVIDER" == "gmail" && ! -f secrets/credentials.json ]]; then
  warn "No Gmail OAuth client found at secrets/credentials.json."
  warn "Create a Google Cloud project, enable the Gmail API, create an OAuth 'Desktop app'"
  warn "client, and save its JSON to $INSTALL_DIR/secrets/credentials.json — then re-run."
fi
if [[ "$PROVIDER" == "yahoo" ]]; then
  warn "Make sure YAHOO_EMAIL and YAHOO_APP_PASSWORD are set in $INSTALL_DIR/.env"
  warn "(app password: Yahoo Mail -> Settings -> Accounts -> Manage app passwords)."
fi

log "Building and starting OpenMailSweep"
$COMPOSE up -d --build sweeper

log "Done.  UI:  http://localhost:$(sed -n 's/^UI_PORT=//p' .env | tail -n1 | grep -E '^[0-9]+$' || echo 8787)"
if [[ "$PROVIDER" == "gmail" ]]; then
  log "One-time Gmail authorisation:"
  log "  cd $INSTALL_DIR && $COMPOSE run --rm -p 127.0.0.1:8765:8765 sweeper auth"
fi
log "Logs:    cd $INSTALL_DIR && $COMPOSE logs -f sweeper"
log "Stop:    cd $INSTALL_DIR && $COMPOSE down"

#!/usr/bin/env bash
# Start RoadScan and expose it over HTTPS so a phone can record rides.
#
#   scripts/start_tunnel.sh            # random access key, port 8000
#   PORT=8001 ROADSCAN_KEY=mykey scripts/start_tunnel.sh
#
# Prints a phone link (+ QR code) for /record and a dashboard link. Uses a free Cloudflare
# quick tunnel: no account needed, the URL changes on every start, and it stops when you press Ctrl+C.
# The Mac is kept awake while this runs (lid must stay open on battery).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
export ROADSCAN_KEY="${ROADSCAN_KEY:-$(openssl rand -hex 8)}"
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

command -v cloudflared >/dev/null || { echo "cloudflared not found. Install it with: brew install cloudflared"; exit 1; }
[ -x .venv/bin/uvicorn ] || { echo "Python environment missing. See README → Quick start."; exit 1; }
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Stop that server or run with PORT=<other>."; exit 1
fi

pids=()
cleanup() {
  echo; echo "Stopping server, tunnel and keep-awake…"
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/uvicorn server.app:app --host 127.0.0.1 --port "$PORT" > "$LOG_DIR/server.log" 2>&1 &
pids+=($!)
caffeinate -dims -w "${pids[0]}" &
pids+=($!)

for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/api/stats" >/dev/null && break; sleep 0.5; done
curl -sf "http://127.0.0.1:$PORT/api/stats" >/dev/null || { echo "Server failed to start — see $LOG_DIR/server.log"; exit 1; }

cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$LOG_DIR/tunnel.log" 2>&1 &
pids+=($!)

URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" | head -1 || true)
  [ -n "$URL" ] && break
  sleep 0.5
done
[ -n "$URL" ] || { echo "Tunnel did not start — see $LOG_DIR/tunnel.log"; exit 1; }

# Don't probe $URL from this Mac: the new hostname takes a few seconds to appear in DNS, and an early
# lookup gets a "not found" cached by macOS for a while. Give DNS a moment instead.
sleep 5

RECORD="$URL/record?key=$ROADSCAN_KEY"
cat <<EOF

  RoadScan is online
  ──────────────────
  Phone recorder : $RECORD
  Dashboard      : $URL/?key=$ROADSCAN_KEY
  Local          : http://localhost:$PORT

  Scan with the phone camera to open the recorder:
EOF
.venv/bin/python -c "import qrcode,sys; q=qrcode.QRCode(border=1); q.add_data(sys.argv[1]); q.print_ascii(invert=True)" "$RECORD"
cat <<EOF

  If the link doesn't open straight away, wait ~30 s and retry (the tunnel's DNS is still spreading).
  Anyone with the link can view the dashboard; uploads and changes need the key.
  Logs: $LOG_DIR/server.log, $LOG_DIR/tunnel.log    ·    Ctrl+C to stop.
EOF

wait "${pids[0]}"

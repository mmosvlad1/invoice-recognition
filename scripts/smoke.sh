#!/usr/bin/env bash
# Smoke-test the app container: build and start the real compose.yaml under a project name of its own
# (one per run, so two runs never meet) and on a free loopback port, check what a visitor and an operator
# rely on, then remove everything it made.
#
#   scripts/smoke.sh
#
# Needs Docker with Compose 2.24 or newer, curl and python3. Nothing leaves the machine: the key is a
# dummy and the container's proxy points at a closed local port, so every model request fails at once.
# It never reads or touches your .env, your port or your running stack.
set -euo pipefail

cd "$(dirname "$0")/.."

project="invoice-recognition-smoke-$$"
code=smoke-test-access-code-0001
sample=src/invoice_recognition/web/static/samples/invoice-small-misprint.pdf
worker=static/vendor/pdfjs-4.10.38/pdf.worker.min.mjs

scratch="$(mktemp -d "${TMPDIR:-/tmp}/invoice-recognition-smoke.XXXXXX")"
envfile="$scratch/app.env"
cat > "$envfile" <<EOF
OPENROUTER_API_KEY=smoke-not-a-key
ACCESS_CODE=$code
HTTPS_PROXY=http://127.0.0.1:9
HTTP_PROXY=http://127.0.0.1:9
EOF

port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
base="http://127.0.0.1:$port"
export ENV_FILE="$envfile" HOST_PORT="$port"
# Compose's own variables and the container's settings are kept apart, so that nothing meant for
# the container (such as the proxy above) can affect the build.
printf 'ENV_FILE=%s\nHOST_PORT=%s\n' "$envfile" "$port" > "$scratch/compose.env"

# -f: exactly compose.yaml, never an override file or COMPOSE_FILE. --env-file: Compose takes its own
# variables from compose.env and the environment, never from .env.
compose() {
  docker compose --file compose.yaml --project-name "$project" --env-file "$scratch/compose.env" "$@"
}

passed=0
started=0
ok() {
  passed=$((passed + 1))
  echo "ok - $1"
}

finish() {
  status=$?
  # Clean up once, and to the end: a second signal does not cut it short.
  trap - EXIT
  trap "" INT TERM
  if [ "$status" -ne 0 ]; then
    if [ "$started" -eq 0 ] && [ -s "$scratch/up.log" ]; then
      echo "smoke: FAILED; the output of compose up follows" >&2
      tail -n 60 "$scratch/up.log" >&2
    fi
    echo "smoke: FAILED; the app's log follows" >&2
    compose logs --no-color --tail 200 app >&2 || true
  fi
  compose down --rmi local --volumes --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$scratch"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
  echo "not ok - $1" >&2
  exit 1
}

# The body to $scratch/body, the headers to $scratch/headers; prints the status code.
get() {
  curl -sS --max-time 10 -o "$scratch/body" -D "$scratch/headers" -w '%{http_code}' "$@"
}

header() {
  python3 - "$scratch/headers" "$1" <<'EOF'
import sys
name = sys.argv[2].lower()
for line in open(sys.argv[1], encoding="latin-1"):
    key, sep, value = line.partition(":")
    if sep and key.strip().lower() == name:
        print(value.strip())
EOF
}

# 1. Built and healthy.
compose up --detach --build --wait --wait-timeout 90 > "$scratch/up.log" 2>&1 \
  || fail "the stack did not become healthy"
started=1
ok "built and healthy on 127.0.0.1:$port"

# 2. Locked down as compose.yaml says.
container="$(compose ps --quiet app)"
docker inspect --format '{{json .HostConfig}}' "$container" > "$scratch/hostconfig.json"
python3 - "$scratch/hostconfig.json" <<'EOF' || fail "the container is not locked down as compose.yaml says"
import json, sys
host = json.load(open(sys.argv[1]))
want = {
    "ReadonlyRootfs": True,
    "Tmpfs": {"/tmp": "size=128m,mode=1777"},
    "CapDrop": ["ALL"],
    "SecurityOpt": ["no-new-privileges:true"],
    "Memory": 1536 * 2**20,
    "MemorySwap": 1536 * 2**20,
    "NanoCpus": 2 * 10**9,
    "PidsLimit": 256,
    "RestartPolicy": "unless-stopped",
}
got = {key: host.get(key) for key in want}
got["RestartPolicy"] = (host.get("RestartPolicy") or {}).get("Name")
wrong = {key: got[key] for key in want if got[key] != want[key]}
assert not wrong, f"differs from compose.yaml: {wrong}"
EOF
[ "$(compose port app 8000)" = "127.0.0.1:$port" ] || fail "the port is not bound to 127.0.0.1 only"
ok "read-only root with a 128m /tmp, no capabilities, no new privileges, memory = swap = 1536m, 2 CPUs, 256 pids, restart unless-stopped, loopback port"

# 3. Health, without a server header.
[ "$(get "$base/health")" = 200 ] || fail "/health did not answer 200"
[ "$(cat "$scratch/body")" = '{"status":"ok"}' ] || fail "/health body"
[ -z "$(header server)" ] || fail "a server header is sent"
ok "/health answers {\"status\":\"ok\"} without a server header"

# 4. The page with its security policy.
[ "$(get "$base/")" = 200 ] || fail "/ did not answer 200"
csp="$(header content-security-policy)"
case "$csp" in *"default-src 'none'"*) ;; *) fail "CSP lacks default-src 'none'" ;; esac
case "$csp" in *"script-src 'self' 'wasm-unsafe-eval'"*) ;; *) fail "CSP script-src" ;; esac
ok "/ answers 200 with the page's Content-Security-Policy"

# 5. Static files from the installed package.
[ "$(get "$base/$worker")" = 200 ] || fail "the PDF.js worker did not answer 200"
case "$(header content-type)" in text/javascript*) ;; *) fail "the worker's content type" ;; esac
[ "$(get "$base/static/samples/invoice-small-misprint.pdf")" = 200 ] || fail "the sample"
ok "the PDF.js worker and a sample are served"

# 6. No code, wrong code: refused.
for auth in "X-No-Code: 1" "Authorization: Bearer wrong-code-wrong-code-0001"; do
  [ "$(get -X POST -H "$auth" -F "file=@$sample" "$base/invoices")" = 401 ] \
    || fail "an upload without the right code was not refused"
  python3 -c 'import json, sys; assert json.load(open(sys.argv[1]))["code"] == "unauthorized"' \
    "$scratch/body" || fail "the 401 body"
done
ok "uploads without the code or with a wrong one are refused with 401"

# 7. An upload over 1 MB is received in full: the form parser spools it to /tmp, which only the tmpfs
#    makes writable (without it the upload fails with 400). Not a real image, so the content check
#    refuses it once it is read.
auth="Authorization: Bearer $code"
python3 -c 'import os, sys; open(sys.argv[1], "wb").write(b"\xff\xd8\xff\xe0" + os.urandom(3 * 2**20))' \
  "$scratch/large.jpg"
[ "$(get -X POST -H "$auth" -F "file=@$scratch/large.jpg" "$base/invoices")" = 422 ] \
  || fail "a 3 MB upload was not read to the end (expected 422 unreadable_file)"
python3 -c 'import json, sys; assert json.load(open(sys.argv[1]))["code"] == "unreadable_file"' \
  "$scratch/body" || fail "the 422 body"
ok "a 3 MB upload is received in full and refused by the content check (422 unreadable_file)"

# 8. An upload is accepted, and its reading fails inside the container: nothing left it.
[ "$(get -X POST -H "$auth" -F "file=@$sample" "$base/invoices")" = 202 ] \
  || fail "the upload was not accepted"
id="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["id"])' "$scratch/body")"
for _ in $(seq 60); do
  [ "$(get -H "$auth" "$base/invoices/$id")" = 200 ] || fail "the job did not answer 200"
  status="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "$scratch/body")"
  case "$status" in queued | processing) sleep 0.5 ;; *) break ;; esac
done
python3 - "$scratch/body" <<'EOF' || fail "the reading did not fail at the black-holed proxy"
import json, sys
job = json.load(open(sys.argv[1]))
assert job["status"] == "failed", job["status"]
assert job["error"]["code"] == "model_unavailable", job["error"]
errors = [a["error"] for a in job["result"]["details"]["attempts"]]
assert errors and all((e or "").startswith("ConnectError") for e in errors), errors
EOF
ok "an upload is read and fails with model_unavailable: every model request stopped at the proxy"

# 9. Inside the container.
compose exec -T app python - <<'EOF' || fail "inside the container"
import os, pathlib, shutil
assert os.getuid() != 0, "runs as root"
try:
    pathlib.Path("/app/x").touch()
except OSError:
    pass
else:
    raise AssertionError("/app is writable")
assert not pathlib.Path("/app/.env").exists(), "/app/.env exists"
assert shutil.which("uv") is None, "uv is on PATH"
assert os.environ.get("MALLOC_ARENA_MAX") == "2", "MALLOC_ARENA_MAX"
EOF
ok "not root, /app read-only, no .env, no uv, MALLOC_ARENA_MAX=2"

echo "smoke: $passed checks passed"

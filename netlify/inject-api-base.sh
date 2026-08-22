#!/usr/bin/env bash
# Bake the API endpoint into the published page at deploy time.
#
# The frontend resolves its API base in this order:
#   ?api= query param  >  localStorage  >  window.__API_BASE__  >  same origin
#
# This sets that window.__API_BASE__ default from the Netlify environment, so a
# deployed site works on first load without every visitor pasting a URL into the
# Setup panel. If API_BASE is unset the page is published untouched and falls
# back to same-origin — correct when the API serves the frontend itself.
#
# Pure sed on purpose: no python3, no node, no bundler. The page is one
# self-contained file, and a build toolchain would be more moving parts than the
# thing it builds. It also means this script runs identically on the Netlify
# Linux image and in Git Bash on Windows, so it can actually be tested locally.

set -euo pipefail

OUT="app/static/index.html"
MARKER="API_BASE_PLACEHOLDER"

if [ -z "${API_BASE:-}" ]; then
  echo "API_BASE not set — publishing as-is (same-origin API)."
  exit 0
fi

if ! grep -q "$MARKER" "$OUT"; then
  # Either already injected, or someone removed the marker. Both are worth
  # failing loudly for: silently publishing a page pointing at the wrong API
  # is the failure mode this script exists to prevent.
  if grep -q "__API_BASE__" "$OUT"; then
    echo "already injected; nothing to do."
    exit 0
  fi
  echo "ERROR: marker '$MARKER' not found in $OUT — cannot inject API base." >&2
  exit 1
fi

# Strip any trailing slash so url() cannot produce a double slash.
BASE="${API_BASE%/}"

# Reject anything that is not a plain http(s) origin. The value lands inside a
# <script> tag, so a stray quote would be script injection via an env var.
if ! printf '%s' "$BASE" | grep -qE '^https?://[A-Za-z0-9._~:/?#@!$&()*+,;=%-]+$'; then
  echo "ERROR: API_BASE is not a valid http(s) URL: $BASE" >&2
  exit 1
fi

echo "Injecting API_BASE=${BASE}"
# Consume the marker (PLACEHOLDER -> INJECTED) so a second run is a no-op
# rather than stacking duplicate script tags.
sed -i.bak "s|<!-- ${MARKER}|<script>window.__API_BASE__ = \"${BASE}\";</script>\n<!-- API_BASE_INJECTED|" "$OUT"
rm -f "${OUT}.bak"

grep -q "__API_BASE__" "$OUT" && echo "injected ok"

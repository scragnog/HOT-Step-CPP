#!/bin/bash
# HOT-Step 9000 CPP — Linux launch script
# Portable distribution launcher with restart loop support.
#
# Usage: ./HOT-Step.sh
# The script sets HOT_STEP_ROOT for portable path resolution,
# opens the browser after a short delay, and loops on restart
# requests from the UI (same behaviour as HOT-Step.bat on Windows).

echo "============================================="
echo "  HOT-Step 9000 CPP"
echo "  High-Performance Music Generation"
echo "============================================="
echo ""

DIR="$(cd "$(dirname "$0")" && pwd)"
export HOT_STEP_ROOT="$DIR"

# Create models directory if it doesn't exist (first run)
mkdir -p "$DIR/models"

# Open browser after a short delay (desktop only — silent on headless)
if [ "$(uname)" = "Darwin" ]; then OPENER=open; else OPENER=xdg-open; fi
(sleep 5 && "$OPENER" "http://localhost:3001" 2>/dev/null) &

# ── Restart loop ──────────────────────────────────────────
# The server writes .restart-requested when the user clicks
# "Restart" in the UI. After node exits, we check for the
# marker — if it exists, we delete it and relaunch.
while true; do
    echo "Starting server..."
    echo ""
    "$DIR/runtime/bin/node" "$DIR/server/server.mjs"

    # Check for restart marker
    if [ -f "$DIR/.restart-requested" ]; then
        rm "$DIR/.restart-requested"
        echo ""
        echo "[HOT-Step] Restarting..."
        echo ""
        continue
    fi

    echo ""
    echo "[HOT-Step] Server stopped."
    break
done

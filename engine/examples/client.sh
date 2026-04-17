#!/bin/bash
# Test ace-server: LM enriches caption, synth renders to MP3.
# Start the server first (./server.sh), then run this.

set -eu

HOST="http://127.0.0.1:8085"

# poll a job until done, exit 1 on failure
wait_job() {
    local id="$1"
    while true; do
        status=$(curl -sf "${HOST}/job?id=${id}" | jq -r '.status')
        case "$status" in
            done) return 0 ;;
            failed|cancelled) echo "Job ${id}: ${status}"; return 1 ;;
        esac
        sleep 2
    done
}

# LM: submit, poll, fetch result
LM_ID=$(curl -sf "${HOST}/lm" \
    -H "Content-Type: application/json" \
    -d @full-sft.json | jq -r '.id')
echo "LM job: ${LM_ID}"
wait_job "${LM_ID}"
curl -sf "${HOST}/job?id=${LM_ID}&result=1" | jq '.[0]' > server-lm0.json

# synth: submit, poll, fetch result
SYNTH_ID=$(curl -sf "${HOST}/synth" \
    -H "Content-Type: application/json" \
    -d @server-lm0.json | jq -r '.id')
echo "Synth job: ${SYNTH_ID}"
wait_job "${SYNTH_ID}"
curl -sf "${HOST}/job?id=${SYNTH_ID}&result=1" -o server0.mp3

#!/bin/bash
# Roundtrip via ace-server: audio -> understand -> synth -> MP3
#
# Usage: ./client-understand.sh input.wav (or input.mp3)
#
# POST /understand (async job):
# input -> server-understand.json (audio codes + metadata)
#
# POST /synth (async job):
# server-understand.json + input -> server-understand.mp3
#
# Start the server first (./server.sh).

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.wav|input.mp3>"
    exit 1
fi

HOST="http://127.0.0.1:8085"
input="$1"

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

# understand: submit, poll, fetch result
UND_ID=$(curl -sf "${HOST}/understand" \
    -F "audio=@${input}" | jq -r '.id')
echo "Understand job: ${UND_ID}"
wait_job "${UND_ID}"
curl -sf "${HOST}/job?id=${UND_ID}&result=1" -o server-understand.json

sed -i \
    -e 's/"audio_cover_strength": *[0-9.]*/"audio_cover_strength": 0.04/' \
    server-understand.json

# synth: submit, poll, fetch result
SYNTH_ID=$(curl -sf "${HOST}/synth" \
    -F "request=@server-understand.json" \
    -F "audio=@${input}" | jq -r '.id')
echo "Synth job: ${SYNTH_ID}"
wait_job "${SYNTH_ID}"
curl -sf "${HOST}/job?id=${SYNTH_ID}&result=1" -o server-understand.mp3

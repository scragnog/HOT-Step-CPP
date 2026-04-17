#!/usr/bin/env python3
# client-batch.py: test batching via ace-server
#
# POST /lm (lm_batch_size=2 in JSON) -> 2 enriched requests
# POST /synth (JSON array of 2 requests) -> 2 MP3s in one GPU batch
#
# Start the server first: ./server.sh

import json
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8085"


def post_json(endpoint, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        URL + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read(), resp.headers


def parse_multipart_mixed(data, content_type):
    """Parse multipart/mixed response into list of body bytes."""
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip().encode()
            break
    if not boundary:
        raise ValueError("no boundary in content-type: " + content_type)

    delimiter = b"--" + boundary
    parts = []

    for chunk in data.split(delimiter):
        if not chunk or chunk.startswith(b"--"):
            continue
        chunk = chunk.strip(b"\r\n")
        if not chunk:
            continue

        sep = chunk.find(b"\r\n\r\n")
        if sep < 0:
            continue
        body = chunk[sep + 4:]
        if body.endswith(b"\r\n"):
            body = body[:-2]
        parts.append(body)

    return parts


# Phase 1: LM generates N variations
try:
    with open("simple-batch.json") as f:
        request_json = json.load(f)
except FileNotFoundError:
    print("ERROR: simple-batch.json not found (run from the examples/ directory)")
    sys.exit(1)

try:
    lm_batch_size = request_json.get("lm_batch_size", 1)
    print("POST /lm (lm_batch_size=%d)..." % lm_batch_size)
    lm_data, _ = post_json("/lm", request_json)
except urllib.error.URLError as e:
    print("ERROR: cannot connect to %s (%s)" % (URL, e.reason))
    print("Start the server first: ./server.sh")
    sys.exit(1)
lm_results = json.loads(lm_data)
print("  -> %d enriched requests" % len(lm_results))

# Phase 2: synth all in one GPU batch (send JSON array)
print("POST /synth (batch=%d, JSON array)..." % len(lm_results))
body = json.dumps(lm_results).encode()
req = urllib.request.Request(
    URL + "/synth",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    resp_data = resp.read()
    content_type = resp.headers.get("Content-Type", "")

if "multipart/mixed" in content_type:
    parts = parse_multipart_mixed(resp_data, content_type)
    for i, mp3_data in enumerate(parts):
        path = "server-batch%d.mp3" % i
        with open(path, "wb") as f:
            f.write(mp3_data)
        print("  -> %s (%d bytes)" % (path, len(mp3_data)))
else:
    path = "server-batch0.mp3"
    with open(path, "wb") as f:
        f.write(resp_data)
    print("  -> %s (%d bytes)" % (path, len(resp_data)))

print("Done: %d MP3(s)" % len(lm_results))

"""Teacher-forced codec NLL probe: does the YuE2 LM read lyric annotations in the ABC?

For each song, the real codec stream is scored under one prompt per variant.
Style, lyric text and the notes of the score stay fixed; only the annotation
lines added under each Vocal music line change:

  plain    no annotation (reference)
  correct  the words sung in each bar, from forced-aligned word times
  shifted  every word annotated where the word one lyric line later was sung
  swapped  the words of two lyric sections exchanged

If the model reads the annotation, correct beats shifted and swapped, mostly in
the bars whose annotation moved. If it ignores it, the deltas are near zero.

Annotations are `w:` lines (bars separated by `|`) or `%` comments (--annot).
Every variant goes through the app's own cover transform (server/src/services/
backends/yue2/coverScoreTransform.ts, cover defaults: Vocal voice only, chords
stripped, free tempo) and the tool fails if a line is lost on the way. The
prompt is upstream's token_prefixes with cot=full, the same layout the engine
builds (yue2-probe --prefix-check).

Bar grid: bar 1 starts at audio time 0, each bar lasts its active M: at the
score's Q: tempo (inline M: changes are honoured). on_note is the share of
timed words whose midpoint lands in a bar where the Vocal voice has a note; a
low value means the grid does not fit the audio, so the song is skipped.

Runs the upstream reference model on the CPU (no GPU, no engine build).
usage (from the repo root, with the upstream venv's python):
  python engine/tools/yue2-w-nll-probe.py --manifest <yue2_preprocess.json> [...]
      [--per-manifest 1] [--limit 8] [--annot w|comment] [--max-frames N]
      [--model K:/yue2/models/YuE2-3B] [--out probe.json]
"""
import argparse, bisect, json, os, re, statistics, subprocess, time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VARIANTS = ("plain", "correct", "shifted", "swapped")
TAG = re.compile(r"^\s*\[[^\]]+\]\s*$")


def bars_of(abc):
    """Vocal bars (start_s, dur_s, has_note) and, per ABC line, its Vocal bar range."""
    tempo, beat, meter = 120.0, 0.25, 1.0
    in_vocal, bars, ranges, t = False, [], {}, 0.0
    for i, raw in enumerate(abc.split("\n")):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if re.match(r"^[A-Za-z]:", line):
            k = line[0]
            if k == "Q":
                m = re.match(r"^Q:\s*(?:(\d+)\s*/\s*(\d+)\s*=\s*)?(\d+(?:\.\d+)?)", line)
                if m:
                    beat = int(m[1]) / int(m[2]) if m[1] else 0.25
                    tempo = float(m[3])
            elif k == "M":
                m = re.match(r"^M:\s*(\d+)\s*/\s*(\d+)", line)
                if m:
                    meter = int(m[1]) / int(m[2])
            elif k == "V":
                first = (line[2:].split() or [""])[0]
                in_vocal = "vocal" in first.lower() or 'name="vocal' in line.lower()
            continue
        if not in_vocal:
            continue
        b0 = len(bars)
        for seg in (s for s in line.split("|") if s.strip()):
            music = re.sub(r'"[^"]*"', "", seg).strip()
            multi = re.fullmatch(r"Z(\d*)", music)
            n = max(1, int(multi[1] or 1)) if multi else 1
            sings = bool(re.search(r"[A-Ga-g]", music)) and not multi
            for _ in range(n):
                dur = meter / beat * 60.0 / tempo
                bars.append((t, dur, sings))
                t += dur
        ranges[i] = (b0, len(bars))
    return bars, ranges


def read_words(path, lyrics):
    rows = np.fromfile(path, dtype="<f4").reshape(-1, 5)
    cps = list(lyrics)
    return [dict(start=float(r[0]), end=float(r[1]), c0=int(r[3]), c1=int(r[4]), text="".join(cps[int(r[3]):int(r[4])]))
            for r in rows if np.isfinite(r[0]) and np.isfinite(r[1]) and r[4] > r[3]]


def lyric_spans(lyrics):
    """Codepoint spans of tagged blocks and of sung lines."""
    blocks, lines, off = [], [], 0
    for line in lyrics.replace("\r\n", "\n").split("\n"):
        n = len(line)
        if TAG.match(line):
            if blocks:
                blocks[-1][1] = off
            blocks.append([off, len(lyrics)])
        elif line.strip():
            lines.append((off, off + n))
        off += n + 1
    return blocks, lines


def assignments(words, bars, lyrics):
    starts = [b[0] for b in bars]
    end = bars[-1][0] + bars[-1][1]
    timed = [w for w in words if (w["start"] + w["end"]) / 2 < end]
    for w in timed:
        w["bar"] = bisect.bisect_right(starts, (w["start"] + w["end"]) / 2) - 1
    correct = [w["bar"] for w in timed]
    blocks, lines = lyric_spans(lyrics)
    # shifted: one lyric line = median timed words per line
    per_line = [sum(1 for w in timed if a <= w["c0"] < b) for a, b in lines]
    k = max(1, int(statistics.median([n for n in per_line if n] or [1])))
    shifted = [correct[min(i + k, len(correct) - 1)] for i in range(len(correct))]
    # swapped: first two blocks with >= 6 timed words, different text, disjoint bars
    members = [[i for i, w in enumerate(timed) if a <= w["c0"] < b] for a, b in blocks]
    norm = lambda idx: " ".join(timed[i]["text"].lower() for i in idx)
    pair = next(((x, y) for x in range(len(members)) for y in range(x + 1, len(members))
                 if len(members[x]) >= 6 and len(members[y]) >= 6 and norm(members[x]) != norm(members[y])
                 and set(correct[i] for i in members[x]).isdisjoint(correct[i] for i in members[y])), None)
    swapped = None
    if pair:
        swapped = list(correct)
        A, B = members[pair[0]], members[pair[1]]
        for src, dst in ((A, B), (B, A)):
            for j, i in enumerate(src):
                swapped[i] = correct[dst[j * len(dst) // len(src)]]
    on_note = sum(bars[b][2] for b in correct) / max(1, len(correct))
    return timed, dict(correct=correct, shifted=shifted, swapped=swapped), dict(phrase_words=k, swap_blocks=pair, on_note=on_note)


def annotate(abc, ranges, timed, bar_of, mode):
    per_bar = {}
    for w, b in zip(timed, bar_of):
        per_bar.setdefault(b, []).append(re.sub(r"\s+", " ", re.sub(r'[-_*~|\\"%]', " ", w["text"])).strip())
    out, added = [], 0
    for i, line in enumerate(abc.split("\n")):
        out.append(line)
        if i in ranges:
            b0, b1 = ranges[i]
            cells = [" ".join(x for x in per_bar.get(b, []) if x) for b in range(b0, b1)]
            if any(cells):
                out.append(("w: " if mode == "w" else "% ") + " | ".join(cells))
                added += 1
    return "\n".join(out), added


def cover_transform(abcs):
    """Run every ABC through the app's cover transform (defaults) via tsx."""
    script = ("import { transformCoverScore, COVER_SCORE_DEFAULTS } from './src/services/backends/yue2/coverScoreTransform.ts';"
              "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{process.stdout.write(JSON.stringify("
              "JSON.parse(s).map(a=>transformCoverScore(a,COVER_SCORE_DEFAULTS).renderedAbc)))});")
    server = os.path.join(REPO, "server")
    p = subprocess.run(["node", os.path.join(server, "node_modules", "tsx", "dist", "cli.mjs"), "--eval", script],
                       cwd=server, input=json.dumps(abcs), capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError("cover transform failed: " + p.stderr[-2000:])
    return json.loads(p.stdout)


def pick(manifests, per_manifest, limit):
    songs = []
    for m in manifests:
        man = json.load(open(m, encoding="utf-8"))
        taken = 0
        for src in man["sources"]:
            if taken >= per_manifest or len(songs) >= limit:
                break
            if not all(src.get(k) for k in ("abc", "lyrics", "codec_ids", "cursor_words")) or src.get("instrumental"):
                continue
            src = dict(src, base=os.path.dirname(os.path.abspath(m)))
            words = read_words(os.path.join(src["base"], src["cursor_words"]), src["lyrics"])
            bars, ranges = bars_of(src["abc"])
            if len(words) < 60 or not bars or not re.search(r"^Q:", src["abc"], re.M):
                continue
            timed, assign, info = assignments(words, bars, src["lyrics"])
            if assign["swapped"] is None or info["on_note"] < 0.6:
                print(f"skip {src['name']}: swap pair {info['swap_blocks']}, on_note {info['on_note']:.2f}", flush=True)
                continue
            songs.append((src, bars, ranges, timed, assign, info))
            taken += 1
    return songs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--per-manifest", type=int, default=1)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--annot", choices=("w", "comment"), default="w")
    ap.add_argument("--max-frames", type=int, default=0, help="score only the first N codec frames (0 = all)")
    ap.add_argument("--model", default="K:/yue2/models/YuE2-3B")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="yue2-w-nll-probe.json")
    a = ap.parse_args()

    import torch
    from yue2.protocol import SongRequest, token_prefixes, CODEC_OFFSET, MUSIC_END, CONTEXT
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from yue2.modeling_yue2 import YuE2ForCausalLM
    torch.set_num_threads(a.threads)

    songs = pick(a.manifest, a.per_manifest, a.limit)
    print(f"{len(songs)} songs", flush=True)
    tok = YuE2TextTokenizer(os.path.join(a.model, "qwen.tiktoken"))
    t0 = time.time()
    model = YuE2ForCausalLM.from_pretrained(a.model, local_files_only=True, torch_dtype=torch.float32,
                                            low_cpu_mem_usage=True).eval()
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    results = []
    for src, bars, ranges, timed, assign, info in songs:
        abcs, added = {"plain": src["abc"]}, {}
        for v in VARIANTS[1:]:
            abcs[v], added[v] = annotate(src["abc"], ranges, timed, assign[v], a.annot)
        rendered = dict(zip(VARIANTS, cover_transform([abcs[v] for v in VARIANTS])))
        marker = "w: " if a.annot == "w" else "% "
        for v in VARIANTS[1:]:
            kept = sum(1 for l in rendered[v].split("\n") if l.startswith(marker)) - \
                sum(1 for l in rendered["plain"].split("\n") if l.startswith(marker))
            if kept != added[v]:
                raise RuntimeError(f"{src['name']} {v}: {added[v]} annotation lines added, {kept} survived the cover transform")
        codes = np.fromfile(os.path.join(src["base"], src["codec_ids"]), dtype="<i4")
        if a.max_frames:
            codes = codes[:a.max_frames]
        targets = [int(c) + CODEC_OFFSET for c in codes] + ([MUSIC_END] if not a.max_frames else [])
        row = dict(song=src["name"], frames=int(len(codes)), words=len(timed), annotated_lines=added, **info, variants={})
        per_frame = {}
        for v in VARIANTS:
            prefix = token_prefixes(SongRequest(style=src.get("caption") or "", lyrics=src["lyrics"], cot="full",
                                                abc=rendered[v]), tok)
            ids = prefix + targets
            if len(ids) > CONTEXT:
                raise RuntimeError(f"{src['name']} {v}: {len(ids)} tokens exceeds context {CONTEXT}")
            t1 = time.time()
            with torch.inference_mode():
                x = torch.tensor([ids])
                h, _ = model.model(input_ids=x, position_ids=torch.arange(len(ids))[None], use_cache=False)
                h = h[0, len(prefix) - 1:-1]
                nll = []
                for c in range(0, h.shape[0], 1024):
                    lp = torch.log_softmax(model.lm_head(h[c:c + 1024]).float(), -1)
                    nll.append(-lp.gather(1, torch.tensor(targets[c:c + 1024])[:, None])[:, 0])
                nll = torch.cat(nll).numpy().astype(np.float64)
            per_frame[v] = nll
            row["variants"][v] = dict(prefix_tokens=len(prefix), nll_mean=float(nll[:len(codes)].mean()),
                                      seconds=round(time.time() - t1, 1))
            print(f"{src['name']} {v}: prefix {len(prefix)}, NLL {row['variants'][v]['nll_mean']:.5f} "
                  f"({time.time() - t1:.0f}s)", flush=True)
        # Where each counterfactual moved words: frames of bars whose annotation differs from correct.
        fps = 25.0
        for v in ("shifted", "swapped"):
            moved_bars = {b for b, c in zip(assign[v], assign["correct"]) if b != c} | \
                         {c for b, c in zip(assign[v], assign["correct"]) if b != c}
            mask = np.zeros(len(codes), bool)
            for b in moved_bars:
                s, d = bars[b][0], bars[b][1]
                mask[int(s * fps):int((s + d) * fps)] = True
            delta = per_frame[v][:len(codes)] - per_frame["correct"][:len(codes)]
            row["variants"][v].update(delta_vs_correct=float(delta.mean()),
                                      delta_moved=float(delta[mask].mean()) if mask.any() else None,
                                      delta_unmoved=float(delta[~mask].mean()) if (~mask).any() else None,
                                      moved_frames=int(mask.sum()))
        row["variants"]["correct"]["delta_vs_plain"] = float((per_frame["correct"] - per_frame["plain"])[:len(codes)].mean())
        results.append(row)
        json.dump(dict(annot=a.annot, model=a.model, songs=results), open(a.out, "w", encoding="utf-8"), indent=1)

    wins = sum(1 for r in results if r["variants"]["correct"]["nll_mean"] <
               min(r["variants"]["shifted"]["nll_mean"], r["variants"]["swapped"]["nll_mean"]))
    print(f"correct beats both counterfactuals on {wins}/{len(results)} songs; results in {a.out}")


if __name__ == "__main__":
    main()

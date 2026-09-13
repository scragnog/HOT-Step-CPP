# tools/mm3-retarget/retarget.py — shorten a track by removing a repeated section (2026-09-13)
#
# WHY: MM3's checkpoint caps audio at 9000 frames = 6:00 (mm3-model.h, max_audio_frames). MM3 LM training currently
# DROPS every dataset track longer than that (--drop-over-frames, mm3-lm-train-run.h). Across the library that is 269
# tracks. This tool tries to save them by cutting one interior span out of the song so the rest fits, keeping the intro
# and — the part whole-song training actually bought us — the real ending.
#
# HOW: the cut is not "delete the middle". It is a jump between two moments the song treats as interchangeable.
#   1. beat track -> pick the metric phase -> downbeat grid (librosa)
#   2. one feature vector per beat: chroma (harmony) + MFCC (timbre)
#   3. a jump from downbeat a to downbeat b sounds seamless when the music LEADING INTO each is the same, so the cost
#      is a decayed sum of distances over the W beats before a vs the W beats before b. Matching the two beats AT the
#      seam is the common mistake and does not work.
#   4. constraints: both ends on downbeats (so the bar grid stays in phase), the removal long enough to fit the cap,
#      and both ends inside a vocal gap (SuperSep stem energy, same signal as tools/vocal-end).
#   5. refuse when the best cost is above --max-cost: a through-composed track has no clean pair and an audible lurch
#      in the training set is worse than losing the track.
#   6. lyrics are forced-aligned to the vocal stem (torchaudio MMS_FA) so the lines inside the removed span can be
#      dropped. Forced alignment starts from the known text, which is NOT the free transcription that hallucinated on
#      stems during the endings investigation.
#
# Writes: edited audio, edited lyric sidecar, a ±N s seam snippet for auditioning, and a JSON report. Nothing is
# written next to the source; nothing here re-encodes codes. LISTEN TO THE SEAMS before any of this reaches a dataset.
#
# Usage:
#   py -3.13 tools/mm3-retarget/retarget.py <audio> [<audio> ...]
#     --target 360      seconds the result must fit inside (MM3 cap)
#     --margin 4        extra seconds to remove beyond the overrun
#     --bpb 4           beats per bar
#     --phase -1        force the downbeat phase (0..bpb-1), -1 = estimate
#     --lookback 8      beats of incoming context matched either side of the jump
#     --max-cost 0.45   refuse above this (calibrate by ear)
#     --guard 1.0       seconds around each cut end that must be vocal-free
#     --xfade 30        crossfade milliseconds at the join
#     --snippet 10      seconds either side of the seam in the audition clip
#     --alts 3          also render this many runner-up cuts as seam snippets
#     --no-vocal        skip SuperSep (no vocal constraint, no lyric edit) — fast iteration only
#     --no-lyrics       keep the vocal constraint but skip forced alignment
#     --out <dir>       default tools/mm3-retarget/out
#   Env: HOTSTEP_ROOT, HOTSTEP_ENGINE (default http://127.0.0.1:8085), MM3_RETARGET_OUT
import io, json, os, re, sys, time, hashlib, urllib.request, urllib.error
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('HOTSTEP_ROOT', os.path.abspath(os.path.join(HERE, '..', '..')))
ENGINE = os.environ.get('HOTSTEP_ENGINE', 'http://127.0.0.1:8085')

argv = sys.argv[1:]


def opt(name, default, cast=float):
    return cast(argv[argv.index(name) + 1]) if name in argv else default


TARGET   = opt('--target', 360.0)
MARGIN   = opt('--margin', 4.0)
BPB      = opt('--bpb', 4, int)
PHASE    = opt('--phase', -1, int)
PHRASE   = opt('--phrase', 4, int)   # removal must be a whole number of THIS many bars (1 disables)
LOOKBACK = opt('--lookback', 8, int)
MAX_COST = opt('--max-cost', 0.45)
GUARD    = opt('--guard', 1.0)
# Head and tail are NOT symmetric. Removing from 27 s in still leaves the song opening the way it opens, and Rob
# judged those early cuts good on 2026-09-13 ("it neatly cuts and puts the sections back together"), so the head is
# barely protected. The tail is different: a removal running to near the end deletes the approach to the ending, and
# the ending is the whole reason these tracks are being saved.
PROTECT_HEAD = opt('--protect-head', 8.0)
PROTECT_TAIL = opt('--protect-tail', 45.0)
XFADE_MS = opt('--xfade', 30.0)
SNIPPET  = opt('--snippet', 10.0)
N_ALTS   = opt('--alts', 3, int)
NO_VOCAL = '--no-vocal' in argv
ALIGN_MIX = '--align-mix' in argv   # align lyrics against the full mix when no stem is available. Degraded: the
                                    # backing track pulls word boundaries around. Debug and no-SuperSep fallback.
NO_LYRIC = '--no-lyrics' in argv or (NO_VOCAL and not ALIGN_MIX)
FORCE_CPU = '--cpu' in argv         # keep off a GPU that a training run is using
OUT      = os.environ.get('MM3_RETARGET_OUT', opt('--out', os.path.join(HERE, 'out'), str))
FLAGS    = {'--no-vocal', '--no-lyrics', '--align-mix', '--cpu'}
VALUED   = {'--target', '--margin', '--bpb', '--phase', '--lookback', '--max-cost',
            '--guard', '--xfade', '--snippet', '--alts', '--out', '--protect-head', '--protect-tail'}
files = [a for i, a in enumerate(argv)
         if not a.startswith('--') and (i == 0 or argv[i - 1] not in VALUED)]
os.makedirs(OUT, exist_ok=True)
STEMS = os.path.join(OUT, '_stems'); os.makedirs(STEMS, exist_ok=True)


# ── SuperSep vocal stem (same call as tools/vocal-end/vocal_end.py) ──────────────────────────────
def http_raw(url, data=None, timeout=1800):
    req = urllib.request.Request(url, data=data, method='POST' if data is not None else 'GET',
                                 headers={'content-type': 'application/octet-stream'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def separate(path):
    """Vocal stem as (mono float32, sr). Cached by basename + a hash of the absolute path."""
    tag = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:8]
    cache = os.path.join(STEMS, os.path.splitext(os.path.basename(path))[0] + f'.{tag}.vocals.wav')
    if os.path.exists(cache):
        y, sr = sf.read(cache, dtype='float32')
        return (y.mean(axis=1) if y.ndim > 1 else y), sr
    if path.lower().endswith('.wav'):
        body = open(path, 'rb').read()
    else:
        y0, sr0 = sf.read(path, dtype='float32')
        if sr0 != 44100:
            import librosa
            y0 = librosa.resample(y0.T, orig_sr=sr0, target_sr=44100).T
        buf = io.BytesIO(); sf.write(buf, y0, 44100, format='WAV', subtype='PCM_16'); body = buf.getvalue()
    sub = json.loads(http_raw(ENGINE + '/supersep/separate?level=4', body))
    jid = sub.get('job_id') or sub.get('id')
    if not jid:
        raise RuntimeError(f'supersep submit: {sub}')
    while True:
        time.sleep(3)
        try:
            r = json.loads(http_raw(f'{ENGINE}/supersep/result?id={jid}'))
            if r.get('stems'):
                break
        except urllib.error.HTTPError as e:
            if e.code not in (202, 404, 409, 425):
                raise
    names = [s.get('name', '') for s in r['stems']]
    idx = next((i for i, n in enumerate(names) if 'vocal' in n.lower()), None)
    if idx is None:
        raise RuntimeError(f'no vocal stem in {names}')
    wav = http_raw(f'{ENGINE}/supersep/serve?id={jid}&stem={idx}')
    try:
        http_raw(f'{ENGINE}/supersep/release?id={jid}', b'')
    except Exception:
        pass
    y, sr = sf.read(io.BytesIO(wav), dtype='float32')
    sf.write(cache, y, sr)
    return (y.mean(axis=1) if y.ndim > 1 else y), sr


def vocal_mask(stem, sr, hop=0.1, thresh_db=-30.0, min_len=0.6):
    """Boolean vocal-active array at `hop` resolution, runs shorter than min_len removed."""
    n = max(1, int(hop * sr)); m = len(stem) // n
    rms = np.array([np.sqrt(np.mean(stem[i * n:(i + 1) * n] ** 2)) + 1e-12 for i in range(m)])
    db = 20 * np.log10(rms / max(rms.max(), 1e-12))
    act = db > thresh_db
    i = 0
    while i < m:
        if act[i]:
            j = i
            while j < m and act[j]:
                j += 1
            if (j - i) * hop < min_len:
                act[i:j] = False
            i = j
        else:
            i += 1
    return act


def quiet_span(act, hop, t0, t1):
    """True when the vocal is inactive across [t0, t1]."""
    if act is None:
        return True
    a = max(0, int(np.floor(t0 / hop))); b = min(len(act), int(np.ceil(t1 / hop)) + 1)
    return b <= a or not act[a:b].any()


# ── analysis ────────────────────────────────────────────────────────────────────────────────────
def analyse(path):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, trim=False)
    beat_t = librosa.frames_to_time(beats, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1:]        # drop c0: that is loudness
    cs = librosa.util.sync(chroma, beats, aggregate=np.median)
    ms = librosa.util.sync(mfcc, beats, aggregate=np.mean)
    S = np.abs(librosa.stft(y, n_fft=2048))
    lowband = librosa.util.sync(S[:int(120 / (sr / 2048.0)) + 1].sum(axis=0, keepdims=True),
                                beats, aggregate=np.mean)[0]      # kick/bass weight per beat
    def unit(x):
        x = x - x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True); s[s < 1e-9] = 1.0
        x = x / s
        n = np.linalg.norm(x, axis=0, keepdims=True); n[n < 1e-9] = 1.0
        return x / n
    feat = np.vstack([unit(cs), unit(ms)])                        # harmony and timbre weighted alike
    feat = feat / np.maximum(np.linalg.norm(feat, axis=0, keepdims=True), 1e-9)
    dist = 1.0 - (feat.T @ feat)                                  # cosine distance, beats x beats
    np.clip(dist, 0.0, 2.0, out=dist)
    # Metric phase. Onset strength alone is close to a coin flip on rock (the backbeat is louder than the downbeat),
    # so three cues are z-scored and summed: attack energy, kick/bass weight, and harmonic change — chords turn at the
    # bar line far more reliably than the loudest hit lands on it.
    ob = np.array([onset[min(b, len(onset) - 1)] for b in beats])
    chg = np.concatenate([[0.0], np.linalg.norm(np.diff(cs, axis=1), axis=0)])
    def z(v):
        v = np.asarray(v, dtype=float)
        return (v - v.mean()) / max(v.std(), 1e-9)
    cue = z([ob[p::BPB].mean() if len(ob[p::BPB]) else 0.0 for p in range(BPB)]) \
        + z([lowband[p::BPB].mean() if len(lowband[p::BPB]) else 0.0 for p in range(BPB)]) \
        + z([chg[p::BPB].mean() if len(chg[p::BPB]) else 0.0 for p in range(BPB)])
    scores = list(map(float, cue))
    phase = PHASE if PHASE >= 0 else int(np.argmax(scores))
    sortd = sorted(scores, reverse=True)
    conf = (sortd[0] - sortd[1]) / max(abs(sortd[0]) + abs(sortd[1]), 1e-9) if len(sortd) > 1 else 1.0
    return dict(sr=sr, tempo=float(np.atleast_1d(tempo)[0]), beat_t=beat_t, dist=dist,
                phase=phase, phase_conf=float(conf), n_beats=len(beat_t))


def find_cuts(an, need, act, hop, n_keep=8):
    """Ranked (a, b) downbeat pairs whose removal covers `need` seconds. Cost = incoming-context match."""
    beat_t, dist, phase = an['beat_t'], an['dist'], an['phase']
    nb = len(beat_t)
    end_t = float(beat_t[-1]) if nb else 0.0
    dbs = [i for i in range(phase, nb, BPB)
           if i - LOOKBACK >= 0 and beat_t[i] >= PROTECT_HEAD and beat_t[i] <= end_t - PROTECT_TAIL]
    w = np.array([0.9 ** k for k in range(LOOKBACK)]); w /= w.sum()
    out = []
    for a in dbs:
        # the join plays ... beat_t[a-1] then jumps to beat_t[b]: match the run-in either side
        if not quiet_span(act, hop, beat_t[a] - GUARD, beat_t[a]):
            continue
        for b in dbs:
            if b <= a:
                continue
            # A whole number of bars keeps the meter continuous; a whole number of PHRASE bars also keeps the
            # hypermeter in phase, so the sections after the join still start where the ear expects them.
            if PHRASE > 1 and ((b - a) // BPB) % PHRASE:
                continue
            removal = beat_t[b] - beat_t[a]
            if removal < need:
                continue
            if b + 1 >= nb or not quiet_span(act, hop, beat_t[b], beat_t[b] + GUARD):
                continue
            ctx = float(sum(w[k] * dist[a - 1 - k, b - 1 - k] for k in range(LOOKBACK)))
            cost = ctx + 0.10 * (removal - need) / max(need, 1.0)   # prefer the smallest clean removal
            out.append(dict(a=a, b=b, t_a=float(beat_t[a]), t_b=float(beat_t[b]),
                            removal=float(removal), context_cost=round(ctx, 4), cost=round(cost, 4),
                            bars=int((b - a) / BPB)))
    out.sort(key=lambda c: c['cost'])
    # keep the ranked list sparse: near-identical neighbours of the winner teach nothing at audition
    keep = []
    for c in out:
        if all(abs(c['t_a'] - k['t_a']) > 5.0 or abs(c['t_b'] - k['t_b']) > 5.0 for k in keep):
            keep.append(c)
        if len(keep) >= n_keep:
            break
    return keep, len(out)


# ── lyric sidecar + forced alignment ────────────────────────────────────────────────────────────
def read_sidecar(audio_path):
    """(all lines, lyric block) from the dataset .txt sidecar. Once `lyrics:` starts, everything is lyrics."""
    p = os.path.splitext(audio_path)[0] + '.txt'
    if not os.path.exists(p):
        return None, None
    raw = open(p, encoding='utf-8').read().splitlines()
    for i, line in enumerate(raw):
        if line.strip().lower().startswith('lyrics:'):
            rest = line.split(':', 1)[1].strip()
            body = ([rest] if rest else []) + raw[i + 1:]
            return raw[:i], body
    return raw, []


def align_lines(stem, sr, lines):
    """Per-line (start, end) seconds via torchaudio MMS_FA on the vocal stem. None when it cannot align."""
    import torch, torchaudio
    from torchaudio.pipelines import MMS_FA as B
    dev = 'cpu' if FORCE_CPU else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = B.get_model().to(dev).eval()
    tokenizer, aligner = B.get_tokenizer(), B.get_aligner()
    if sr != 16000:
        import librosa
        stem = librosa.resample(stem, orig_sr=sr, target_sr=16000)
    wav = torch.tensor(stem, dtype=torch.float32, device=dev)[None, :]
    # one word list, but the encoder runs in chunks: self-attention over a 9-minute track does not fit, and the
    # per-chunk frame count is recorded so the frame->time map carries no accumulated drift.
    CH = int(20.0 * 16000)
    ems, ftime = [], []
    with torch.inference_mode():
        for s in range(0, wav.shape[1], CH):
            seg = wav[:, s:s + CH]
            if seg.shape[1] < 640:
                break
            em, _ = model(seg)
            ems.append(em[0])
            ftime.append(s / 16000.0 + np.arange(em.shape[1]) * (seg.shape[1] / 16000.0) / em.shape[1])
    if not ems:
        return None
    emission = torch.cat(ems, dim=0)
    ftime = np.concatenate(ftime)
    words, owner = [], []          # normalised word -> index of the lyric line it came from
    for li, line in enumerate(lines):
        if not line.strip() or line.strip().startswith('['):
            continue
        for w in re.findall(r"[A-Za-z']+", line):
            w = w.strip("'").lower()
            if w:
                words.append(w); owner.append(li)
    if not words:
        return None
    try:
        with torch.inference_mode():
            spans = aligner(emission.to(dev), tokenizer(words))
    except Exception as e:
        print(f'  [align] failed: {e}', file=sys.stderr)
        return None
    acc = {}
    for wi, sp in enumerate(spans):
        if not sp:
            continue
        t0 = float(ftime[min(sp[0].start, len(ftime) - 1)])
        t1 = float(ftime[min(sp[-1].end, len(ftime) - 1)])
        sc = float(np.mean([getattr(t, 'score', 1.0) for t in sp]))
        li = owner[wi]
        cur = acc.get(li)
        if cur is None:
            acc[li] = [t0, t1, [sc]]
        else:
            cur[0] = min(cur[0], t0); cur[1] = max(cur[1], t1); cur[2].append(sc)
    # CTC forced alignment places EVERY word somewhere, absent or not — the score is what separates a line the
    # singer actually sang from one the path was forced through. It is the only honest confidence here.
    return {li: (v[0], v[1], float(np.mean(v[2]))) for li, v in acc.items()}


def edit_lyrics(lines, per_line, t_a, t_b, min_score=0.0):
    """Drop lyric lines mostly sung inside the removed span; drop section tags left with nothing under them."""
    keep, dropped = [], []
    for li, line in enumerate(lines):
        sp = per_line.get(li) if per_line else None
        if sp and sp[2] >= min_score:
            ov = max(0.0, min(sp[1], t_b) - max(sp[0], t_a))
            if ov > 0.5 * max(sp[1] - sp[0], 1e-6):    # a half-cut line mismatches the audio either way
                dropped.append((li, line, round(sp[0], 1), round(sp[1], 1)))
                continue
        keep.append((li, line))
    out = [l for _, l in keep]
    pruned = []
    for i, line in enumerate(out):
        if line.strip().startswith('['):
            nxt = next((x for x in out[i + 1:] if x.strip()), None)
            if nxt is None or nxt.strip().startswith('['):
                continue
        pruned.append(line)
    return pruned, dropped


# ── render ──────────────────────────────────────────────────────────────────────────────────────
def splice(path, t_a, t_b, xfade_ms):
    """Original-rate, original-channel splice with an equal-power crossfade centred on the join."""
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    h = max(1, int(sr * xfade_ms / 2000.0))
    sa, sb = int(round(t_a * sr)), int(round(t_b * sr))
    sa = max(h, min(sa, len(y) - h)); sb = max(h, min(sb, len(y) - h))
    if sb - sa < 4 * h:
        raise RuntimeError('cut too short for the crossfade')
    ramp = (np.arange(2 * h) / (2 * h - 1)).astype(np.float32)[:, None]
    fade = y[sa - h:sa + h] * np.cos(ramp * np.pi / 2) ** 1 + y[sb - h:sb + h] * np.sin(ramp * np.pi / 2) ** 1
    out = np.concatenate([y[:sa - h], fade, y[sb + h:]], axis=0)
    return out, sr, (sa - h) / sr


def splice_mono(y, sr, t_a, t_b, xfade_ms):
    """The same cut applied to a mono analysis signal, so the edited vocal stem needs no second separation."""
    h = max(1, int(sr * xfade_ms / 2000.0))
    sa, sb = int(round(t_a * sr)), int(round(t_b * sr))
    sa = max(h, min(sa, len(y) - h)); sb = max(h, min(sb, len(y) - h))
    if sb - sa < 4 * h:
        return None
    r = (np.arange(2 * h) / (2 * h - 1)).astype(np.float32)
    fade = y[sa - h:sa + h] * np.cos(r * np.pi / 2) + y[sb - h:sb + h] * np.sin(r * np.pi / 2)
    return np.concatenate([y[:sa - h], fade, y[sb + h:]])


def main():
    if not files:
        print(__doc__ or 'no input files', file=sys.stderr)
        return 2
    report = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        info = sf.info(path)
        dur = info.frames / info.samplerate
        need = dur - TARGET + MARGIN
        print(f'\n=== {name}  {dur/60:.2f} min ===')
        if need <= 0:
            print(f'  already under {TARGET:.0f}s, nothing to do')
            continue
        print(f'  must remove >= {need:.1f}s')
        act, hop = None, 0.1
        stem = stem_sr = None
        if not NO_VOCAL:
            print('  separating vocals (SuperSep)...')
            stem, stem_sr = separate(path)
            act = vocal_mask(stem, stem_sr, hop=hop)
            print(f'  vocal-active {100*act.mean():.0f}% of the track')
        print('  beat tracking + features...')
        an = analyse(path)
        print(f"  {an['tempo']:.1f} bpm, {an['n_beats']} beats, phase {an['phase']}/{BPB} "
              f"(confidence {an['phase_conf']:.2f})")
        cands, n_all = find_cuts(an, need, act, hop)
        if not cands:
            print('  REFUSED: no downbeat pair clears the length and vocal-gap constraints')
            report.append(dict(file=path, status='no-candidate', need=round(need, 1)))
            continue
        best = cands[0]
        print(f'  {n_all} candidate pairs; best cost {best["cost"]:.3f} '
              f'(context {best["context_cost"]:.3f}) removing {best["removal"]:.1f}s '
              f'= {best["bars"]} bars at {best["t_a"]:.1f}s -> {best["t_b"]:.1f}s')
        status = 'ok' if best['context_cost'] <= MAX_COST else 'over-threshold'
        if status != 'ok':
            print(f'  WARNING: context cost {best["context_cost"]:.3f} > --max-cost {MAX_COST} '
                  f'— rendering anyway so it can be heard, but this is the refuse case')
        out_audio, sr, seam_t = splice(path, best['t_a'], best['t_b'], XFADE_MS)
        dst = os.path.join(OUT, f'{name}.edit.wav')
        sf.write(dst, out_audio, sr)
        s0 = max(0, int((seam_t - SNIPPET) * sr)); s1 = min(len(out_audio), int((seam_t + SNIPPET) * sr))
        sf.write(os.path.join(OUT, f'{name}.seam.wav'), out_audio[s0:s1], sr)
        for k, alt in enumerate(cands[1:1 + N_ALTS], start=1):
            try:
                ay, asr, at = splice(path, alt['t_a'], alt['t_b'], XFADE_MS)
                a0 = max(0, int((at - SNIPPET) * asr)); a1 = min(len(ay), int((at + SNIPPET) * asr))
                sf.write(os.path.join(OUT, f'{name}.alt{k}.seam.wav'), ay[a0:a1], asr)
            except Exception as e:
                print(f'  alt{k} render failed: {e}')
        rec = dict(file=path, status=status, duration=round(dur, 1),
                   edited=round(len(out_audio) / sr, 1), need=round(need, 1),
                   tempo=round(an['tempo'], 1), phase=an['phase'], phase_conf=round(an['phase_conf'], 2),
                   seam_at=round(seam_t, 2), best=best, alts=cands[1:1 + N_ALTS], n_candidates=n_all)
        head, lyr = read_sidecar(path)
        if lyr and not NO_LYRIC:
            if stem is not None:
                a_y, a_sr, a_src = stem, stem_sr, 'vocal stem'
            else:
                import librosa
                a_y, a_sr, a_src = librosa.load(path, sr=16000, mono=True), 16000, 'FULL MIX (degraded)'
                a_y = a_y[0] if isinstance(a_y, tuple) else a_y
            print(f'  forced-aligning lyrics to the {a_src}...')
            per_line = align_lines(a_y, a_sr, lyr)
            if per_line is None:
                print('  alignment unavailable — lyrics left untouched (FLAG: they no longer match the audio)')
                rec['lyrics'] = 'align-failed'
            else:
                kept, dropped = edit_lyrics(lyr, per_line, best['t_a'], best['t_b'])
                with open(os.path.join(OUT, f'{name}.edit.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join((head or []) + ['lyrics:'] + kept) + '\n')
                print(f'  dropped {len(dropped)} lyric line(s) inside the cut:')
                for _, line, a, b in dropped[:8]:
                    print(f'    [{a:7.1f}-{b:7.1f}] {line.strip()[:70]}')
                if len(dropped) > 8:
                    print(f'    ... and {len(dropped)-8} more')
                rec['lyrics'] = dict(dropped=len(dropped), kept=len(kept),
                                     lines=[d[1].strip() for d in dropped])
                # Verify rather than assume. Re-align the KEPT sheet against the EDITED audio: a sheet that still
                # matches scores about as well as it did before. This is what catches a lyric sheet whose repeats
                # were written once (so the removed span's words are still sung elsewhere and must NOT be dropped)
                # and the reverse, a line half-removed and left in.
                ed = splice_mono(a_y, a_sr, best['t_a'], best['t_b'], XFADE_MS)
                if ed is not None:
                    print('  verifying: re-aligning the edited sheet to the edited audio...')
                    v = align_lines(ed, a_sr, kept)
                    if v:
                        before = [per_line[li][2] for li in per_line
                                  if lyr[li].strip() and not lyr[li].strip().startswith('[')]
                        after = [x[2] for x in v.values()]
                        bad = sorted(((sc, kept[li]) for li, (_, _, sc) in v.items()), key=lambda z: z[0])[:3]
                        print(f'  alignment score {np.mean(before):.3f} before -> {np.mean(after):.3f} after '
                              f'({len(after)} lines)')
                        for sc, line in bad:
                            print(f'    weakest {sc:.3f}  {line.strip()[:64]}')
                        rec['lyrics']['score_before'] = round(float(np.mean(before)), 4)
                        rec['lyrics']['score_after'] = round(float(np.mean(after)), 4)
                        rec['lyrics']['weakest'] = [[round(sc, 4), l.strip()] for sc, l in bad]
        report.append(rec)
        print(f'  wrote {dst}  ({len(out_audio)/sr/60:.2f} min)')
    with open(os.path.join(OUT, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'\nreport: {os.path.join(OUT, "report.json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

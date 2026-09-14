# tools/mm3-retarget/lyric_align.py — Route A: whisper-as-anchor-source lyric line timing (2026-09-14)
#
# WHY THIS EXISTS: mm3-retarget (retarget.py / retarget-*.h) rescues dataset tracks longer than MM3's 9000-frame
# cap by cutting one interior span out of the song. It currently REFUSES any cut whose span contains singing,
# because the lyric sheet (sidecarIO.ts) is untimed plain text and the trainer builds its prompt from caption +
# that sheet verbatim — cut through singing without editing the sheet and the model is taught a prompt line has
# no audio, which is close to the sparse-vocals failure mode this project has already chased down once. Today's
# run proved the vocal-free-gap rule alone is not viable: 0 of 5 tracks on a vocal-dense pop album passed it.
#
# The fix needs per-line timings for the KNOWN lyric sheet against the real recording, so lines inside a cut can
# be dropped from the sheet before it ever reaches the trainer. This file is one of two candidate routes to that
# timing. (Route B, in parallel, is forced alignment — torchaudio MMS_FA — starting from the known text and never
# putting words in the model's mouth; see the "lyrics are forced-aligned" step in retarget.py.)
#
# THE TRAP THIS FILE EXISTS TO AVOID: whisper does not know the lyrics. It transcribes what it THINKS it hears and
# timestamps its OWN guess. On separated vocal stems it hallucinates hard — a previous investigation
# (tools/vocal-end/vocal_end.py) measured 654 "words" for a 199-word sheet, including words invented past the end
# of the file. Treating that transcript as ground truth would be exactly backwards. So it is never trusted as text
# — only mined for ANCHORS: sequence-align whisper's word list against the KNOWN sheet; wherever they agree, that
# whisper word's timestamp is a trustworthy anchor for the matching sheet word; wherever whisper invented
# something, the alignment should show it as an insertion that costs nothing to the sheet's own position, not a
# forced (and wrong) pairing that drags every anchor after it out of phase. Lines with no matched words get an
# honestly-labelled interpolated guess (confidence 0.0), never silently dropped.
#
# This is a STANDALONE PROTOTYPE. It does not touch the engine, the trainer, or retarget.py — only reads
# whisperTranscribe.ts and vocal_end.py for how this codebase already shells out to whisper-cli, and matches
# their flags/model discovery so a later integration (if Route A wins) has nothing new to debug.
#
# Usage:
#   py -3.13 tools/mm3-retarget/lyric_align.py <stem.wav> <sheet.txt> [--cut T_A T_B] [--json out.json]
#     --cut T_A T_B     report which sheet lines fall (>= --min-overlap) inside the removed span [T_A, T_B)
#     --json OUT        also write the table + summary as JSON
#     --model NAME      whisper model filename override (default: MODEL_PRIORITY search in models/whisper)
#     --gpu             let whisper-cli use the GPU (default: CPU-only, -ng — this machine is shared)
#     --min-overlap F   fraction of a line's own span that must sit inside the cut to count as "inside" (default 0.5)
#   Env: HOTSTEP_ROOT (checkout root, default = two levels up from this file), WHISPER_EXE, WHISPER_MODELS_DIR
#        (same variables server/src/config.ts reads, so this prototype and the app agree on where things live).
#
# Python 3.13, standard library + numpy only.

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ── Paths — mirror server/src/config.ts's whisper.exe / whisper.modelsDir resolution ───────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('HOTSTEP_ROOT', HERE.parent.parent))
WHISPER_CLI = Path(os.environ.get(
    'WHISPER_EXE', ROOT / 'tools' / 'whisper' / ('whisper-cli.exe' if os.name == 'nt' else 'whisper-cli')))
WHISPER_MODELS_DIR = Path(os.environ.get('WHISPER_MODELS_DIR', ROOT / 'models' / 'whisper'))

# Same fallback order as whisperTranscribe.ts's MODEL_PRIORITY — biggest/best first.
MODEL_PRIORITY = [
    'ggml-large-v3-turbo.bin',
    'ggml-large-v3.bin',
    'ggml-medium.bin',
    'ggml-base.bin',
]

WHISPER_LANGUAGE = 'en'      # dataset lyric sheets are English; whisper.cpp's 'auto' costs an extra detection pass
WHISPER_BEAM_SIZE = 5        # matches whisperTranscribe.ts's default
WHISPER_TIMEOUT_S = 1800     # matches vocal_end.py / retarget.py's other subprocess calls

# ── Needleman-Wunsch scoring constants — module-level and justified, not buried in the DP loop ─────────────────
#
# The one property everything below is tuned for: a hallucinated run of whisper words must come out as an
# INSERTION (heard-only gap), never as a chain of forced substitutions against real sheet words. If it became
# substitutions, every anchor after the hallucination would be shifted out of phase with it, which is worse than
# not aligning at all — a wrong timestamp looks exactly like a right one until you cut through the wrong line.
#
# That property falls out of one inequality: MISMATCH_PENALTY must be more negative than 2 * GAP_PENALTY. Then
# "pair these two unrelated words" (one mismatch) always scores worse than "skip both separately" (two gaps), so
# the DP never has a reason to force a bad pairing — it always prefers to treat a hallucinated stretch as pure
# insertion and a missed sheet word as pure deletion. With GAP_PENALTY = -1 and MISMATCH_PENALTY = -3, a
# substitution costs -3 against -2 for gap+gap: gaps win, every time, not just on average.
#
# CLOSE_MATCH exists because whisper mishears SUNG consonants constantly ("shine" heard as "shy", "heart" as
# "hard") — an edit distance of 1 on a word long enough that the collision isn't coincidental (4+ chars; short
# words are one edit apart from dozens of unrelated words) is still worth aligning, just not as confidently as an
# exact hit. 0.5 sits comfortably above the -2 a gap-gap pair would cost, so the DP takes the close match, and
# comfortably below the exact-match score, so line_times' confidence still reads a close match as weaker evidence.
#
# Gap penalty is a single linear constant (not affine / no separate gap-open cost). An affine penalty exists to
# stop the DP fragmenting one real deletion into many one-word gaps when a single long gap would score the same
# either way — but here every gap position is independently scored against real words on both sides (there is no
# run of "free" positions with no evidence), so fragmentation isn't a distinct failure mode from what the mismatch
# inequality above already prevents. Simple penalty keeps the DP a plain 2D table instead of three interacting
# ones, which matters more for a prototype meant to be read once and judged, not tuned for a decade.
MATCH_SCORE = 2.0
CLOSE_MATCH_SCORE = 0.5
# How much a close (one-edit) match is worth when scoring a LINE's confidence, as opposed to when scoring the
# alignment path. Deliberately not 1.0: see line_times.
CLOSE_MATCH_WEIGHT = 0.5
CLOSE_MATCH_MIN_LEN = 4
MISMATCH_PENALTY = -3.0
GAP_PENALTY = -1.0

WORD_RE = re.compile(r"[a-z']+")

# Lyric sheets and ASR do not agree on punctuation. Genius and Word write the curly right single quote (U+2019)
# in "don't"; whisper emits the ASCII one. Without folding, every contraction in the sheet tokenises as two words
# ("don", "t") while the heard side stays one, so it can never match — and the damage is invisible, showing up
# only as a depressed match rate, which is the number this tool is judged on. Accented letters have the same
# shape of problem: "cafe" with an acute truncated to "caf" under a bare [a-z] class.
_APOSTROPHES = {0x2019: "'", 0x2018: "'", 0x02BC: "'", 0x00B4: "'", 0x0060: "'"}


def fold_text(text: str) -> str:
    """Map apostrophe-likes to ASCII and strip diacritics, so sheet and transcript tokenise the same way."""
    folded = text.translate(_APOSTROPHES)
    decomposed = unicodedata.normalize('NFKD', folded)
    return ''.join(c for c in decomposed if not unicodedata.combining(c))
TAG_RE = re.compile(r'^\[.*\]$')
PUNCT_ONLY_RE = re.compile(r'^[\s.,!?;:\'"()\-–—…]+$')


# ── Data model ───────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Line:
    index: int          # position in the sheet, 0-based, counting only non-blank lines
    text: str            # original line text (stripped), for display/output — never the normalised form
    words: list[str]      # normalised words (empty for a tag line)
    is_tag: bool          # True for a whole-line section marker like "[Chorus]" — carries no audio


# ── Normalisation ────────────────────────────────────────────────────────────────────────────────────────────

def normalize_word(word: str) -> str:
    """Normalise ONE token (e.g. a single whisper word segment) to lowercase a-z + apostrophe.

    Takes the FIRST run of allowed characters, not all of them joined — a whisper token is already one word, so
    "Hello," -> "hello" is right, but concatenating unrelated runs across stripped punctuation (e.g. a hyphenated
    or slashed token) would invent a word that was never said. Returns '' if the token has no lowercase-alpha
    content at all (pure punctuation/numbers), which callers must treat as "not a word" and drop.
    """
    m = WORD_RE.findall(fold_text(word).lower())
    return m[0] if m else ''


def normalize_words(text: str) -> list[str]:
    """Split a whole line into normalised a-z+apostrophe tokens, in order. Unlike normalize_word, every run in the
    line becomes its own token — this is for splitting a lyric LINE into its constituent words, not cleaning up
    one already-segmented word."""
    return WORD_RE.findall(fold_text(text).lower())


# ── 2. parse_sheet ───────────────────────────────────────────────────────────────────────────────────────────

def parse_sheet(text: str) -> list[Line]:
    """Parse an untimed lyric sheet (sidecarIO.ts format) into Lines.

    Blank lines are skipped entirely (they carry neither audio nor position information worth tracking, and
    keeping them would just be another off-by-one to get wrong later). A line that is ENTIRELY a bracketed tag,
    e.g. "[Chorus]" or "[Verse 1]", is marked is_tag=True with an empty word list — it has no audio and must never
    be scored against whisper's transcript. `index` is dense over the lines this function actually keeps, so it is
    also the position used to walk "the previous/next line" during interpolation in line_times.
    """
    out: list[Line] = []
    idx = 0
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        is_tag = bool(TAG_RE.match(stripped))
        words = [] if is_tag else normalize_words(stripped)
        out.append(Line(index=idx, text=stripped, words=words, is_tag=is_tag))
        idx += 1
    return out


def flatten_sheet_words(sheet: list[Line]) -> list[str]:
    """The flat, tag-stripped word list `align()` expects as `sheet_words` — every non-tag line's words,
    concatenated in file order. `line_times` reconstructs this exact same flattening internally to map each
    `sheet_idx` out of `align()`'s pairs back to a line, so this helper exists specifically so both call sites
    can never drift apart: always build the array you hand to `align()` with this function, never by hand.
    """
    out: list[str] = []
    for line in sheet:
        if not line.is_tag:
            out.extend(line.words)
    return out


# ── 1. run_whisper ───────────────────────────────────────────────────────────────────────────────────────────

def find_whisper_model(preferred: str | None = None) -> Path | None:
    """Resolve a whisper GGML model path: `preferred` if it exists, else MODEL_PRIORITY in order, else any .bin
    in the models dir. Returns None if nothing is found (mirrors findWhisperModel in whisperTranscribe.ts)."""
    if preferred:
        p = WHISPER_MODELS_DIR / preferred
        if p.exists():
            return p
    for name in MODEL_PRIORITY:
        p = WHISPER_MODELS_DIR / name
        if p.exists():
            return p
    if WHISPER_MODELS_DIR.exists():
        for p in sorted(WHISPER_MODELS_DIR.glob('*.bin')):
            return p
    return None


def _parse_whisper_json(json_path: Path) -> list[tuple[str, float, float]]:
    """Parse whisper-cli's -oj output produced with --split-on-word --max-len 1: each `transcription[]` entry is
    already one word with its own offsets, so — unlike whisperTranscribe.ts's token-merging path for --dtw output
    — there is no sub-word reassembly to do here. Matches the segment-level reading vocal_end.py already uses for
    this exact invocation."""
    data = json.loads(json_path.read_text(encoding='utf-8'))
    words: list[tuple[str, float, float]] = []
    for seg in data.get('transcription', []):
        text = (seg.get('text') or '').strip()
        if not text or PUNCT_ONLY_RE.match(text):
            continue
        off = seg.get('offsets') or {}
        start = off.get('from', 0) / 1000.0
        end = off.get('to', 0) / 1000.0
        words.append((text, start, end))
    return words


def run_whisper(stem_wav: str | Path, model: str | None = None, cpu: bool = True) -> list[tuple[str, float, float]]:
    """Transcribe `stem_wav` (a vocal stem, ideally) with whisper-cli and return its own word list with its own
    timestamps: [(word_text, start_s, end_s), ...] in time order, exactly as whisper produced it — including any
    hallucinated words. Nothing here judges the transcript; that is align()'s job.

    Flags mirror whisperTranscribe.ts / vocal_end.py: -oj (JSON output), --split-on-word --max-len 1 (real
    per-word timestamps instead of segment timestamps), --no-prints, --language en, --beam-size 5.

    cpu=True (default) passes whisper-cli's -ng/--no-gpu flag — this prototype must never contend with a shared
    GPU on its own initiative. cpu=False is exposed for a later, deliberate opt-in and is NOT exercised by
    anything in this file.

    Caching: the whisper JSON is written next to the stem as `<stem_wav>.json` (whisper-cli's own default naming
    for -oj with no -of override) and is reused as-is if it already exists, so re-running against the same stem
    costs nothing. This cache is keyed ONLY on that path — it does not know if you asked for a different model or
    cpu setting since the last run. Delete the .json sidecar to force a fresh transcription.

    Raises FileNotFoundError if the stem, the whisper-cli binary, or a model can't be found, and RuntimeError if
    whisper-cli exits non-zero or produces no JSON — never returns a silently-empty result for a real failure.
    """
    stem_wav = Path(stem_wav)
    if not stem_wav.exists():
        raise FileNotFoundError(f'stem audio not found: {stem_wav}')

    json_path = stem_wav.parent / f'{stem_wav.name}.json'  # whisper-cli's default: appends ".json" to the WHOLE filename
    if not json_path.exists():
        if not WHISPER_CLI.exists():
            raise FileNotFoundError(f'whisper-cli not found at {WHISPER_CLI} (set WHISPER_EXE to override)')
        model_path = find_whisper_model(model)
        if model_path is None:
            raise FileNotFoundError(f'no whisper model found under {WHISPER_MODELS_DIR} (set WHISPER_MODELS_DIR)')

        args = [
            str(WHISPER_CLI),
            '-m', str(model_path),
            '-f', str(stem_wav),
            '-oj',                          # write <stem_wav>.json
            '--split-on-word',
            '--max-len', '1',
            '--beam-size', str(WHISPER_BEAM_SIZE),
            '--no-prints',
            '--language', WHISPER_LANGUAGE,
        ]
        if cpu:
            args.append('-ng')              # --no-gpu — see the cpu= docstring above

        proc = subprocess.run(args, capture_output=True, text=True, timeout=WHISPER_TIMEOUT_S)
        if proc.returncode != 0:
            try:
                json_path.unlink()          # don't leave a partial/stale sidecar behind after a failed run
            except FileNotFoundError:
                pass
            raise RuntimeError(f'whisper-cli exited {proc.returncode}: {proc.stderr[-1000:]}')
        if not json_path.exists():
            raise RuntimeError('whisper-cli exited cleanly but wrote no JSON output — check the args above')

    return _parse_whisper_json(json_path)


def normalize_heard_words(
    heard_raw: list[tuple[str, float, float]],
) -> tuple[list[str], list[tuple[str, float, float]]]:
    """Normalise whisper's raw word list for alignment, dropping any token that normalises to '' (a residual
    punctuation-only segment run_whisper's own filter missed). Returns (normalised_words, kept_tuples) as two
    PARALLEL lists — kept_tuples[k] is the (word, start, end) that normalised_words[k] came from, so a heard_idx
    out of align() indexes correctly into either one. `line_times` must be given `kept_tuples` as its `heard`
    argument, not the original `heard_raw` — the indices only agree with the filtered list.
    """
    words: list[str] = []
    kept: list[tuple[str, float, float]] = []
    for word, start, end in heard_raw:
        w = normalize_word(word)
        if not w:
            continue
        words.append(w)
        kept.append((w, start, end))
    return words, kept


# ── 3. align ─────────────────────────────────────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance, small-string DP. Only ever called on word-length strings from score_pair."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def score_pair(a: str, b: str) -> float:
    """Pairwise score used by the diagonal move in align()'s DP. See the scoring-constants comment above the
    constants themselves for why the three tiers (exact / close / mismatch) are set where they are."""
    if a == b:
        return MATCH_SCORE
    if len(a) >= CLOSE_MATCH_MIN_LEN and len(b) >= CLOSE_MATCH_MIN_LEN and _levenshtein(a, b) == 1:
        return CLOSE_MATCH_SCORE
    return MISMATCH_PENALTY


def align(sheet_words: list[str], heard_words: list[str]) -> list[tuple[int, int]]:
    """Global (Needleman-Wunsch) alignment of the known sheet against whisper's heard transcript, both as flat
    lists of normalised word tokens. Returns [(sheet_idx, heard_idx), ...] in increasing order — ONLY the pairs
    the traceback scored as an actual match or close match (score_pair > 0); a sheet word whisper never heard, and
    a whisper word invented from nothing, both fall out as gaps and are simply absent from this list, never paired
    with something wrong.

    Given the MISMATCH_PENALTY / GAP_PENALTY relationship documented above the constants, the optimal alignment
    should never contain a diagonal move that ISN'T a match or close match — two gaps always score better than one
    bad pairing — so the score_pair>0 filter below is a safety net for that guarantee, not load-bearing behaviour
    it depends on.
    """
    n, m = len(sheet_words), len(heard_words)
    if n == 0 or m == 0:
        return []

    F = np.zeros((n + 1, m + 1), dtype=np.float64)
    # Traceback move at [i, j]: 0 = diagonal (pair sheet[i-1] with heard[j-1]), 1 = up (sheet[i-1] unmatched,
    # i.e. a sheet word whisper never heard), 2 = left (heard[j-1] unmatched, i.e. a hallucinated/extra word).
    T = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        F[i, 0] = i * GAP_PENALTY
        T[i, 0] = 1
    for j in range(1, m + 1):
        F[0, j] = j * GAP_PENALTY
        T[0, j] = 2

    for i in range(1, n + 1):
        a = sheet_words[i - 1]
        row, prev_row = F[i], F[i - 1]
        for j in range(1, m + 1):
            diag = prev_row[j - 1] + score_pair(a, heard_words[j - 1])
            up = prev_row[j] + GAP_PENALTY
            left = row[j - 1] + GAP_PENALTY
            best, move = diag, 0
            if up > best:       # ties prefer diagonal, then up, then left — see traceback loop for why
                best, move = up, 1
            if left > best:
                best, move = left, 2
            row[j] = best
            T[i, j] = move

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = int(T[i, j])
        if i > 0 and j > 0 and move == 0:
            if score_pair(sheet_words[i - 1], heard_words[j - 1]) > 0:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# ── 4. line_times ────────────────────────────────────────────────────────────────────────────────────────────

def line_times(
    sheet: list[Line],
    pairs: list[tuple[int, int]],
    heard: list[tuple[str, float, float]],
) -> dict[int, tuple[float, float, float]]:
    """Turn word-level alignment pairs into a per-line span: {line_index: (start_s, end_s, confidence)}.

    `pairs` are (sheet_idx, heard_idx) as returned by align(sheet_words, heard_words) where sheet_words came from
    flatten_sheet_words(sheet) — this function reconstructs that same flattening to map each sheet_idx back to a
    line. `heard` must be the SAME list `heard_words` was normalised from (normalize_heard_words's second return
    value), so heard_idx indexes it correctly.

    A line's span is the min start / max end over its own matched words' timings. confidence is the fraction of
    that line's words that matched (0.0..1.0) — a tag line has zero words and so is always 0.0 by construction,
    same as a sung line whisper missed entirely. A line with NO matched words gets its span interpolated from nel
    neighbouring lines that DO have one (see below) and is marked confidence 0.0 regardless of how plausible the
    interpolated number looks — never let a guess read as measured.
    """
    line_word_count: dict[int, int] = {ln.index: len(ln.words) for ln in sheet}
    line_hits: dict[int, list[tuple[float, float]]] = {ln.index: [] for ln in sheet}

    flat_line_of_word: list[int] = []
    for ln in sheet:
        if not ln.is_tag:
            flat_line_of_word.extend([ln.index] * len(ln.words))

    sheet_words_flat = flatten_sheet_words(sheet)
    for sheet_idx, heard_idx in pairs:
        if not (0 <= sheet_idx < len(flat_line_of_word)) or not (0 <= heard_idx < len(heard)):
            continue  # defensive: mismatched sheet_words/heard vs. what align() actually saw
        line_idx = flat_line_of_word[sheet_idx]
        hw, hs, he = heard[heard_idx]
        # An exact match is evidence; a close match is a guess that happened to be one edit away. Counting both
        # as 1.0 lets a line whisper only half-heard report full confidence, which is precisely what this number
        # exists to prevent.
        exact = sheet_words_flat[sheet_idx] == normalize_word(hw)
        line_hits[line_idx].append((hs, he, 1.0 if exact else CLOSE_MATCH_WEIGHT))

    raw: dict[int, tuple[float, float, float] | None] = {}
    for ln in sheet:
        hits = line_hits[ln.index]
        if hits:
            start = min(h[0] for h in hits)
            end = max(h[1] for h in hits)
            total = line_word_count[ln.index]
            conf = (sum(h[2] for h in hits) / total) if total else 0.0
            raw[ln.index] = (start, end, conf)
        else:
            raw[ln.index] = None

    ordered = sheet
    known_positions = [p for p, ln in enumerate(ordered) if raw[ln.index] is not None]

    if not known_positions:
        # Total failure — whisper's transcript shares nothing with the sheet at all. Every line is unknown; hand
        # back zero-confidence placeholders rather than raising, so a caller building a table still gets one row
        # per line. The CLI's summary counters make this loud (match rate 0%, every line at confidence 0).
        return {ln.index: (0.0, 0.0, 0.0) for ln in ordered}

    result: dict[int, tuple[float, float, float]] = {ln.index: raw[ln.index] for ln in ordered if raw[ln.index] is not None}

    # Before the first anchor: can't know what came before it, so clamp to its own start (zero-length, conf 0).
    first_p = known_positions[0]
    first_start = raw[ordered[first_p].index][0]
    for p in range(0, first_p):
        result[ordered[p].index] = (first_start, first_start, 0.0)

    # After the last anchor: same idea, clamp to its own end.
    last_p = known_positions[-1]
    last_end = raw[ordered[last_p].index][1]
    for p in range(last_p + 1, len(ordered)):
        result[ordered[p].index] = (last_end, last_end, 0.0)

    # Between two anchors: split the gap evenly across however many unmatched lines sit inside it. This is a
    # straight-line guess, nothing more — it is what "interpolate from neighbours" means here.
    for a, b in zip(known_positions, known_positions[1:]):
        missing = list(range(a + 1, b))
        if not missing:
            continue
        gap_start = raw[ordered[a].index][1]
        gap_end = raw[ordered[b].index][0]
        if gap_end < gap_start:
            gap_end = gap_start  # a bad alignment can put an anchor "before" the previous one; never emit a negative span
        span = gap_end - gap_start
        k = len(missing)
        for i, p in enumerate(missing):
            s = gap_start + span * i / k
            e = gap_start + span * (i + 1) / k
            result[ordered[p].index] = (s, e, 0.0)

    return result


# ── 5. lines_in_span ─────────────────────────────────────────────────────────────────────────────────────────

def lines_in_span(
    line_times_: dict[int, tuple[float, float, float]],
    t_a: float,
    t_b: float,
    min_overlap: float = 0.5,
) -> list[int]:
    """The actual product: which sheet line indices are MOSTLY inside the removed span [t_a, t_b).

    "Mostly" means at least `min_overlap` of the line's own [start, end) sits inside the cut — a line that merely
    touches the cut boundary should survive; a line the cut swallows should not. Tag lines are not special-cased:
    if a section marker's interpolated position lands inside the cut it is reported too, since dropping it from
    the excised lyric sheet is harmless (it carries no words to protect) — callers that only care about singing
    can filter on Line.is_tag themselves.
    """
    if t_b <= t_a:
        raise ValueError(f'empty or inverted cut span: [{t_a}, {t_b})')
    hits = []
    for idx, (start, end, _conf) in line_times_.items():
        dur = end - start
        overlap = max(0.0, min(end, t_b) - max(start, t_a))
        if dur > 0:
            frac = overlap / dur
        else:
            # Zero-length placeholder (e.g. clamped before the first anchor): "inside" iff that instant is in the cut.
            frac = 1.0 if t_a <= start < t_b else 0.0
        if frac >= min_overlap:
            hits.append(idx)
    return sorted(hits)


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='Align a known lyric sheet against whisper-heard anchors, to find which lines fall inside a proposed cut.')
    ap.add_argument('stem_wav', help='vocal stem (or full mix) WAV to transcribe')
    ap.add_argument('sheet_txt', help='path to the untimed lyric sheet (plain text, [Tag] lines allowed)')
    ap.add_argument('--cut', nargs=2, type=float, metavar=('T_A', 'T_B'), help='report lines inside this removed span, in seconds')
    ap.add_argument('--json', metavar='OUT_JSON', help='also write the table + summary to this JSON file')
    ap.add_argument('--model', default=None, help='whisper model filename override, e.g. ggml-large-v3-turbo.bin')
    ap.add_argument('--gpu', action='store_true', help='let whisper-cli use the GPU (default: CPU-only, -ng)')
    ap.add_argument('--min-overlap', type=float, default=0.5, help='fraction of a line inside --cut to count as "inside" (default 0.5)')
    args = ap.parse_args(argv)

    sheet_text = Path(args.sheet_txt).read_text(encoding='utf-8')
    sheet = parse_sheet(sheet_text)
    sheet_words = flatten_sheet_words(sheet)

    heard_raw = run_whisper(args.stem_wav, model=args.model, cpu=not args.gpu)
    heard_words, heard = normalize_heard_words(heard_raw)

    pairs = align(sheet_words, heard_words)
    times = line_times(sheet, pairs, heard)

    cut_lines: set[int] = set()
    if args.cut:
        cut_lines = set(lines_in_span(times, args.cut[0], args.cut[1], args.min_overlap))

    # ── honesty-first summary — this number is the actual answer to "is whisper usable on sung vocals at all" ──
    n_heard = len(heard_words)
    n_sheet = len(sheet_words)
    n_matched = len(pairs)
    match_rate = (n_matched / n_sheet) if n_sheet else 0.0
    zero_conf = sum(1 for (_, _, c) in times.values() if c == 0.0)

    print(f'words heard (whisper, post-hallucination-filter): {n_heard}')
    print(f'words in known sheet:                              {n_sheet}')
    print(f'words matched (exact + close):                     {n_matched}  ({match_rate:.0%} of sheet)')
    print(f'lines at confidence 0.0 (unmatched/interpolated):   {zero_conf} / {len(times)}')
    if match_rate < 0.5:
        print('!! MATCH RATE BELOW 50% — whisper barely agrees with the known lyrics on this stem.')
        print('!! Treat every timing below as a rough guess, not a measurement.')
    print()

    header = f"{'line':>4}  {'cut':^3}  {'start':>8}  {'end':>8}  {'conf':>5}  text"
    print(header)
    print('-' * len(header))
    rows = []
    for ln in sheet:
        s, e, c = times[ln.index]
        inside = ln.index in cut_lines
        mark = '>>>' if inside else ('tag' if ln.is_tag else '')
        print(f'{ln.index:>4}  {mark:^3}  {s:8.1f}  {e:8.1f}  {c:5.2f}  {ln.text}')
        rows.append({'index': ln.index, 'text': ln.text, 'is_tag': ln.is_tag,
                     'start_s': s, 'end_s': e, 'confidence': c, 'in_cut': inside})

    if args.json:
        out = {
            'stem_wav': str(args.stem_wav),
            'sheet_txt': str(args.sheet_txt),
            'cut': args.cut,
            'summary': {
                'words_heard': n_heard,
                'words_in_sheet': n_sheet,
                'words_matched': n_matched,
                'match_rate': match_rate,
                'lines_zero_confidence': zero_conf,
                'lines_total': len(times),
            },
            'lines': rows,
        }
        Path(args.json).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f'\nwrote {args.json}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

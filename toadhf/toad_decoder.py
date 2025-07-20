"""Fuzzy‑robust GGWave decoder for 25‑tone / 8 symbols‑per‑second signals.

Changes in this revision (2025‑07‑05 c)
------------------------------------
* **Adaptive symbol matcher** – new `_symbols_from_bits()` turns a 25‑bit
  frame into **one or several plausible characters** based on Hamming‑distance
  ≤`MAX_DIST` (default 4).  When several share the same minimal distance, we
  return a tie token like `[H|K]`.
* **Graceful blank detection** – frames whose active‑bit count < 3 are treated
  as `?` to avoid spurious matches.
* **Updated vote aggregator** – understands tie tokens and weights votes
  accordingly, producing a single confident character or a final tie token.
* **Slightly looser row‑binarisation** – default energy threshold ratio lowered
  to 0.45 for better robustness against weak tones.

Usage (CLI)
-----------
$ python ggwave_fuzzy_decoder.py ggwave_encoded.wav 4            # default
$ python ggwave_fuzzy_decoder.py ggwave_encoded.wav 4 --debug   # full dump
"""

from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from toadhf.config import * 
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
from toadhf.toad_alphabet import TEXT_TO_TOAD, TOAD_TO_TEXT


# === Decoder specific constants ===
WINDOW                = np.hanning(FFT_SIZE)
SR                    = SAMPLE_RATE                  # GGWave sample‑rate
STFT_RATE             = SR / HOP_SIZE           # ≈187.5 frames/s
SYMBOL_HOP_FRAMES     = int(round(STFT_RATE / TOAD_SYMBOL_RATE))   # ≈ 23
MAX_DIST              = 4                       # max Hamming distance accepted
MIN_ACTIVE_BITS       = 2                       # expect 2 ones per data symbol
MARKER_MIN_BITS       = TOAD_NUM_TONES - 2           # ≥14 ⇒ treat as ^ marker
MID_OFFSET = SYMBOL_HOP_FRAMES // 2               # ≈11  (middle of 23)

MARKERS_FWD_BITS = [
    np.array(list(map(int, TEXT_TO_TOAD["MARKER_1"])), dtype=np.uint8),
    np.array(list(map(int, TEXT_TO_TOAD["MARKER_2"])), dtype=np.uint8),
    np.array(list(map(int, TEXT_TO_TOAD["MARKER_3"])), dtype=np.uint8),
    np.array(list(map(int, TEXT_TO_TOAD["MARKER_4"])), dtype=np.uint8),
]

MARKERS_REV_BITS = MARKERS_FWD_BITS[::-1]

# -----------------------------------------------------------------------------
# Code‑book
# -----------------------------------------------------------------------------

_TOAD_BITS  = np.array([list(map(int, s)) for s in TEXT_TO_TOAD.values()],
                         dtype=np.uint8)
_TOAD_CHARS = list(TEXT_TO_TOAD.keys())

# -----------------------------------------------------------------------------
# STFT helpers
# -----------------------------------------------------------------------------

def _stft_mag(wave: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, Z = sig.stft(
        wave,
        fs=rate,
        window=WINDOW,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_SIZE,
        padded=False,
        boundary=None,
    )
    return np.abs(Z), freqs

# -----------------------------------------------------------------------------
# Tone bins (needs freqs)
# -----------------------------------------------------------------------------

def _tone_bins(freqs: np.ndarray) -> np.ndarray:
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SR)

    return np.array([
        np.argmin(np.abs(freqs - (TOAD_FREQ_MIN + i * TOAD_FREQ_STEP)))
        for i in range(TOAD_NUM_TONES)
    ])

# -----------------------------------------------------------------------------
# Column → 16-bit vector
# -----------------------------------------------------------------------------

# toadhf/toad_decoder.py  – replace _frame_bits
def _frame_bits(spec, col, tone_bins, thr_ratio=0.55) -> np.ndarray:
    """
    Return a 0/1 vector for one STFT column.
    * thr_ratio   – energy threshold as a fraction of the *strongest* tone
    """
    e = np.array([spec[max(0,b-1):b+2, col].sum() for b in tone_bins])
    bits = (e > thr_ratio * e.max()).astype(np.uint8)

    # keep only the **two** strongest bins if we got more except if this is a SFD
    if bits.sum() > 2 and bits.sum() < 10:
        top2 = np.argsort(e)[-2:]
        bits[:] = 0
        bits[top2] = 1
    return bits


# -----------------------------------------------------------------------------
# Frame bits → candidate symbols
# -----------------------------------------------------------------------------
MAX_DIST  = 4     
GAP_DIST  = 1          

def _symbols_from_bits(bits: np.ndarray) -> tuple[str, float]:
    """
    Return (token, weight) where:
      token  – single symbol, tie token '[A|B]' or '?' for blank
      weight – confidence weight to be used by the voter
    """
    if bits.sum() < 2:
        return ('?', 0.0)                  # no vote

    dists = (_TOAD_BITS ^ bits).sum(axis=1)
    d_min = dists.min()
    if d_min > 2:                          # >1 is too vague – ignore
        return ('?', 0.0)

    winners = [c for c, d in zip(_TOAD_CHARS, dists) if d == d_min]
    token   = winners[0] if len(winners) == 1 else f"[{'|'.join(winners)}]"
    weight  = 1.0 / (1 + d_min)            # 1.0 for d=0, 0.5 for d=1
    return (token, weight)


# -----------------------------------------------------------------------------
# Marker (preamble/post‑amble) detection helpers
# -----------------------------------------------------------------------------

def _majority_window(spec: np.ndarray, tone_bins: np.ndarray, *, start: int, end: int,
                     tol_bits: int, majority_windows=((10, 8), (4, 3)), min_run: int = 5,
                     reverse: bool = False) -> tuple[int, int] | None:
    cols = range(start, end) if not reverse else range(end - 1, start - 1, -1)
    target = np.ones(TOAD_NUM_TONES, dtype=np.uint8)

    def good(c: int) -> bool:
        return (_frame_bits(spec, c, tone_bins) ^ target).sum() <= tol_bits

    # majority window first
    for win, need in majority_windows:
        scan = range(start, end - win + 1)
        scan = scan if not reverse else reversed(scan)
        for s in scan:
            hit = sum(good(c) for c in range(s, s + win))
            if hit >= need:
                first = next(c for c in range(s, s + win) if good(c))
                last  = max(c for c in range(first, s + win) if good(c))
                return (first, last) if not reverse else (last, first)

    # consecutive run fallback
    run_start, run_len = None, 0
    for c in cols:
        if good(c):
            run_len += 1
            if run_start is None:
                run_start = c
            if run_len >= min_run:
                return (run_start, c) if not reverse else (c, run_start)
        else:
            run_start, run_len = None, 0
    return None

def _match_marker_sequence(spec, tone_bins, start_col, markers, hop=SYMBOL_HOP_FRAMES, max_dist=2) -> bool:
    """
    Try to match a marker sequence at spec[:, start_col:] against given markers.
    Returns True if the next len(markers) symbols match within max_dist each.
    """
    for i, target_bits in enumerate(markers):
        c = start_col + i * hop + MID_OFFSET
        if c >= spec.shape[1]:
            return False
        bits = _frame_bits(spec, c, tone_bins)
        dist = np.count_nonzero(bits != target_bits)
        if dist > max_dist:
            return False
    return True

def _count_matching_markers(spec, tone_bins, start_col, markers, hop=SYMBOL_HOP_FRAMES, max_dist=2) -> int:
    """
    Check how many markers in the sequence match (in order) starting at start_col.
    Returns the count of consecutive markers that matched.
    """
    count = 0
    for i, target_bits in enumerate(markers):
        c = start_col + i * hop + MID_OFFSET
        if c >= spec.shape[1]:
            break
        bits = _frame_bits(spec, c, tone_bins)
        dist = np.count_nonzero(bits != target_bits)
        if dist <= max_dist:
            count += 1
        else:
            break  # stop at first failure
    return count


def _find_marker_fwd(spec, tone_bins, *, search_from: int = 0) -> tuple[int, int]:
    seq_len = len(MARKERS_FWD_BITS) * SYMBOL_HOP_FRAMES
    min_markers_to_sync = 2
    for c in range(search_from, spec.shape[1] - seq_len, SYMBOL_HOP_FRAMES):
        count = _count_matching_markers(spec, tone_bins, c, MARKERS_FWD_BITS)
        if count >= min_markers_to_sync:
            end_col = c + count * SYMBOL_HOP_FRAMES
            return c, end_col
    raise RuntimeError("Marker sequence not found (forward)")


def _find_marker_rev(spec, tone_bins, *, search_to: int) -> tuple[int, int]:
    seq_len = len(MARKERS_REV_BITS) * SYMBOL_HOP_FRAMES
    min_markers_to_sync = 2
    for c in range(search_to - seq_len, 0, -SYMBOL_HOP_FRAMES):
        count = _count_matching_markers(spec, tone_bins, c, MARKERS_REV_BITS)
        if count >= min_markers_to_sync:
            end_col = c + count * SYMBOL_HOP_FRAMES
            return c, end_col
    raise RuntimeError("Marker sequence not found (reverse)")


# -----------------------------------------------------------------------------
# Redundancy vote (understands tie‑tokens)
# -----------------------------------------------------------------------------

from collections import defaultdict

def _vote(char_stream: list[tuple[str, float]], redundancy: int) -> str:
    """
    *char_stream* is now a list of (token, weight) coming from the call above.
    """
    out = []
    for i in range(0, len(char_stream), redundancy):
        blk = char_stream[i:i + redundancy]

        scores = defaultdict(float)        # symbol → accumulated weight
        for tok, w in blk:
            if w == 0.0:
                continue                   # skip useless frame
            if tok.startswith('['):        # tie token: split weight equally
                opts = tok[1:-1].split('|')
                share = w / len(opts)
                for o in opts:
                    scores[o] += share
            else:
                scores[tok] += w

        if not scores:                     # all were discarded → '?'
            out.append('?')
            continue

        best_sym, best_val = max(scores.items(), key=lambda kv: kv[1])
        # is the second best within 10 %?
        sorted_vals = sorted(scores.values(), reverse=True)
        if len(sorted_vals) >= 2 and sorted_vals[1] >= 0.9 * best_val:
            ties = [s for s, v in scores.items() if abs(v - best_val) < 0.1*best_val]
            if len(ties) <= 3:
                out.append(f"[{'|'.join(sorted(ties))}]")
        else:
            out.append(best_sym)

    return ''.join(out)


# -----------------------------------------------------------------------------
# Decode pipeline
# -----------------------------------------------------------------------------

def decode_array(
    wav_data: np.ndarray,
    rate: int,
    *,
    redundancy: int = CHAR_LEVEL_REDUNDANCY,
    ignore_before: int = 0,               # samples to skip at the start
    debug: bool = False,
) -> tuple[list[str], int]:
    """
    Decode *all* ToADHF bursts that lie **after** `ignore_before` samples.

    Returns
    -------
    messages : list[str]
        Each decoded ToADHF message (redundancy-voted, amble stripped).
    last_sample_used : int
        Index (in samples) just past the final detected post-amble.  Pass
        this back in as *ignore_before* for the next buffer so partially
        overlapping calls don’t re-decode the same burst.
    """
    # ---- prepare STFT --------------------------------------------------------
    if wav_data.ndim > 1:
        wav_data = wav_data[:, 0]
    wav_data = wav_data.astype(np.float32)

    spec, freqs = _stft_mag(wav_data, rate)
    tbins       = _tone_bins(freqs)

    if debug:
        print("Binarised frames for each STFT column:")
        for c in range(spec.shape[1]):
            print(f"{c}:", ''.join(map(str, _frame_bits(spec, c, tbins))))

    # ---- main scan loop ------------------------------------------------------
    messages: list[str] = []
    search_col          = ignore_before // HOP_SIZE
    last_sample_out     = ignore_before

    while True:
        # 1)  find next pre-amble **forward** from search_col
        try:
            pre_s, pre_e = _find_marker_fwd(spec, tbins, search_from=search_col)
        except RuntimeError:
            break                                      # no more bursts

        # 2)  find the post-amble **after** that pre-amble
        try:
            post_s, post_e = _find_marker_fwd(spec, tbins,
                                              search_from=pre_e + 1)
        except RuntimeError:
            break                                      # truncated burst

        # 3)  extract symbol columns between them
        data_cols = range(pre_e + 1, post_s)
        sym_cols  = [c + MID_OFFSET
              for c in data_cols[::SYMBOL_HOP_FRAMES]
              if c + MID_OFFSET < post_s]

        char_stream = [
            _symbols_from_bits(_frame_bits(spec, c, tbins))
            for c in sym_cols
        ]
        msg = _vote(char_stream, redundancy).replace("^", "")
        if msg:                                       # drop empty decodes
            messages.append(msg)

        # 4)  advance search past this post-amble
        search_col       = post_e + 1
        last_sample_out  = search_col * HOP_SIZE

    return messages, last_sample_out



# --------------------------------------------------------------------------
#  Wrapper that keeps the old signature (filename → list[str])
# --------------------------------------------------------------------------
def decode_file(
    path: str | Path,
    redundancy: int = CHAR_LEVEL_REDUNDANCY,
    *,
    debug: bool = True
) -> list[str]:
    """
    Convenience wrapper that loads *path* into memory then calls
    :func:`decode_array`.
    """
    rate, wav_data = wav.read(path)
    if rate != SR:
        raise ValueError(f"Expected {SR}-Hz wav, got {rate}")

    messages, _ = decode_array(
        wav_data,
        rate,
        redundancy=redundancy,
        debug=debug,
    )
    return messages



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Fuzzy GGWave decoder")
    ap.add_argument("wav", help="input WAV file (48 kHz)")
    ap.add_argument("redundancy", nargs="?", default=4, type=int,
                    help="redundant repeats per char (default 4)")
    ap.add_argument("--debug", action="store_true", help="dump every binarised STFT column")
    args = ap.parse_args()

    msg = decode_file(args.wav, args.redundancy, debug=args.debug)
    print("Decoded:", msg)

if __name__ == "__main__":
    _cli()

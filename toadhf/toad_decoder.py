"""Fuzzy‑robust GGWave decoder for 25‑tone / 8 symbols‑per‑second signals.

Changes in this revision (2025‑07‑05 c)
------------------------------------
* **Adaptive symbol matcher** – new `_symbols_from_bits()` turns a 25‑bit
  frame into **one or several plausible characters** based on Hamming‑distance
  ≤ `MAX_DIST` (default 4).  When several share the same minimal distance, we
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


# === Decoder specific constants ===
WINDOW                = np.hanning(FFT_SIZE)
SR                    = SAMPLE_RATE                  # GGWave sample‑rate
STFT_RATE             = SR / HOP_SIZE           # ≈187.5 frames/s
SYMBOL_HOP_FRAMES     = int(round(STFT_RATE / TOAD_SYMBOL_RATE))   # ≈ 23
MAX_DIST              = 4                       # max Hamming distance accepted
MIN_ACTIVE_BITS       = 2                       # expect 2 ones per data symbol
MARKER_MIN_BITS       = TOAD_NUM_TONES - 2           # ≥14 ⇒ treat as ^ marker
MID_OFFSET = SYMBOL_HOP_FRAMES // 2               # ≈11  (middle of 23)

# -----------------------------------------------------------------------------
# Code‑book
# -----------------------------------------------------------------------------
from toadhf.toad_alphabet import TEXT_TO_TOAD, TOAD_TO_TEXT

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
# Column → 25‑bit vector
# -----------------------------------------------------------------------------

def _frame_bits(spec: np.ndarray, col: int, tone_bins: np.ndarray, thr_ratio: float = 0.45) -> np.ndarray:
    en = np.array([spec[max(0, b - 1): b + 2, col].sum() for b in tone_bins])
    return (en > thr_ratio * en.max()).astype(np.uint8)

# -----------------------------------------------------------------------------
# Frame bits → candidate symbols
# -----------------------------------------------------------------------------
MAX_DIST  = 4     
GAP_DIST  = 1          

def _symbols_from_bits(bits: np.ndarray) -> str:
    """Return a single char or a tie‑token like "[H|K]" or '?' for blank."""
    pop = bits.sum()
    if pop < MIN_ACTIVE_BITS:
        return '?'
    dists = (_TOAD_BITS ^ bits).sum(axis=1)
    d_min = dists.min()
    if d_min > MAX_DIST:
        return '?'

    # gap test ---------------------------------------------------------------
    second_best = np.partition(dists, 1)[1]     # 2nd smallest distance
    if second_best - d_min >= GAP_DIST:
        return _TOAD_CHARS[dists.argmin()]    # clear winner

    # otherwise keep current tie-token behaviour
    winners = [c for c, d in zip(_TOAD_CHARS, dists) if d == d_min]
    return f"[{'|'.join(winners)}]"

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


def _find_marker_fwd(spec: np.ndarray, tone_bins: np.ndarray, *, search_from: int = 0, **kw) -> tuple[int, int]:
    res = _majority_window(spec, tone_bins, start=search_from, end=spec.shape[1], reverse=False, tol_bits=2, **kw)
    if res is None:
        raise RuntimeError("Marker not found (forward)")
    return res


def _find_marker_rev(spec: np.ndarray, tone_bins: np.ndarray, *, search_to: int, **kw) -> tuple[int, int]:
    res = _majority_window(spec, tone_bins, start=0, end=search_to + 1, reverse=True, tol_bits=2, **kw)
    if res is None:
        raise RuntimeError("Marker not found (reverse)")
    return res

# -----------------------------------------------------------------------------
# Redundancy vote (understands tie‑tokens)
# -----------------------------------------------------------------------------

def _vote(char_stream: list[str], redundancy: int) -> str:
    blocks = [char_stream[i:i + redundancy] for i in range(0, len(char_stream), redundancy)]
    out = []
    for blk in blocks:
        scores: Counter[str] = Counter()
        for tok in blk:
            if tok == '?':
                continue
            if tok.startswith('[') and tok.endswith(']'):
                opts = tok[1:-1].split('|')
                w = 1 / len(opts)
                scores.update({o: w for o in opts})
            else:
                scores[tok] += 1
        max_v = max(scores.values()) if scores else 0
        winners = sorted(c for c, v in scores.items() if v == max_v)
        out.append(winners[0] if len(winners) == 1 else f"[{'|'.join(winners)}]")
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
    debug: bool = False
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

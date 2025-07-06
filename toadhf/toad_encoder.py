import numpy as np
import scipy.io.wavfile as wav
from toadhf.toad_alphabet import TEXT_TO_TOAD
import scipy.signal as sig
from scipy.signal.windows import tukey

from toadhf.config import *

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_nyq = 0.5 * TOAD_SAMPLE_RATE
_LOW_F      = TOAD_FREQ_MIN                     # 100 Hz
_HIGH_F     = TOAD_FREQ_MIN + TOAD_NUM_TONES * TOAD_FREQ_STEP  # e.g. 900 Hz
_SOS = sig.butter(4, [_LOW_F/_nyq, _HIGH_F/_nyq],
                  btype='band', output='sos')   # order 4 is plenty

def _raised_cosine(fade_samps: int) -> np.ndarray:
    """Half‑cosine window from 0→1 (len = fade_samps)."""
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samps)))

def _fade_envelope(n_samples: int, fade_ms: float, sr: int) -> np.ndarray:
    k = int(sr * fade_ms / 1000)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    env  = np.ones(n_samples, dtype=np.float32)
    env[:k]        = ramp          # fade-in
    env[-k:]       = ramp[::-1]    # fade-out
    return env

def _smooth_gate(bit_vec: np.ndarray, fade_samps: int) -> np.ndarray:
    """Return a smoothed 0/1 gate (raised‑cosine on/off)."""
    if fade_samps == 0:
        return bit_vec.astype(np.float32)

    win = _raised_cosine(fade_samps)
    # build symmetric window 0→1→0 (len = 2*fade)
    win_full = np.concatenate([win, win[::-1]])
    # normalise so plateau stays at 1 when consecutive symbols are 1‑1‑1 …
    win_full /= win_full.max()
    # Convolve & trim
    g = np.convolve(bit_vec.astype(np.float32), win_full, mode="same")
    return np.clip(g, 0.0, 1.0)


def bandpass(x: np.ndarray) -> np.ndarray:
    """Fast forward-only IIR band-pass (no phase-linear double pass)."""
    return sig.sosfilt(_SOS, x)

# -----------------------------------------------------------------------------
# Encoder (continuous‑phase, per‑tone raised‑cosine smoothing, power EQ)
# -----------------------------------------------------------------------------

def encode_text_to_waveform(text: str,
                            preamble_len: int = 8,
                            amplitude: float = 0.8,
                            fade_ms: float = 6.0,
                            guard_len: int = 1
                            ) -> np.ndarray:
    """
    Generate float32 base-band waveform with two active tones per symbol.
    Now each tone is multiplied by a raised-cosine gate so there are no hard
    on/off edges (reduces decoding ties & adjacent-channel splatter).
    """
    text        = " " + text                 # keep your leading blank
    sr          = TOAD_SAMPLE_RATE
    sym_dur     = 1.0 / TOAD_SYMBOL_RATE
    sym_samps   = int(round(sr * sym_dur))
    fade_samps  = int(sr * fade_ms / 1000)   # ≈ 288 @ 48 kHz

    # ------------------------------------------------------------------ symbols
    ones        = "1" * TOAD_NUM_TONES
    guard       = ["0" * TOAD_NUM_TONES] * guard_len
    pats        = ([ones] * preamble_len +
                   guard +
                   [TEXT_TO_TOAD.get(c, ones) for c in text] +
                   [ones] * preamble_len)
    bits        = np.array([[int(b) for b in p] for p in pats], dtype=np.uint8)

    total       = len(pats) * sym_samps
    t           = np.arange(total, dtype=np.float32) / sr
    freqs       = TOAD_FREQ_MIN + np.arange(TOAD_NUM_TONES) * TOAD_FREQ_STEP
    waveform    = np.zeros(total, dtype=np.float32)
    active_cnt  = np.zeros(total, dtype=np.float32)

    # ----------------------------------------------------------------- gate win
    # A Tukey window with α = 1 is a Bartlett (triangular); α = 0 is a rect.
    # We build ONE window of length 'sym_samps' and reuse it for all tones.
    if fade_samps > 0:
        alpha     = 2 * fade_samps / sym_samps      # fraction of tapered part
        win_single = tukey(sym_samps, alpha)
    else:
        win_single = np.ones(sym_samps, dtype=np.float32)

    gate_long = np.repeat(bits, sym_samps, axis=0) * np.tile(win_single, len(pats))[:, None]

    # ------------------------------------------------------------------ synth
    two_pi = 2 * np.pi
    for k, f in enumerate(freqs):
        phase     = two_pi * f * t
        tone      = np.sin(phase)
        gate_k    = gate_long[:, k]
        waveform += tone * gate_k
        active_cnt += gate_k

    waveform *= amplitude / np.max(np.abs(waveform))
    return waveform.astype(np.float32)

# -----------------------------------------------------------------------------
# Convenience I/O
# -----------------------------------------------------------------------------

def write_waveform_to_wav(waveform: np.ndarray, filename: str) -> None:
    wav.write(filename, TOAD_SAMPLE_RATE, waveform)


# -----------------------------------------------------------------------------
# Quick CLI test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    plain_msg = "HELLO WORLD"
    re_msg = "".join([c * 4 for c in plain_msg])
    wf  = encode_text_to_waveform(re_msg)
    write_waveform_to_wav(wf, "ggwave_encoded.wav")
    print("Saved encoded GGWave to ggwave_encoded.wav")

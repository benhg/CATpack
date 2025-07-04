import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import sys

# === FFT Parameters ===
FFT_SIZE = 1024
HOP_SIZE = FFT_SIZE / 3.55          # 256 samples → 48000/256 ≈ 187.5 Hz
WINDOW   = np.hanning(FFT_SIZE)


# === GGWave AudibleFast Parameters ===
GGWAVE_SAMPLE_RATE       = 48000
# STFT update rate = 48000/256 ≈ 187.5 Hz → to get 2 frames/symbol:
GGWAVE_SYMBOL_RATE       = int(round( (GGWAVE_SAMPLE_RATE/HOP_SIZE) / 2 ))
GGWAVE_NUM_TONES         = 25
GGWAVE_FREQ_MIN          = 1875.0
GGWAVE_FREQ_STEP         = 46.875
GGWAVE_SYMBOL_DURATION_S = 1.0 / GGWAVE_SYMBOL_RATE
GGWAVE_SYMBOL_HOP_FRAMES = 2     # exactly 2 STFT‐columns per symbol

# === Codebook Definitions ===
from ggwave_alphabet import TEXT_TO_GGWAVE, GGWAVE_TO_TEXT

# === Spectrogram & I/O ===
def compute_spectrogram(waveform, rate):
    _, _, spec = scipy.signal.stft(
        waveform, fs=rate, nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_SIZE, window=WINDOW,
        padded=False, boundary=None)
    return np.abs(spec)


def load_wav(filename):
    rate, data = wav.read(filename)
    if data.ndim > 1:
        data = data[:, 0]
    return rate, data.astype(np.float32)

# === Boundary Detection ===
def detect_start_frame(spectrogram, freqs):
    """
    Find the first run of at least 3 consecutive high-energy, all-ones frames.
    Returns (start_idx, end_idx, energy, threshold, min_idx, max_idx).
    """
    # locate frequency bin range for GGWave tones
    min_idx = np.argmin(np.abs(freqs - GGWAVE_FREQ_MIN))
    max_idx = np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + GGWAVE_FREQ_STEP * (GGWAVE_NUM_TONES - 1))))
    energy = np.sum(spectrogram[min_idx:max_idx+1, :], axis=0)
    threshold = 0.5 * np.mean(energy)

    # precompute tone-bin indices
    tone_bins = [
        np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + j * GGWAVE_FREQ_STEP)))
        for j in range(GGWAVE_NUM_TONES)
    ]
    all_ones = np.ones(GGWAVE_NUM_TONES, dtype=np.uint8)

    def frame_bits(col):
        # extract energy per tone at column `col`
        en = np.array([
            np.sum(spectrogram[max(0, b-1):min(b+2, spectrogram.shape[0]), col])
            for b in tone_bins
        ])
        return (en > 0.5 * en.max()).astype(np.uint8)

    # scan for 3+ consecutive high‐energy frames whose first two (or three) are all‐ones
    for i in range(len(energy) - 2):
        if energy[i] > threshold and energy[i+1] > threshold and energy[i+2] > threshold:
            # require the first 3 frames to have exactly the all‐ones pattern
            if (np.array_equal(frame_bits(i), all_ones)
                and np.array_equal(frame_bits(i+1), all_ones)
                and np.array_equal(frame_bits(i+2), all_ones)):
                # now extend the run until energy drops below threshold
                j = i + 3
                while np.array_equal(frame_bits(j), all_ones):
                    j += 1
                end_idx = j - 1
                print(f"[INFO] SFD starts at column {i}, ends at column {end_idx}")
                return i, end_idx, energy, threshold, min_idx, max_idx

    raise RuntimeError(
        "No valid SFD found: need ≥3 consecutive high‐energy, all-ones frames"
    )



def detect_end_frame(spectrogram, freqs, signal_start=0):
    """
    Find first column where at least 3 consecutive frames exceed threshold
    AND the first two frames have an all-ones tone pattern.
    Returns (index, energy, threshold, min_idx, max_idx).
    """
    # locate frequency bin range for GGWave tones
    min_idx = np.argmin(np.abs(freqs - GGWAVE_FREQ_MIN))
    max_idx = np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + GGWAVE_FREQ_STEP * (GGWAVE_NUM_TONES - 1))))
    energy = np.sum(spectrogram[min_idx:max_idx+1, :], axis=0)
    threshold = 0.5 * np.mean(energy)
    # precompute exact tone-bin indices
    tone_bins = [np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + j * GGWAVE_FREQ_STEP)))
                 for j in range(GGWAVE_NUM_TONES)]
    # helper to compute bit pattern for a given STFT column
    def frame_bits(col):
        energies = np.array([
            np.sum(spectrogram[max(0, b-1):min(b+2, spectrogram.shape[0]), col])
            for b in tone_bins
        ])
        max_e = np.max(energies)
        return (energies > 0.5 * max_e).astype(np.uint8)

    all_ones = np.ones(GGWAVE_NUM_TONES, dtype=np.uint8)
    # scan for 3 high-energy in a row
    for i in range(signal_start, len(energy) - 2):
        if (energy[i] > threshold and energy[i+1] > threshold and energy[i+2] > threshold):
            # require exact all-ones pattern in first two frames
            bits0 = frame_bits(i)
            bits1 = frame_bits(i+1)
            bits2 = frame_bits(i+2)
            if np.array_equal(bits0, all_ones) and np.array_equal(bits1, all_ones) and np.array_equal(bits2, all_ones):
                print(f"[INFO] EFD starts at column {i}")
                return i
    raise RuntimeError("No valid EFD found (need 3 consecutive high-energy frames with first two all-ones)")


# === Binarization & Trimming ===
def binarize_symbol_frames(symbol_frames, threshold_ratio=0.5):
    bits = []
    for frame in symbol_frames:
        max_e = np.max(frame)
        bits.append((frame > (threshold_ratio * max_e)).astype(np.uint8))
    return np.array(bits)


def trim_and_binarize_between(spectrogram, freqs, metadata_frames=11, threshold_ratio=0.5):
    try:
        start, start_e, energy, thresh, mi, ma = detect_start_frame(spectrogram, freqs)
        end = detect_end_frame(spectrogram, freqs, signal_start=start_e+metadata_frames)
    except:
        start = 0
        end = len(spectrogram)
    idxs = list(range(start, end, GGWAVE_SYMBOL_HOP_FRAMES))
    tone_bins = [np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + i * GGWAVE_FREQ_STEP)))
                 for i in range(GGWAVE_NUM_TONES)]
    frames = []
    for t in idxs:
        frames.append(np.array([
            np.sum(spectrogram[max(0, b-1):min(b+2, spectrogram.shape[0]), t])
            for b in tone_bins]))
    bits_all = binarize_symbol_frames(np.array(frames), threshold_ratio)
    sfd_pat = np.ones(GGWAVE_NUM_TONES, dtype=np.uint8)
    is_data = np.any(bits_all != sfd_pat, axis=1)
    first_d = np.argmax(is_data)
    last_d = len(is_data)-1 - np.argmax(is_data[::-1])
    start_p = first_d + metadata_frames
    end_p = last_d - metadata_frames
    return bits_all[start_p:end_p]

# === Decoding Helpers ===
def decode_payload_runs(bits):
    bitstrs = [''.join(str(x) for x in frame) for frame in bits]
    chars = [GGWAVE_TO_TEXT.get(bs, '?') for bs in bitstrs]
    runs, prev, count = [], chars[0], 1
    for c in chars[1:]:
        if c == prev:
            count += 1
        else:
            runs.append(count)
            prev, count = c, 1
    runs.append(count)
    return runs


def estimate_frames_per_char(runs):
    return int(np.median(runs))


# === Fuzzy Decode ===
def decode_message_fuzzy(bits, frames_per_char=4):
    """
    Perform a majority-vote across each block of `frames_per_char` frames.
    If one symbol wins >50%, pick it. If a 2-2 tie, return both as "[a|b]".
    """
    bitstrs = [''.join(str(x) for x in frame) for frame in bits]
    total = len(bitstrs)
    char_count = total // frames_per_char
    decoded = []
    for i in range(char_count):
        slice_ = bitstrs[i*frames_per_char:(i+1)*frames_per_char]
        # map each frame to char
        chars = [GGWAVE_TO_TEXT.get(bs, '?') for bs in slice_]
        # count votes
        votes = {}
        for c in chars:
            votes[c] = votes.get(c, 0) + 1
        # find winners
        max_votes = max(votes.values())
        winners = [c for c, v in votes.items() if v == max_votes]
        if len(winners) == 1:
            decoded.append(winners[0])
        else:
            # tie -> list both
            decoded.append("[" + "|".join(sorted(winners)) + "]")
    return ''.join(decoded)


# === Utility: Frame/Sample Calculators ===
def calc_hop_size_for_target_frames(target_frames, sample_rate=GGWAVE_SAMPLE_RATE, symbol_rate=GGWAVE_SYMBOL_RATE):
    return int(sample_rate / (symbol_rate * target_frames))


def calc_sample_rate_for_target_frames(target_frames, hop_size=HOP_SIZE, symbol_rate=GGWAVE_SYMBOL_RATE):
    return int(hop_size * symbol_rate * target_frames)

# === Main ===
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.wav>")
        sys.exit(1)

    target = 4
    hop_needed = calc_hop_size_for_target_frames(target)
    rate_needed = calc_sample_rate_for_target_frames(target)
    #print(f"For ~{target} frames/symbol at {GGWAVE_SAMPLE_RATE}Hz, HOP_SIZE≈{hop_needed}")
    #print(f"Or keep HOP_SIZE={HOP_SIZE} and use sample rate≈{rate_needed}Hz")

    rate, wavf = load_wav(sys.argv[1])
    if rate != GGWAVE_SAMPLE_RATE:
        print(f"Warning: input rate {rate} != expected {GGWAVE_SAMPLE_RATE}")
    spec = compute_spectrogram(wavf, rate)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0/rate)

    bits = trim_and_binarize_between(spec, freqs)
    for i, b in enumerate(bits):
        print(f"[DEBUG] {i}: {''.join(str(x) for x in b)}")

    #runs = decode_payload_runs(bits)
    fpc = 4
    msg = decode_message_fuzzy(bits, fpc)
    print(f"Frames/char: {fpc}")
    print(f"Decoded: {msg}")

if __name__ == '__main__':
    main()

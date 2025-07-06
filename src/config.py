"""
Save configuration options here
"""
import numpy as np

from radio_common import *

# === Signal Processing ===
SAMPLE_MULTIPLICATION_FACTOR = 1
CHAR_LEVEL_REDUNDANCY = 2

# === Radio and I/O ===
RADIO_CLASS = IC7300
# If None, use default from Radio class
RADIO_AUDIO_NAME = None 
RADIO_CAT_PORT = None
RADIO_BAUD_RATE = None

# === FFT Parameters ===
FFT_SIZE              = 2048
HOP_SIZE              = FFT_SIZE // 4
WINDOW   = np.hanning(FFT_SIZE)

# === Audio Options ===
SAMPLE_RATE = 48_000          # Hz. Sample rate for audio
TOAD_SAMPLE_RATE = 48_000
AUDIO_CHUNK_LEN = 10 # Seconds. Length of chunks to record for

# === ToAD Mode Configurations ===
TOAD_SYMBOL_RATE = 8               # 1 symbol = 1/8 s
TOAD_NUM_TONES   = 16
TOAD_FREQ_MIN    = 100.0          # Hz of the first bin
TOAD_FREQ_STEP   = 50         # Hz spacing between bins
TOAD_AMBLE_LENGTH = 8 # Number of frames for preamble/postamble. Must be even.

if (TOAD_FREQ_MIN + (TOAD_NUM_TONES * TOAD_FREQ_STEP)) > 3000:
	print("WARNING: TOAD frequency range should be <= 3000 Hz")

# === Internal options ===
DEBUG_MODE = True # in debug mode, print debug logs and echo own transmissions
SAVE_RECORDINGS = True # Save or throw away the 10s recordings at exit
RECORDINGS_DIR = "recordings/"


# ggwave_terminal.py
from radio_common import IC7300, K3S
from toad_decoder import decode_file
from toad_encoder import encode_text_to_waveform

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
import threading, time, os, sys

import ggwave
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import collections
import os
import datetime
import atexit
from scipy.signal import butter, lfilter, spectrogram 

from config import *

import numpy as np

def listen_loop(session, radio, device="USB Audio CODEC", samplerate=48000):
    record_duration = 10.0
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)

    with patch_stdout():
        print(f"[ToAD] Listening on {radio} ('{device}')")
        stream = sd.InputStream(device=device, channels=1, samplerate=samplerate, dtype='float32')
        stream.start()

        while True:
            audio, _ = stream.read(int(record_duration * samplerate))
            filename = os.path.join(output_dir, f"toad_{time.strftime('%Y%m%d_%H%M%S')}.wav")
            sf.write(filename, audio, samplerate)

            try:
                results = decode_file(filename)
            except RuntimeError:
                print(f"[ToAD] No decode from last {record_duration}s")
                continue

            for msg in results:
                print(f"[RECV] {msg}")

def main():
    radio = RADIO_CLASS()
    if RADIO_AUDIO_NAME is not None:
        radio.audio = RADIO_AUDIO_NAME

    if RADIO_CAT_PORT is not None:
        radio.device = RADIO_CAT_PORT

    if RADIO_BAUD_RATE is not None:
        radio.baud = RADIO_BAUD_RATE

    freq = float(input("Enter operating frequency in KHz: ").replace(",", ""))

    radio.set_freq(int(freq*1000))

    if freq >= 10000:
        radio.set_mode('DATA-U')
    else:
        radio.set_mode('DATA-L')
    samplerate = SAMPLE_RATE

    # These functions are idempotent so they can be called more than once.
    atexit.register(radio.ptt_off)
    atexit.register(radio.close)

    # Build our PromptSession once
    session = PromptSession("[SEND] > ")

    # Start listener thread, passing the session so it can print above it
    rx_thread = threading.Thread(target=listen_loop, args=(session, radio), daemon=True)
    rx_thread.start()
    time.sleep(0.5)

    try:
        while True:
            # prompt_toolkit will redraw the [SEND] > prompt after any print()
            text = session.prompt().upper()
            multiplied_text = ""
            for char in text:
                multiplied_text += char*CHAR_LEVEL_REDUNDANCY
            payload = encode_text_to_waveform(multiplied_text, preamble_len=TOAD_AMBLE_LENGTH)

            with radio.tx_lock:
                radio.ptt_on()
                samples = SAMPLE_MULTIPLICATION_FACTOR * np.frombuffer(payload, dtype=np.float32)
                padding = np.zeros(int(0.02 * samplerate), dtype=np.float32)
                sd.play(np.concatenate([padding, samples, padding]), samplerate=samplerate,
                        device=radio.audio)
                sd.wait()
                radio.ptt_off()
    except sd.PortAudioError:
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        radio.ptt_off()
        radio.close()

if __name__ == '__main__':
    main()

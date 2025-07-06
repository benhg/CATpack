# ggwave_terminal.py
from toadhf.radio_common import IC7300, K3S
from toadhf.toad_decoder import decode_file
from toadhf.toad_encoder import encode_text_to_waveform
from toadhf.listener import listen_loop
from toadhf.config import *

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
import threading, time, os, sys

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
import argparse

import numpy as np


# --- Serial ports ------------------------------------------------------------
try:
    from serial.tools import list_ports
except ImportError:   # pyserial not installed
    list_ports = None

# --- Audio devices -----------------------------------------------------------
try:
    import sounddevice as sd
except ImportError:   # sounddevice not installed
    sd = None


stop_event = threading.Event()

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
    rx_thread = threading.Thread(target=listen_loop, args=(session, radio, stop_event), daemon=True)
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
        stop_event.set()
        print("[ToAD] Shutting down...")
        sys.exit(0)
    finally:
        rx_thread.join(timeout=AUDIO_CHUNK_LEN + 0.25)
        radio.ptt_off()
        radio.close()
        print("[ToAD] Exiting")

def scan_serial_ports() -> list[tuple[str, str]]:
    """
    Return a list of `(device, description)` tuples for all serial/USB ports.

    Requires **pyserial**.  Returns an empty list if pyserial is missing.
    """
    if list_ports is None:
        return []

    ports = []
    for p in list_ports.comports():
        ports.append((p.device, p.description))
    return ports


def scan_audio_devices() -> list[dict]:
    """
    Return a list of dicts with keys::

        {index, name, max_input_channels, max_output_channels, default_samplerate}

    Requires **sounddevice**.  Returns an empty list if sounddevice is missing.
    """
    if sd is None:
        return []

    devices = []
    for idx, info in enumerate(sd.query_devices()):
        devices.append(
            dict(
                index=idx,
                name=info["name"],
                in_ch=info["max_input_channels"],
                out_ch=info["max_output_channels"],
                rate=int(info["default_samplerate"]),
            )
        )
    return devices


# -----------------------------------------------------------------------------


def print_device_summary() -> None:  # noqa: D401
    """Print a human-friendly overview of serial and audio devices."""
    # Serial
    print("\n=== Serial / USB–COM Ports ===")
    serials = scan_serial_ports()
    if not serials:
        print("  (no serial ports found or pyserial not installed)")
    else:
        for dev, desc in serials:
            print(f"  {dev:15}  {desc}")

    # Audio
    print("\n=== Audio Devices (sounddevice) ===")
    audio = scan_audio_devices()
    if not audio:
        print("  (no audio devices found or sounddevice not installed)")
    else:
        hdr = f"{'Idx':>3}  {'In':>2}/{ 'Out':<3}  {'Rate':>6}  Name"
        print(hdr)
        print("-" * len(hdr))
        for d in audio:
            io = f"{d['in_ch']:>2}/{d['out_ch']:<3}"
            print(f"{d['index']:>3}  {io}  {d['rate']:>6}  {d['name']}")

if __name__ == '__main__':
    main()

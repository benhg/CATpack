from __future__ import annotations
import os, time
import numpy as np
import sounddevice as sd, soundfile as sf
from prompt_toolkit.patch_stdout import patch_stdout

from toadhf.toad_decoder import decode_array
from toadhf.config import (
    RECORDINGS_DIR, AUDIO_CHUNK_LEN, SAMPLE_RATE, SAVE_RECORDINGS,
    CHAR_LEVEL_REDUNDANCY
)

# ----------------------------------------------------------------------
ROLLING_SEC = 30.0          # “live” part we keep
GUARD_SEC   = 2.0           # extra tail kept to catch trailing post-amble
MAX_BUF     = int((ROLLING_SEC + GUARD_SEC) * SAMPLE_RATE)


def listen_loop(session, radio, stop_event,
                *, device="USB Audio CODEC", samplerate=SAMPLE_RATE):
    chunk_frames   = int(AUDIO_CHUNK_LEN * samplerate)
    last_save_time = time.time()

    buf = np.empty(0, dtype=np.float32)

    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    with patch_stdout():
        print(f"[ToAD] Listening on {radio} ('{device}')")
        try:
            with sd.InputStream(device=device,
                                channels=1,
                                samplerate=samplerate,
                                dtype='float32') as stream:

                while not stop_event.is_set():
                    try:
                        chunk, _ = stream.read(chunk_frames)
                    except sd.PortAudioError as e:
                        if stop_event.is_set():
                            break
                        print("[ToAD] PortAudio error:", e)
                        continue

                    buf = np.concatenate((buf, chunk[:, 0]))
                    if buf.size > MAX_BUF:
                        # keep the newest MAX_BUF samples
                        buf = buf[-MAX_BUF:]

                    # optional wav logging ----------------------------------
                    if SAVE_RECORDINGS and time.time() - last_save_time >= SAVE_RECORDINGS:
                        ts = time.strftime('%Y%m%d_%H%M%S')
                        sf.write(os.path.join(RECORDINGS_DIR, f"toad_{ts}.wav"),
                                 buf, samplerate)
                        last_save_time = time.time()

                    # try decoding -------------------------------------------
                    try:
                        messages, consumed = decode_array(
                            buf, samplerate,
                            redundancy=CHAR_LEVEL_REDUNDANCY,
                            ignore_before=0
                        )
                        for m in messages:
                            print(f"[RECV] {m.lstrip()}")
                    except RuntimeError:
                        consumed = 0

                    # *Start* of the next burst is safer to cut at ----------
                    if consumed:
                        buf = buf[consumed:]

        finally:
            print("[ToAD] Listener thread stopping…")

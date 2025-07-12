# 🐸 ToAD-HF  
**Text-over-Audio Digital for HF radios**  
_A single-repo sound-card mode for quick keyboard QSOs via data-over-sound — **and a re-usable modem library**_

---

## 1 | What is ToAD-HF?  
ToAD-HF is a compact sound-card digital mode that sends short text bursts through the **SSB audio chain** of any HF transceiver.

* **No native binaries** – 100 % Python + NumPy/SciPy  
* **16 non-overlapping tones** (50 Hz spacing, 100 Hz-2 800 Hz)  
* **8 symbols/s** for live QSOs  
* Fuzzy FSK decoder with Hamming-tie handling and configurable redundancy  
* Works as **CLI _or_ importable library**

Please see [this page](mode_description.md) for more technical details on ToAD.

---

## 2 | Repository layout (library & tools)

| Path | Purpose |
|------|---------|
| `toad_encoder.py` | `encode_text_to_waveform(text, preamble_len, send_volume, …)` → `np.float32` |
| `toad_decoder.py` | `decode_file(path, redundancy, debug=False)` **or** streaming helpers |
| `toad_alphabet.py` | Auto-generated 16-tone code-book (`TEXT_TO_TOAD`, `TOAD_TO_TEXT`) |
| `toad_terminal.py` | Ready-made interactive console client (Tx + Rx thread) |
| `radio_common.py` | CAT / rigctl wrappers (`IC7300`, `K3S`, `FTDX10`, `MockRig` - Add yours here with a PR :)  |
| `config.py` | One-stop place for tone spacing, FFT, audio device names, redundancy … |
| `examples/` | Short, self-contained API demos (see below) |

---

## 3 | Install & CLI quick-start

### 3.1 Install
```bash
python -m venv radio_venv && source radio_venv/bin/activate
pip install -r requirements.txt
sudo apt install hamlib          # rigctl backend
sudo apt install python3-dev
python toad_terminal.py          # prompts for frequency, then [SEND] >
```
### 3.2 Hook up your rig

This will be different for evrey radio. But the basics are: Plug in your USB cable for CAT and sound port connection. 

- We default to `/dev/ttyUSB0` for CAT control.
- We default to `USB Audio Codec` for sound device. You can use `sd.query_devices()` to get the correct name for your setup.

A helper script `list_toad_devices` is provided to help with this task.

Both of these are configurable by editing config.py

### 3.3 Talk!

```bash
python toad_terminal.py # Prompts you for frequency in kHz
                        # Chooses USB/LSB based on frequency

[SEND] > CQ TOAD DE KN6UBF
```
## 4 | Protocol / config cheat sheet

| Item               | Value                                         |
| ------------------ | --------------------------------------------- |
| `NUM_TONES`        | **16**                                        |
| `FREQ_MIN`         | **100 Hz**                                    |
| `FREQ_STEP`        | **50 Hz**                                     |
| Symbol duration    | **125 ms** (8 sym/s)                          |
| Data symbol        | **two “1”s** (non-adjacent bins)              |
| Preamble / Trailer | `^` = all 16 bins “1” (default 8 frames each) |

```python
wave = encode_text_to_waveform("HELLO") # assume redundancy=2 and preamble_len=8 from config file
sd.play(wave)

# Emits:
# ^^^^^^^^  HH EE LL LL OO  ^^^^^^^^
# Over the air!
```
## 5 | Common troubleshooting

| Symptom                  | Hint                                                                 |
| ------------------------ | -------------------------------------------------------------------- |
| **PortAudioError -9999** | USB codec disappeared – re-plug, restart script                      |
| All frames print `1111…` | Audio clipping → reduce rig gain / computer volume                   |
| Many `[A\|B]` ties       | Keep redundancy ≥ 2 **and ensure Rx & Tx are both 48 kHz**           |
| No decode                | Check `FREQ_MIN + FREQ_STEP × NUM_TONES < 3 kHz` (inside SSB filter) |

## 6 | Extending

- Different spacing/rates → edit constants in config.py, regenerate toad_alphabet.py
- New radios → subclass radio_common.BaseRig, override _rigctl map
- Waterfall GUI → feed toad_decoder.\_stft_mag() into matplotlib/pyqtgraph

## 7 | License & Credits

Built with Hamlib, NumPy, SciPy, sounddevice, prompt_toolkit. Inspired in part by GGWave. 

Thank you to W6EFI, AD1M, and AJ6X for assistance, suggestions, testing, etc.

73 and croak on! KN6UBF
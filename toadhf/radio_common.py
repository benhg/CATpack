# radio_common.py

import subprocess
import threading

class IC7300:
    def __init__(self, device="/dev/ttyUSB0", audio_in="USB Audio CODEC", audio_out="USB Audio CODEC", model=3073, baud=115200):
        self.device = device
        self.model = model
        self.baud = baud
        self.tx_lock = threading.Lock()
        self.audio_in = audio_in
        self.audio_out = audio_out

    def _rigctl(self, *args):
        cmd = [
            "rigctl",
            "-m", str(self.model),
            "-r", self.device,
            "-s", str(self.baud),
        ]
        cmd.extend(*args)
        subprocess.run(cmd, check=True)

    def enable_sidetone(self, level=1.0):
        """Enable monitor audio for sidetone generation during PTT."""
        self._rigctl(["l", "MONITOR_GAIN", str(level)])

    def set_freq(self, freq_hz):
        """Set operating frequency."""
        self._rigctl(["F", str(freq_hz)])

    def set_mode(self, mode):
        """Set mode and passband."""
        mode_map = {
            # Sideband
            "USB": ("USB", "2400"),
            "LSB": ("LSB", "2400"),
            # Data
            "DATA-U": ("USB-D", "3000"),
            "DATA": ("USB-D", "3000"),
            "DATA-L": ("LSB-D", "3000"),
            # CW 
            "CW": ("CW", "500")
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")
        cmd = ["M"] + [m for m in mode_map[mode]]
        self._rigctl(cmd)

    def ptt_on(self):
        self._rigctl(["T", "1"])

    def ptt_off(self):
        self._rigctl(["T", "0"])

    def close(self):
        pass  # rigctl does not require closing

    def transmit_audio_block(self, audio_callback):
        """Key PTT, call audio_callback() to play audio, then unkey."""
        with self.tx_lock:
            self.ptt_on()
            audio_callback()
            self.ptt_off()

    def __repr__(self):
        return f"<IC 7300 port={self.device} audio_in={self.audio_in} audio_out={self.audio_out}>"

class MockIC7300:
    def __init__(self, device="/dev/cu.SLAB_USBtoUART",audio_in="USB Audio CODEC", audio_out="USB Audio CODEC", model=3073, baud=115200):
        self.device = device
        self.model = model
        self.baud = baud
        self.tx_lock = threading.Lock()
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.commands = []  # Store issued commands for inspection

    def _rigctl(self, *args):
        cmd = [
            "rigctl",
            "-m", str(self.model),
            "-r", self.device,
            "-s", str(self.baud),
            *args
        ]
        print(f"[MOCK] Would run: {' '.join(cmd)}")
        self.commands.append(cmd)

    def enable_sidetone(self, level=1.0):
        self._rigctl("l", "MONITOR_GAIN", str(level))

    def set_freq(self, freq_hz):
        self._rigctl("F", str(freq_hz))

    def set_mode(self, mode):
        mode_map = {
            "USB": ("USB", "2400"),
            "CW": ("CW", "500")
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")
        mode_str, passband = mode_map[mode]
        self.set_freq(14070000)
        self._rigctl("M", mode_str, passband)

    def ptt_on(self):
        self._rigctl("T", "1")

    def ptt_off(self):
        self._rigctl("T", "0")

    def close(self):
        print("[MOCK] Closing mock rig interface.")

    def transmit_audio_block(self, audio_callback):
        with self.tx_lock:
            self.ptt_on()
            print("[MOCK] Playing audio...")
            audio_callback()
            self.ptt_off()


class FTDX10:
    def __init__(self, device="/dev/ttyUSB0", audio_in="USB Audio CODEC", audio_out="USB Audio CODEC", model=1042, baud=38400):
        self.device = device
        self.model = model
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.baud = baud
        self.tx_lock = threading.Lock()

    def _rigctl(self, *args):
        cmd = [
            "rigctl",
            "-m", str(self.model),
            "-r", self.device,
            "-s", str(self.baud),
            *args
        ]
        subprocess.run(cmd, check=True)

    def enable_sidetone(self, level=1.0):
        """Enable sidetone or monitor audio if available (Yaesu may not support this directly)."""
        try:
            self._rigctl("l", "MONITOR_GAIN", str(level))
        except subprocess.CalledProcessError:
            print("Monitor gain setting not supported on FTDX10.")

    def set_freq(self, freq_hz):
        """Set operating frequency."""
        self._rigctl("F", str(freq_hz))

    def set_mode(self, mode):
        """Set mode and bandwidth if supported."""
        mode_map = {
            "USB": ("USB", "2400"),
            "LSB": ("LSB", "2400"),
            "CW": ("CW", "500"),
            "CW-R": ("CW-R", "500"),
            "RTTY": ("RTTY", "300"),
            "DATA": ("DATA", "2400"),
            "DATA-U": ("DATA", "2400"),
            "DATA-L": ("DATA", "2400"),
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")

        self._rigctl("M", mode_map[mode][0])
        # Elecraft K3 supports custom filter widths, but setting bandwidth may not work with rigctl

    def ptt_on(self):
        self._rigctl("T", "1")

    def ptt_off(self):
        self._rigctl("T", "0")

    def close(self):
        pass  # Nothing to close for rigctl

    def transmit_audio_block(self, audio_callback):
        """Key PTT, call audio_callback() to play audio, then unkey."""
        with self.tx_lock:
            self.ptt_on()
            audio_callback()
            self.ptt_off()

    def __repr__(self):
        return f"<FTDX10 port={self.device} audio_in={self.audio_in} audio_out={self.audio_out}>"


class K3S:
    def __init__(self, device="/dev/ttyUSB0", audio_in="USB Audio CODEC", audio_out="USB Audio CODEC",  model=2043, baud=38400):
        self.device = device
        self.model = model  # 229 is the Hamlib model for Elecraft K3/K3S
        self.baud = baud
        self.tx_lock = threading.Lock()
        self.audio_in = audio_in
        self.audio_out = audio_out

    def _rigctl(self, *args):
        cmd = [
            "rigctl",
            "-m", str(self.model),
            "-r", self.device,
            "-s", str(self.baud),
            *args
        ]
        subprocess.run(cmd, check=True)

    def enable_sidetone(self, level=1.0):
        """Enable sidetone or monitor audio if available (Elecraft may not support this directly)."""
        try:
            self._rigctl("l", "MONITOR_GAIN", str(level))
        except subprocess.CalledProcessError:
            print("Monitor gain setting not supported on K3S.")

    def set_freq(self, freq_hz):
        """Set operating frequency."""
        self._rigctl("F", str(freq_hz))

    def set_mode(self, mode):
        """Set mode and bandwidth if supported."""
        mode_map = {
            "USB": ("USB", "2400"),
            "LSB": ("LSB", "2400"),
            "CW": ("CW", "500"),
            "CW-R": ("CW-R", "500"),
            "RTTY": ("RTTY", "300"),
            "DATA": ("DATA", "2400"),
            "DATA-U": ("DATA", "2400"),
            "DATA-L": ("DATA", "2400"),
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")

        self._rigctl("M", mode_map[mode][0])
        # Elecraft K3 supports custom filter widths, but setting bandwidth may not work with rigctl

    def ptt_on(self):
        self._rigctl("T", "1")

    def ptt_off(self):
        self._rigctl("T", "0")

    def close(self):
        pass  # Nothing to close for rigctl

    def transmit_audio_block(self, audio_callback):
        """Key PTT, call audio_callback() to play audio, then unkey."""
        with self.tx_lock:
            self.ptt_on()
            audio_callback()
            self.ptt_off()

    def __repr__(self):
        return f"<Elecraft K3s port={self.device} audio_in={self.audio_in} audio_out={self.audio_out}>"


class truSDX:
    # This assumes that rigctld is running on port 7006:
    #
    #    rigctld -m 2055 -T 127.0.0.1 -t 7006 -s 115200 -r $PORT &
    #
    # where $PORT is the /dev/tty.usbserial-nnn that the (tr)uSDX was assigned.
    # This also assumes that the Audio and Mic/Key jacks on the (tr)uSDX are
    # connected to the headphone jack of a Mac. The baud parameter below is not
    # used.
    def __init__(self, device="127.0.0.1:7006", audio_in="External Microphone",
                 audio_out="External Headphones", model=2, baud=115200):
        self.device = device
        self.model = model
        self.baud = baud
        self.tx_lock = threading.Lock()
        self.audio_in = audio_in
        self.audio_out = audio_out

    def _rigctl(self, *args):
        cmd = [
            "rigctl",
            "-m", str(self.model),
            "-r", self.device
        ]
        cmd.extend(*args)
        subprocess.run(cmd, check=True)

    def enable_sidetone(self, level=1.0):
        """Enable monitor audio for sidetone generation during PTT."""
        # Can't use the (tr)uSDX speaker to monitor the input audio.
        pass

    def set_freq(self, freq_hz):
        """Set operating frequency."""
        self._rigctl(["F", str(freq_hz)])

    def set_mode(self, mode):
        """Set mode and passband."""
        mode_map = {
            # Sideband
            "USB": ("USB", "2400"),
            "LSB": ("LSB", "2400"),
            # Data
            "DATA-U": ("USB", "3000"),
            "DATA": ("USB", "3000"),
            "DATA-L": ("LSB", "3000"),
            # CW
            "CW": ("CW", "500")
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")
        cmd = ["M"] + [m for m in mode_map[mode]]
        self._rigctl(cmd)

    def ptt_on(self):
        self._rigctl(["T", "1"])

    def ptt_off(self):
        self._rigctl(["T", "0"])

    def close(self):
        pass  # rigctl does not require closing

    def transmit_audio_block(self, audio_callback):
        """Key PTT, call audio_callback() to play audio, then unkey."""
        with self.tx_lock:
            self.ptt_on()
            audio_callback()
            self.ptt_off()

    def __repr__(self):
        return (f"<(tr)uSDX port={self.device} audio_in={self.audio_in}"
                f" audio_out={self.audio_out}>")


class IC705:
    # This assumes that SDR-Control.app is connected to the IC-705, and has a
    # CAT Server running with RigCtrl / Hamlib on port 5001. It also assumes
    # that there are audio devices Radio to External and External to Radio,
    # created by, e.g. Loopback or BlackHole, and that they are configured and
    # enabled in SDR-Control as the audio output and input devices. The baud
    # setting is not used.
    def __init__(self, device="127.0.0.1:5001", audio_in="Radio to External",
                 audio_out="External to Radio", model=2, baud=115200):
        self.device = device
        self.model = model
        self.baud = baud
        self.tx_lock = threading.Lock()
        self.audio_in = audio_in
        self.audio_out = audio_out

    def _rigctl(self, *args):
        # Need to repeat mode set to change bandwidth. Go figure.
        for _ in range(2 if len(args) > 0 and args[0] == "M" else 1):
            cmd = [
                "rigctl",
                "-m", str(self.model),
                "-r", self.device
            ]
            cmd.extend(*args)
            subprocess.run(cmd, check=True)

    def enable_sidetone(self, level=1.0):
        """Enable monitor audio for sidetone generation during PTT."""
        # MONI setting is not implemented in the SDR-Control CAT server.
        # self._rigctl(["L", "MONITOR_GAIN", str(level)])
        pass

    def set_freq(self, freq_hz):
        """Set operating frequency."""
        self._rigctl(["F", str(freq_hz)])

    def set_mode(self, mode):
        """Set mode and passband."""
        mode_map = {
            # Sideband
            "USB": ("USB", "2400"),
            "LSB": ("LSB", "2400"),
            # Data
            "DATA-U": ("USB-D", "3000"),
            "DATA": ("USB-D", "3000"),
            "DATA-L": ("LSB-D", "3000"),
            # CW
            "CW": ("CW", "500")
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")
        cmd = ["M"] + [m for m in mode_map[mode]]
        self._rigctl(cmd)

    def ptt_on(self):
        self._rigctl(["T", "1"])

    def ptt_off(self):
        self._rigctl(["T", "0"])

    def close(self):
        pass  # rigctl does not require closing

    def transmit_audio_block(self, audio_callback):
        """Key PTT, call audio_callback() to play audio, then unkey."""
        with self.tx_lock:
            self.ptt_on()
            audio_callback()
            self.ptt_off()

    def __repr__(self):
        return f"<IC-705 port={self.device} audio_in={self.audio_in} audio_out={self.audio_out}>"

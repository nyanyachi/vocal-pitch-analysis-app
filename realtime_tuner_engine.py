import math
from collections import deque, Counter

import librosa
import numpy as np


NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


class RealtimeTunerEngine:
    def __init__(
        self,
        sample_rate=44100,
        min_frequency=70,
        max_frequency=1000,
        volume_threshold=0.01,
        smoothing_frames=5,
        note_history_frames=10,
        cent_history_frames=20,
        perfect_cents=10,
        good_cents=30,
    ):
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.volume_threshold = volume_threshold

        self.pitch_history = deque(maxlen=smoothing_frames)
        self.note_history = deque(maxlen=note_history_frames)
        self.cent_history = deque(maxlen=cent_history_frames)

        self.perfect_cents = perfect_cents
        self.good_cents = good_cents

    def frequency_to_note(self, frequency):
        if frequency is None or frequency <= 0:
            return None, None, None

        midi_float = 69 + 12 * math.log2(frequency / 440.0)
        midi_number = round(midi_float)

        note_index = midi_number % 12
        octave = (midi_number // 12) - 1

        note_name = f"{NOTE_NAMES[note_index]}{octave}"
        target_frequency = 440.0 * (2 ** ((midi_number - 69) / 12))
        cent_diff = 1200 * math.log2(frequency / target_frequency)

        return note_name, target_frequency, cent_diff

    def get_tuning_status(self, cent_diff):
        if cent_diff is None:
            return "No Pitch"

        abs_cent = abs(cent_diff)

        if abs_cent <= self.perfect_cents:
            return "Perfect"
        elif abs_cent <= self.good_cents:
            return "Good"
        elif cent_diff > self.good_cents:
            return "High"
        else:
            return "Low"

    def make_cent_bar(self, cent_diff, width=21):
        if cent_diff is None:
            return "[----------|----------]"

        cent_diff = max(-50, min(50, cent_diff))

        center = width // 2
        position = round(center + (cent_diff / 50) * center)
        position = max(0, min(width - 1, position))

        bar = ["-"] * width
        bar[center] = "|"
        bar[position] = "●"

        return "[" + "".join(bar) + "]"

    def get_stable_note(self, note_name):
        if note_name is None:
            return None

        self.note_history.append(note_name)

        note_counts = Counter(self.note_history)
        stable_note, count = note_counts.most_common(1)[0]

        return stable_note

    def calculate_stability(self):
        if len(self.cent_history) < 5:
            return None

        cent_std = float(np.std(self.cent_history))

        stability = 100 - (cent_std * 2)
        stability = max(0, min(100, stability))

        return stability

    def detect_pitch_pyin(self, audio):
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                sr=self.sample_rate,
                frame_length=2048,
                hop_length=256,
            )

            valid_f0 = f0[~np.isnan(f0)]

            if len(valid_f0) == 0:
                return None

            pitch = float(np.median(valid_f0))

            if pitch < self.min_frequency or pitch > self.max_frequency:
                return None

            return pitch

        except Exception:
            return None

    def detect_pitch_fft(self, audio):
        audio = audio.astype(np.float32)

        if len(audio) < 512:
            return None

        audio = audio - np.mean(audio)

        window = np.hanning(len(audio))
        windowed_audio = audio * window

        spectrum = np.fft.rfft(windowed_audio)
        magnitude = np.abs(spectrum)

        frequencies = np.fft.rfftfreq(len(audio), d=1.0 / self.sample_rate)

        min_index = np.searchsorted(frequencies, self.min_frequency)
        max_index = np.searchsorted(frequencies, self.max_frequency)

        if max_index <= min_index:
            return None

        search_magnitude = magnitude[min_index:max_index]

        if len(search_magnitude) == 0:
            return None

        peak_index = int(np.argmax(search_magnitude)) + min_index
        peak_frequency = float(frequencies[peak_index])

        if peak_frequency < self.min_frequency or peak_frequency > self.max_frequency:
            return None

        peak_strength = magnitude[peak_index]
        average_strength = np.mean(search_magnitude) + 1e-8

        if peak_strength < average_strength * 5:
            return None

        return peak_frequency

    def detect_pitch(self, audio):
        audio = audio.astype(np.float32)

        volume = np.max(np.abs(audio))

        if volume < self.volume_threshold:
            return None

        pitch = self.detect_pitch_pyin(audio)

        if pitch is not None:
            return pitch

        pitch = self.detect_pitch_fft(audio)

        if pitch is not None:
            return pitch

        return None

    def analyze_audio(self, audio):
        pitch = self.detect_pitch(audio)

        if pitch is None:
            return {
                "has_pitch": False,
                "pitch": None,
                "note": None,
                "stable_note": None,
                "cent_diff": None,
                "status": "No Pitch",
                "cent_bar": self.make_cent_bar(None),
                "stability": self.calculate_stability(),
            }

        self.pitch_history.append(pitch)
        smooth_pitch = float(np.median(self.pitch_history))

        note_name, target_frequency, cent_diff = self.frequency_to_note(
            smooth_pitch
        )

        stable_note = self.get_stable_note(note_name)
        status = self.get_tuning_status(cent_diff)
        cent_bar = self.make_cent_bar(cent_diff)

        self.cent_history.append(cent_diff)
        stability = self.calculate_stability()

        return {
            "has_pitch": True,
            "pitch": smooth_pitch,
            "note": note_name,
            "stable_note": stable_note,
            "target_frequency": target_frequency,
            "cent_diff": cent_diff,
            "status": status,
            "cent_bar": cent_bar,
            "stability": stability,
        }
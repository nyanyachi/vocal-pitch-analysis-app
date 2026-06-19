import tkinter as tk
from tkinter import ttk
import threading
import queue

import json
import os

import numpy as np
import sounddevice as sd

from realtime_tuner_engine import RealtimeTunerEngine

audio_queue = queue.Queue()
engine = RealtimeTunerEngine()

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048

WINDOW_WIDTH = 420
WINDOW_HEIGHT = 320

CONFIG_FILE = "overlay_config.json"
DEFAULT_X = 100
DEFAULT_Y = 100
WINDOW_ALPHA = 0.88

audio_thread = None
stop_audio_event = threading.Event()
selected_device_id = None
device_map = {}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config():
    try:
        config = {
            "x": root.winfo_x(),
            "y": root.winfo_y(),
            "device_id": selected_device_id
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    except Exception:
        pass


def close_app(event=None):
    save_config()
    stop_audio_event.set()
    root.destroy()


def make_cent_bar(cent, width=21):
    if cent is None:
        return "[----------|----------]"

    cent = max(-50, min(50, cent))
    center = width // 2
    pos = int((cent + 50) / 100 * (width - 1))

    chars = ["-"] * width
    chars[center] = "|"
    chars[pos] = "●"

    return "[" + "".join(chars) + "]"


def get_input_devices():
    devices = sd.query_devices()
    input_devices = []

    hidden_keywords = [
        "Microsoft Sound Mapper",
        "Primary Sound Capture Driver",
        "주 사운드 캡처 드라이버"
    ]

    for index, device in enumerate(devices):
        name = device["name"]

        if device["max_input_channels"] <= 0:
            continue

        if any(keyword in name for keyword in hidden_keywords):
            continue

        label = f"{index}: {name}"
        input_devices.append((label, index))

    return input_devices


def audio_callback(indata, frames, time_info, status):
    audio = indata[:, 0].copy()
    volume = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    audio_queue.put((audio, volume, peak))


def prepare_audio_for_analysis(audio):
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.3

    return audio.astype(np.float32)


def update_overlay():
    audio = None
    volume = 0.0
    peak = 0.0

    while not audio_queue.empty():
        audio, volume, peak = audio_queue.get()

    if audio is not None:
        volume_label.config(text=f"Input: {volume:.4f} / Peak: {peak:.4f}")

        analysis_audio = prepare_audio_for_analysis(audio)

        try:
            result = engine.analyze_audio(analysis_audio)
        except Exception as e:
            note_label.config(text="ERR")
            cent_label.config(text="Cent: --")
            bar_label.config(text=make_cent_bar(None))
            status_label.config(text="Engine Error")
            stability_label.config(text=str(e)[:40])
            root.after(30, update_overlay)
            return

        if result["has_pitch"]:
            cent = result["cent_diff"]

            note_label.config(text=result["stable_note"])
            cent_label.config(text=f"Cent: {cent:+.1f}")
            bar_label.config(text=make_cent_bar(cent))
            status_label.config(text=result["status"])

            stability = result.get("stability")

            if stability is not None:
                stability_label.config(text=f"Stability: {stability:.1f}%")
            else:
                stability_label.config(text="Stability: --%")
        else:
            note_label.config(text="--")
            cent_label.config(text="Cent: --")
            bar_label.config(text=make_cent_bar(None))
            status_label.config(text="No Pitch")
            stability_label.config(text="Stability: --%")

    root.after(30, update_overlay)


def start_audio():
    global selected_device_id

    try:
        with sd.InputStream(
            device=selected_device_id,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        ):
            status_message.config(text="Mic: Running")

            while not stop_audio_event.is_set():
                sd.sleep(100)

    except Exception as e:
        status_message.config(text=f"Mic Error: {e}")


def restart_audio():
    global audio_thread

    stop_audio_event.set()

    if audio_thread is not None and audio_thread.is_alive():
        audio_thread.join(timeout=1)

    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    stop_audio_event.clear()

    audio_thread = threading.Thread(
        target=start_audio,
        daemon=True
    )
    audio_thread.start()


def on_device_change(event=None):
    global selected_device_id

    selected_label = device_var.get()

    if selected_label in device_map:
        selected_device_id = device_map[selected_label]
        save_config()
        restart_audio()


config = load_config()
saved_x = config.get("x", DEFAULT_X)
saved_y = config.get("y", DEFAULT_Y)
saved_device_id = config.get("device_id")

root = tk.Tk()
root.title("Vocal Tuner Overlay")

root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{saved_x}+{saved_y}")
root.attributes("-topmost", True)
root.attributes("-alpha", WINDOW_ALPHA)
root.configure(bg="black")

root.bind("<Escape>", close_app)
root.protocol("WM_DELETE_WINDOW", close_app)

device_frame = tk.Frame(root, bg="black")
device_frame.pack(pady=(8, 0))

device_var = tk.StringVar()

input_devices = get_input_devices()

for label, device_id in input_devices:
    device_map[label] = device_id

device_labels = list(device_map.keys())

device_combo = ttk.Combobox(
    device_frame,
    textvariable=device_var,
    values=device_labels,
    state="readonly",
    width=42
)
device_combo.pack()

if device_labels:
    default_label = device_labels[0]

    for label, device_id in device_map.items():
        if device_id == saved_device_id:
            default_label = label
            break

    device_var.set(default_label)
    selected_device_id = device_map[default_label]

device_combo.bind("<<ComboboxSelected>>", on_device_change)

note_label = tk.Label(
    root,
    text="A3",
    font=("Arial", 50, "bold"),
    fg="white",
    bg="black"
)
note_label.pack(pady=(8, 0))

cent_label = tk.Label(
    root,
    text="Cent: +0.0",
    font=("Arial", 18),
    fg="white",
    bg="black"
)
cent_label.pack()

bar_label = tk.Label(
    root,
    text="[----------|----------]",
    font=("Consolas", 18),
    fg="white",
    bg="black"
)
bar_label.pack(pady=(2, 0))

status_label = tk.Label(
    root,
    text="Perfect",
    font=("Arial", 16),
    fg="white",
    bg="black"
)
status_label.pack()

stability_label = tk.Label(
    root,
    text="Stability: --%",
    font=("Arial", 13),
    fg="gray",
    bg="black"
)
stability_label.pack(pady=(2, 0))

status_message = tk.Label(
    root,
    text="Mic: Ready",
    font=("Arial", 9),
    fg="gray",
    bg="black"
)
status_message.pack(pady=(2, 0))

volume_label = tk.Label(
    root,
    text="Input: -- / Peak: --",
    font=("Arial", 9),
    fg="gray",
    bg="black"
)
volume_label.pack(pady=(2, 0))

hint_label = tk.Label(
    root,
    text="ESC to close",
    font=("Arial", 9),
    fg="gray",
    bg="black"
)
hint_label.pack(pady=(2, 0))

restart_audio()
update_overlay()

root.mainloop()
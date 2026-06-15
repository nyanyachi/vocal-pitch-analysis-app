import time
import queue

import numpy as np
import sounddevice as sd

from realtime_tuner_engine import RealtimeTunerEngine


SAMPLE_RATE = 44100
BLOCK_SIZE = 4096
DEFAULT_VOLUME_THRESHOLD = 0.01

HOLD_SECONDS = 0.5

audio_queue = queue.Queue()

last_result = None
last_pitch_time = 0


def list_input_devices():
    print("\nAvailable input devices")
    print("-" * 60)

    devices = sd.query_devices()
    input_devices = []

    for index, device in enumerate(devices):
        max_input_channels = device["max_input_channels"]

        if max_input_channels > 0:
            input_devices.append(index)
            print(
                f"{index}: {device['name']} "
                f"/ input channels: {max_input_channels}"
            )

    print("-" * 60)

    default_input, _ = sd.default.device
    print(f"Default input device index: {default_input}")

    return input_devices


def choose_input_device():
    input_devices = list_input_devices()

    user_input = input(
        "\nEnter input device index. "
        "Press Enter to use default input device: "
    ).strip()

    default_input, _ = sd.default.device

    if user_input == "":
        print(f"Using default input device: {default_input}")
        return default_input

    try:
        device_index = int(user_input)

        if device_index not in input_devices:
            print("Invalid input device index. Using default input device.")
            return default_input

        print(f"Using selected input device: {device_index}")
        return device_index

    except ValueError:
        print("Invalid input. Using default input device.")
        return default_input


def calibrate_noise_floor(device_index, duration=2.0):
    print("\n마이크 노이즈 자동 보정을 시작합니다.")
    print("약 2초 동안 조용히 있어주세요...")
    print()

    try:
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_index
        )

        sd.wait()

        audio = recording[:, 0]
        volume_values = np.abs(audio)

        noise_level = float(np.median(volume_values))

        auto_threshold = noise_level * 3.0
        auto_threshold = max(auto_threshold, 0.003)
        auto_threshold = min(auto_threshold, 0.03)

        print(f"Noise level: {noise_level:.6f}")
        print(f"Auto volume threshold: {auto_threshold:.6f}")

        return auto_threshold

    except Exception as e:
        print("Noise calibration failed.")
        print(e)
        print(f"Using default threshold: {DEFAULT_VOLUME_THRESHOLD}")

        return DEFAULT_VOLUME_THRESHOLD


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio = indata[:, 0].copy()
    audio_queue.put(audio)


def format_stability(stability):
    if stability is None:
        return "Stability: --"

    return f"Stability: {stability:5.1f}%"


def print_result(result):
    stability_text = format_stability(result["stability"])

    print(
        f"\rHz: {result['pitch']:7.2f} | "
        f"Note: {result['stable_note']:4} | "
        f"Cent: {result['cent_diff']:+7.2f} | "
        f"{result['cent_bar']} | "
        f"Status: {result['status']:8} | "
        f"{stability_text}",
        end=""
    )


def main():
    global last_result, last_pitch_time

    print("🎤 Real-time Pitch Tuner Test - engine version")
    print("This test uses your local microphone input.")
    print("Press Ctrl + C to stop.")

    device_index = choose_input_device()
    volume_threshold = calibrate_noise_floor(device_index)

    engine = RealtimeTunerEngine(
        sample_rate=SAMPLE_RATE,
        volume_threshold=volume_threshold,
        smoothing_frames=5,
        note_history_frames=10,
        cent_history_frames=20,
        perfect_cents=10,
        good_cents=30,
    )

    print("\nStarting tuner...")
    print("-" * 60)

    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        ):
            while True:
                if not audio_queue.empty():
                    audio = audio_queue.get()
                    result = engine.analyze_audio(audio)
                    current_time = time.time()

                    if result["has_pitch"]:
                        last_result = result
                        last_pitch_time = current_time
                        print_result(result)

                    elif (
                        last_result is not None
                        and current_time - last_pitch_time <= HOLD_SECONDS
                    ):
                        print_result(last_result)

                    else:
                        print("\rNo stable pitch detected...".ljust(130), end="")

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")

    except Exception as e:
        print("\nError:", e)


if __name__ == "__main__":
    main()
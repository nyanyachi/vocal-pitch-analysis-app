# Vocal Pitch Analysis App

Vocal Pitch Analysis App is a Streamlit web application that helps singers compare their vocal recording with a reference vocal track. After analysis, users receive Pitch Accuracy and Pitch Stability scores, a Coaching Summary, a focused Today's Practice plan, pitch graphs, and a Vocal Profile based on saved results.

The Streamlit app is the main product. A Desktop Overlay Tuner and Command-Line Tuner are included as optional companion tools for local real-time pitch monitoring.

## Live Demo

🔗 [vocal-pitch-analysis-app.streamlit.app](https://vocal-pitch-analysis-app.streamlit.app/)

## Screenshots

(TODO)

## Demo GIF

(TODO)

## Overview

Upload a reference vocal WAV file and your own vocal WAV file to compare pitch, timing, accuracy, and stability. Results can be saved locally to review progress across multiple recordings.


## Features

### Vocal Analysis

* Compare a reference vocal WAV file with your own recording
* View Pitch Accuracy and Pitch Stability scores with clear performance labels
* Read a concise Coaching Summary based on the analysis
* Follow up to three focused actions in Today's Practice
* Find the weakest detected section with its exact time range
* Review pitch comparison, corrected pitch, and cent-error graphs
* Check estimated key difference, timing alignment, and highest and lowest notes

### Growth Tracking

* Save analysis results locally to `records.json`
* Review average Accuracy, Stability, and key difference
* Filter history by song and compare older and newer recordings
* Generate a Vocal Profile from saved performances

### Language Support

* Korean
* English
* Japanese

### Optional Companion Tools

The repository also includes:

* Desktop Overlay Tuner — an always-on-top Windows tuner with note, cent, stability, microphone selection, and position persistence
* Command-Line Tuner — a local microphone test tool with pitch, note, cent, and stability output

The main Vocal Pitch Analysis App runs as the Streamlit web app and is not distributed through GitHub Releases.

The downloadable **V3.5.1 Lite** release is only the optional Windows Desktop Overlay Tuner: [Download the Desktop Overlay Tuner from GitHub Releases](https://github.com/nyanyachi/vocal-pitch-analysis-app/releases)

## Tech Stack

- Python
- Streamlit
- Tkinter
- NumPy
- Pandas
- Matplotlib
- Librosa
- SoundDevice

## Quick Start

```bash
git clone https://github.com/nyanyachi/vocal-pitch-analysis-app.git
cd vocal-pitch-analysis-app
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install and run the main Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Run the Optional Companion Tools

```bash
# Command-Line Tuner
python real_time_pitch.py

# Desktop Overlay Tuner
python overlay_tuner.py
```

## Project Structure

```text
app.py
record_utils.py
real_time_pitch.py
realtime_tuner_engine.py
overlay_tuner.py
overlay_config.json
translations.py
records.json
requirements.txt
README.md
README_KR.md
README_JP.md
```

## Author

Nyanyachi

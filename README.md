## Vocal Pitch Analysis App

A Python-based web application that helps singers analyze vocal recordings, compare performances against reference tracks, and track long-term vocal improvement through data-driven feedback.

Built with Streamlit and Librosa, the application provides pitch analysis, accuracy measurement, stability tracking, and real-time tuning tools for vocal practice.

## Live Demo

🔗 [vocal-pitch-analysis-app.streamlit.app](https://vocal-pitch-analysis-app.streamlit.app/)


## Overview

Vocal Pitch Analysis App was created to provide singers with objective feedback on their vocal performance and long-term vocal improvement.

The application analyzes vocal recordings, compares them with reference tracks, measures pitch accuracy and stability, and stores historical results to help users monitor progress over time.

In addition to offline vocal analysis, the project includes a real-time tuner system and a lightweight desktop overlay tuner for live pitch monitoring during singing practice, recording sessions, and virtual performance environments.


## What This Project Demonstrates

- Python application development
- Audio signal processing and pitch analysis
- Real-time microphone input processing
- Interactive web application development with Streamlit
- Desktop application development with Tkinter
- Data visualization and user feedback systems
- JSON-based data storage and progress tracking
- Multilingual application design
- Software packaging and release distribution
- End-to-end project deployment and maintenance


## Features

### Vocal Analysis

* Upload reference vocal WAV file
* Upload personal vocal WAV file
* Automatic pitch extraction using librosa.pyin()
* Pitch comparison graph
* Automatic key difference estimation
* Internal key correction
* Automatic time alignment
* Cent-based pitch error calculation
* Accuracy score calculation
* Stability score calculation
* Segment-based accuracy analysis
* Highest and lowest pitch detection
* Pitch error visualization

### Growth Tracking

* Save analysis results to records.json
* Vocal history dashboard
* Average Accuracy calculation
* Average Stability calculation
* Average key difference calculation
* Song filtering
* Compare older and newer recordings
* Personal vocal profile generation

### Real-Time Tuner (V3.0)

* Real-time microphone pitch detection
* Frequency (Hz) display
* Musical note detection
* Cent difference display
* Real-time tuning status
 * Perfect
 * Good
 * High
 * Low
* Real-time cent bar visualization
* Pitch smoothing
* Note stabilization
* Pitch hold system
* Automatic microphone noise calibration
* Real-time stability score

### Desktop Overlay Tuner (V3.5)
* Always-on-top desktop overlay
* Real-time note display
* Real-time cent difference display
* Visual cent bar feedback
* Stability monitoring
* Microphone device selection
* Window position persistence
* Lightweight FFT-based pitch detection engine
* Standalone Windows executable release
* Optimized distribution size (121 MB → 21.8 MB)

Download:

[GitHub Releases](https://github.com/nyanyachi/vocal-pitch-analysis-app/releases/tag/v3.5.0)

## Tech Stack

- Python
- Streamlit
- Tkinter
- NumPy
- Pandas
- Matplotlib
- Librosa
- SoundDevice

## Installation

```bash
pip install -r requirements.txt
```

## Releases

### Desktop Overlay Tuner

Windows desktop builds are available through GitHub Releases.

Current release:

- V3.5.1 Lite
- Lightweight FFT-based tuner engine
- Reduced package size from 121 MB to 21.8 MB
- Standalone executable distribution


## Run

### Main Application

```bash
streamlit run app.py
```

### Real-Time Tuner Test

```bash
python real_time_pitch.py
```

### Desktop Overlay Tuner
```bash
python overlay_tuner.py
```

## Project Structure

```text
app.py
real_time_pitch.py
realtime_tuner_engine.py
translations.py
requirements.txt
README.md
README_KR.md
README_JP.md
```

## Roadmap

### Completed
- V3.0
- V3.5

### Next
#### V4.0 

* Practice session mode
* Vocal achievements system
* Advanced vocal profile
* Growth analytics dashboard

## Author

Nyanyachi

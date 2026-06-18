## Vocal Pitch Analysis App

A Python-based web application that helps singers analyze vocal recordings, compare performances against reference tracks, and track long-term vocal improvement through data-driven feedback.

Built with Streamlit and Librosa, the application provides pitch analysis, accuracy measurement, stability tracking, and real-time tuning tools for vocal practice.

## Live Demo

🔗 vocal-pitch-analysis-app.streamlit.app


## Overview

Vocal Pitch Analysis App was created to provide singers with objective feedback on their vocal performance.

The application analyzes vocal recordings, compares them with a reference track, measures pitch accuracy and stability, and stores historical results to help users monitor improvement over time.

In addition to offline analysis, the project includes a real-time tuner module for live pitch monitoring and vocal practice.

## What This Project Demonstrates

- Python application development
- Audio signal processing using Librosa
- Interactive web application development with Streamlit
- Data visualization and user feedback systems
- JSON-based data storage and history tracking
- Real-time microphone input processing
- Multilingual application support
- Deployment and project maintenance workflows


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

## Tech Stack

- Python
- Streamlit
- Librosa
- NumPy
- Pandas
- Matplotlib
- SoundDevice

## Installation

```bash
pip install -r requirements.txt
```

## Run

### Main Application

```bash
streamlit run app.py
```

### Real-Time Tuner Test

```bash
python real_time_pitch.py
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

### V3.0

* Real-time tuner MVP
* Real-time pitch detection
* Stability monitoring
* Automatic noise calibration

### V3.5

* Always-on-top overlay mode
* VR-compatible practice overlay support

### V4.0

* Practice session mode
* Vocal achievements system
* Advanced vocal profile
* Growth analytics dashboard

## Author

Nyanyachi

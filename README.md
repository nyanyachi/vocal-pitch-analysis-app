# Vocal Pitch Analysis App

A personal vocal coaching and vocal growth tracking application built with Python and Streamlit.

## Overview

This project is designed to help singers compare their vocal recordings with a reference vocal track and track their own improvement over time.

The goal is not to compete with other singers, but to compare past recordings with current recordings and monitor personal growth.

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

## Technologies

* Python
* Streamlit
* Librosa
* NumPy
* Pandas
* Matplotlib
* SoundDevice

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
* VRChat practice support

### V4.0

* Practice session mode
* Vocal achievements system
* Advanced vocal profile
* Growth analytics dashboard

## Author

Nyanyachi

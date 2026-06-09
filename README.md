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

### Growth Tracking (V2.5)

* Save analysis results to records.json
* Vocal history dashboard
* Average Accuracy calculation
* Average Stability calculation
* Average key difference calculation
* Song filtering
* Compare older and newer recordings
* Personal vocal profile generation

## Technologies

* Python
* Streamlit
* Librosa
* NumPy
* Pandas
* Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Future Plans

### V3

* Vocal profile improvements
* Reference quality scoring
* Progress charts
* Vocal range analysis
* AI-based coaching feedback
* Multi-language support

## Author

Nyanyachi

# Vocal Pitch Analysis App

Python + Streamlit 기반의 개인용 보컬 분석 및 성장 기록 애플리케이션입니다.

## 프로젝트 소개

Vocal Pitch Analysis App은 보컬 녹음을 분석하고, 기준 보컬과 비교하며, 장기적인 성장 과정을 기록할 수 있도록 만든 개인용 보컬 분석 애플리케이션입니다.

이 프로젝트의 목적은 다른 사람과 비교하거나 음치를 판정하는 것이 아닙니다.

과거의 나와 현재의 나를 비교하고, 연습 과정에서 얼마나 성장했는지 객관적인 데이터로 확인할 수 있도록 설계되었습니다.

분석 기능뿐만 아니라 실시간 튜너와 데스크톱 오버레이 기능을 제공하여 연습 중에도 음정을 확인할 수 있습니다.

---

## 주요 기능

### 보컬 분석

* 기준 보컬 WAV 파일 업로드
* 내 보컬 WAV 파일 업로드
* librosa.pyin() 기반 음정 추출
* 음정 비교 그래프
* 자동 키 차이 추정
* 내부 키 보정
* 자동 시간 정렬
* Cent 기반 음정 오차 계산
* Accuracy 점수 계산
* Stability 점수 계산
* 구간별 Accuracy 분석
* 최고음 / 최저음 구간 탐지
* 음정 오차 시각화

### 성장 기록

* records.json 저장
* 보컬 기록 대시보드
* 평균 Accuracy 계산
* 평균 Stability 계산
* 평균 키 차이 계산
* 곡별 필터
* 과거 녹음과 현재 녹음 비교
* 개인 보컬 프로파일 생성

### 실시간 튜너 (V3.0)

* 실시간 마이크 음정 감지

* 현재 주파수(Hz) 표시

* 음 이름 표시 (C4, A3 등)

* Cent 차이 표시

* 실시간 튜닝 상태 표시

  * Perfect
  * Good
  * High
  * Low

* 실시간 Cent Bar

* Pitch Smoothing

* Note Stabilization

* Pitch Hold 시스템

* 자동 마이크 노이즈 보정

* 실시간 Stability 계산

### 데스크톱 오버레이 튜너 (V3.5)

* Always-On-Top 오버레이 창
* 실시간 음 이름 표시
* 실시간 Cent 표시
* Cent Bar 시각화
* Stability 모니터링
* 마이크 장치 선택
* 창 위치 자동 저장
* 투명도 적용
* Windows 실행 파일(EXE) 배포 지원

### Lite 버전 (V3.5.1)

* FFT 기반 경량 음정 감지 엔진

* Overlay 전용 경량 빌드

* 패키지 크기 최적화

  * 121 MB → 21.8 MB

* 빠른 실행 속도

* 휴대성 향상

---

## 사용 기술

* Python
* Streamlit
* Tkinter
* Librosa
* NumPy
* Pandas
* Matplotlib
* SoundDevice

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 실행

### 메인 앱

```bash
streamlit run app.py
```

### 실시간 튜너

```bash
python real_time_pitch.py
```

### 데스크톱 오버레이 튜너

```bash
python overlay_tuner.py
```

---

## 프로젝트 구조

```text
app.py
overlay_tuner.py
real_time_pitch.py
realtime_tuner_engine.py
translations.py
record_utils.py
records.json
requirements.txt
README.md
README_KR.md
README_JP.md
```

---

## 릴리즈

GitHub Releases를 통해 Windows 실행 파일(EXE)을 다운로드할 수 있습니다.

현재 제공 버전:

* Vocal Pitch Analysis App V3.5 Overlay Tuner
* Vocal Pitch Analysis App V3.5.1 Lite (권장)

Lite 버전은 FFT 기반 경량 엔진을 사용하여 패키지 크기를 121 MB에서 21.8 MB로 줄였습니다.

---

## 로드맵

### V3.0

* 실시간 튜너 MVP
* 실시간 음정 감지
* Stability 모니터링
* 자동 노이즈 보정

### V3.5

* Always-On-Top 오버레이
* 데스크톱 튜너 UI
* 마이크 장치 선택
* EXE 배포

### V3.6

* OVR Toolkit 연동 테스트
* XSOverlay 표시 테스트
* VR 환경 연습 지원

### V4.0

* Practice Mode
* Free Practice
* Long Tone Practice
* Section Practice
* One-Take Practice
* 연습 기록 저장
* 성장 대시보드 강화

---

## 프로젝트 철학

이 프로젝트는 음치 판정기가 아닙니다.

다른 사람과 비교하기 위한 도구도 아닙니다.

과거의 나와 현재의 나를 비교하고, 연습 기록을 쌓아가며 성장 과정을 확인하는 것이 목표입니다.

보컬 연습을 더 즐겁고 지속 가능하게 만들기 위한 개인 성장 도구를 지향합니다.

---

## 제작자

Nyanyachi

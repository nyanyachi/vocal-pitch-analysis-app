import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import json
from datetime import datetime
import os
import pandas as pd

st.title("보컬 음정 분석 보조 앱 - v2.5")
tab_analyze, tab_records = st.tabs(["🎙️ 분석하기", "📊 성장 기록"])
with tab_analyze:
    st.write("기준 보컬과 내 보컬의 pitch 그래프를 비교합니다.")
    st.write(
        "WAV 파일만 지원합니다. "
        "정확한 pitch 분석을 위해 보컬만 포함된 WAV 파일을 사용해주세요."
    )
    st.info(
        "권장사항\n"
        "- WAV 파일 사용\n"
        "- 보컬만 export한 파일 사용\n"
        "- 원곡은 보컬 분리 후 사용\n"
        "- MR이 섞이면 분석 정확도가 떨어질 수 있음\n"
        "- 아주 짧게 튀는 삐용 소리는 자동으로 일부 제거합니다"
    )

    song_name = st.text_input("곡명 입력", value="Unknown Song")

    reference_file = st.file_uploader("기준 보컬 WAV 파일 업로드", type=["wav"])
    my_file = st.file_uploader("내 보컬 WAV 파일 업로드", type=["wav"])

    if "comparison_record" not in st.session_state:
        st.session_state.comparison_record = None


    def remove_short_pitch_spikes(valid_times, valid_pitch, min_duration=0.2, jump_ratio=1.8):
        if len(valid_pitch) < 3:
            return valid_times, valid_pitch

        cleaned_times = []
        cleaned_pitch = []

        for i in range(len(valid_pitch)):
            current_pitch = valid_pitch[i]
            prev_pitch = valid_pitch[i - 1] if i > 0 else current_pitch
            next_pitch = valid_pitch[i + 1] if i < len(valid_pitch) - 1 else current_pitch

            nearby_pitch = np.median([prev_pitch, next_pitch])
            is_spike = current_pitch > nearby_pitch * jump_ratio

            if is_spike:
                prev_time = valid_times[i - 1] if i > 0 else valid_times[i]
                next_time = valid_times[i + 1] if i < len(valid_times) - 1 else valid_times[i]
                duration = next_time - prev_time

                if duration < min_duration:
                    continue

            cleaned_times.append(valid_times[i])
            cleaned_pitch.append(current_pitch)

        return np.array(cleaned_times), np.array(cleaned_pitch)


    def calculate_semitone_diff(my_pitch, ref_pitch):
        if my_pitch <= 0 or ref_pitch <= 0:
            return 0
        return 12 * np.log2(my_pitch / ref_pitch)




    def calculate_accuracy(ref_times, ref_pitch, my_times, corrected_my_pitch):
        """
        기준 보컬 pitch와 키 보정된 내 보컬 pitch를 비교합니다.
        시간축 자동 보정 후 cent 오차, Accuracy, 구간별 Accuracy를 계산합니다.
        """

        best_result = None
        shift_candidates = np.arange(-1.0, 1.01, 0.1)

        for time_shift in shift_candidates:
            shifted_my_times = my_times + time_shift

            interpolated_my_pitch = np.interp(
                ref_times,
                shifted_my_times,
                corrected_my_pitch
            )

            valid_mask = (
                (ref_pitch > 0)
                & (interpolated_my_pitch > 0)
            )

            ref_valid = ref_pitch[valid_mask]
            my_valid = interpolated_my_pitch[valid_mask]
            time_valid = ref_times[valid_mask]

            if len(ref_valid) == 0:
                continue

            cent_diff = 1200 * np.log2(my_valid / ref_valid)

            # 너무 큰 오차는 시간축 차이/오검출 가능성이 높으므로 제외
            cent_valid_mask = np.abs(cent_diff) <= 300

            cent_diff = cent_diff[cent_valid_mask]
            time_valid = time_valid[cent_valid_mask]

            if len(cent_diff) == 0:
                continue

            avg_abs_cent_error = np.mean(np.abs(cent_diff))
            max_abs_cent_error = np.max(np.abs(cent_diff))

            cent_stability_std = np.std(cent_diff)
            stability_score = max(0, 100 - (cent_stability_std / 4))

            accuracy_score = max(0, 100 - (avg_abs_cent_error / 3))

            result = {
                "avg_abs_cent_error": round(float(avg_abs_cent_error), 2),
                "max_abs_cent_error": round(float(max_abs_cent_error), 2),
                "accuracy_score": round(float(accuracy_score), 2),
                "best_time_shift": round(float(time_shift), 2),
                "cent_stability_std": round(float(cent_stability_std), 2),
                "stability_score": round(float(stability_score), 2),
                "cent_diff": cent_diff,
                "cent_times": time_valid
            }

            if best_result is None:
                best_result = result
            elif result["avg_abs_cent_error"] < best_result["avg_abs_cent_error"]:
                best_result = result

        if best_result is None:
            return None

        # 구간별 Accuracy 계산
        segment_length = 10
        total_duration = best_result["cent_times"][-1]
        segment_results = []

        for start_time in np.arange(0, total_duration, segment_length):
            end_time = start_time + segment_length

            segment_mask = (
                (best_result["cent_times"] >= start_time)
                & (best_result["cent_times"] < end_time)
            )

            segment_cent = best_result["cent_diff"][segment_mask]

            if len(segment_cent) > 5:
                segment_avg_error = np.mean(np.abs(segment_cent))
                segment_accuracy = max(0, 100 - (segment_avg_error / 3))

                
                segment_avg_cent = np.mean(segment_cent)

                segment_results.append({
                    "start": round(float(start_time), 1),
                    "end": round(float(end_time), 1),
                    "avg_error": round(float(segment_avg_error), 2),
                    "avg_cent": round(float(segment_avg_cent), 2),
                    "accuracy": round(float(segment_accuracy), 2)
                })

        worst_segment = None

        if len(segment_results) > 0:
            worst_segment = max(segment_results, key=lambda x: x["avg_error"])
        
        # 가장 높게 부른 구간
        highest_segment = None

        if len(segment_results) > 0:
            highest_segment = max(
                segment_results,
                key=lambda x: x["avg_cent"]
            )

        # 가장 낮게 부른 구간
        lowest_segment = None

        if len(segment_results) > 0:
            lowest_segment = min(
                segment_results,
                key=lambda x: x["avg_cent"]
            )

        best_result["highest_segment"] = highest_segment
        best_result["lowest_segment"] = lowest_segment

        best_result["segment_results"] = segment_results
        best_result["worst_segment"] = worst_segment

        return best_result

    def load_records():
        records_file = "records.json"

        if not os.path.exists(records_file):
            return []

        with open(records_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def get_accuracy_comment(accuracy_score):
        if accuracy_score >= 90:
            return "매우 정확하게 따라 부른 편이에요."
        elif accuracy_score >= 80:
            return "전체적으로 꽤 정확하게 따라 부른 편이에요."
        elif accuracy_score >= 70:
            return "멜로디 흐름은 괜찮지만 일부 구간에서 오차가 있어요."
        elif accuracy_score >= 60:
            return "따라가고는 있지만 음정 오차가 꽤 있는 편이에요."
        else:
            return "아직은 참고용으로만 봐주세요. 기준 파일 상태나 시간축 차이도 영향을 줄 수 있어요."


    def analyze_pitch(uploaded_file, label):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        y, sr = librosa.load(temp_path)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6")
        )

        times = librosa.times_like(f0, sr=sr)

        rms = librosa.feature.rms(y=y)[0]
        rms = librosa.util.fix_length(rms, size=len(f0))

        volume_threshold = np.mean(rms) * 0.5

        valid_mask = (
            (~np.isnan(f0))
            & (f0 > 80)
            & (rms > volume_threshold)
        )

        valid_pitch = f0[valid_mask]
        valid_times = times[valid_mask]

        if len(valid_pitch) == 0:
            return None

        valid_times, valid_pitch = remove_short_pitch_spikes(
            valid_times,
            valid_pitch,
            min_duration=0.2,
            jump_ratio=1.8
        )

        if len(valid_pitch) == 0:
            return None

        avg_pitch = np.mean(valid_pitch)
        median_pitch = np.median(valid_pitch)
        pitch_std = np.std(valid_pitch)

        highest_candidates = valid_pitch[valid_pitch <= (median_pitch * 2.0)]
        lowest_candidates = valid_pitch[valid_pitch >= (median_pitch * 0.3)]

        if len(highest_candidates) == 0:
            highest_candidates = valid_pitch

        if len(lowest_candidates) == 0:
            lowest_candidates = valid_pitch

        max_pitch = np.max(highest_candidates)
        min_pitch = np.min(lowest_candidates)

        avg_note = librosa.hz_to_note(avg_pitch)
        median_note = librosa.hz_to_note(median_pitch)
        max_note = librosa.hz_to_note(max_pitch)
        min_note = librosa.hz_to_note(min_pitch)

        max_index = np.where(valid_pitch == max_pitch)[0][0]
        min_index = np.where(valid_pitch == min_pitch)[0][0]

        max_time = valid_times[max_index]
        min_time = valid_times[min_index]

        return {
            "label": label,
            "avg_pitch": round(float(avg_pitch), 2),
            "avg_note": avg_note,
            "median_pitch": round(float(median_pitch), 2),
            "median_note": median_note,
            "highest_pitch": round(float(max_pitch), 2),
            "highest_note": max_note,
            "highest_time": round(float(max_time), 1),
            "lowest_pitch": round(float(min_pitch), 2),
            "lowest_note": min_note,
            "lowest_time": round(float(min_time), 1),
            "pitch_std": round(float(pitch_std), 2),
            "valid_pitch": valid_pitch,
            "valid_times": valid_times,
            "max_time": max_time,
            "max_pitch": max_pitch,
            "max_note": max_note,
            "min_time": min_time,
            "min_pitch": min_pitch,
            "min_note": min_note
        }


    if reference_file is not None:
        st.subheader("기준 보컬 미리듣기")
        st.audio(reference_file)

    if my_file is not None:
        st.subheader("내 보컬 미리듣기")
        st.audio(my_file)

    if st.button("비교 분석 시작"):
        if reference_file is None or my_file is None:
            st.warning("기준 보컬 파일과 내 보컬 파일을 모두 업로드해주세요.")
        else:
            with st.spinner("비교 분석중입니다... 잠시만 기다려주세요."):
                ref_result = analyze_pitch(reference_file, "기준 보컬")
                my_result = analyze_pitch(my_file, "내 보컬")

                if ref_result is None:
                    st.warning("기준 보컬에서 pitch를 찾지 못했어요.")
                elif my_result is None:
                    st.warning("내 보컬에서 pitch를 찾지 못했어요.")
                else:
                    st.session_state.comparison_record = {
                        "song": song_name,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "reference": ref_result,
                        "my_vocal": my_result
                    }


    if st.session_state.comparison_record is not None:
        record = st.session_state.comparison_record
        ref = record["reference"]
        my = record["my_vocal"]

        avg_diff = my["avg_pitch"] - ref["avg_pitch"]
        median_diff = my["median_pitch"] - ref["median_pitch"]

        avg_semitone_diff = calculate_semitone_diff(my["avg_pitch"], ref["avg_pitch"])
        median_semitone_diff = calculate_semitone_diff(my["median_pitch"], ref["median_pitch"])

        estimated_key_shift = round(median_semitone_diff)

        correction_ratio = 2 ** (-estimated_key_shift / 12)
        corrected_my_pitch = my["valid_pitch"] * correction_ratio

        ref_stability = max(0, 100 - ref["pitch_std"])
        my_stability = max(0, 100 - my["pitch_std"])

        accuracy_result = calculate_accuracy(
        ref["valid_times"],
        ref["valid_pitch"],
        my["valid_times"],
        corrected_my_pitch
    )

        # Stability 계산
        ref_stability = max(0, 100 - ref["pitch_std"])
        my_stability = max(0, 100 - my["pitch_std"])

        # =========================
        # 핵심 결과 요약
        # =========================
        st.subheader("분석 요약")

        col1, col2 = st.columns(2)

        with col1:
            if accuracy_result is not None:
                st.metric("Accuracy", f"{accuracy_result['accuracy_score']}점")
            else:
                st.metric("Accuracy", "계산 실패")

            st.metric("추정 키 차이", f"{estimated_key_shift}키")

        with col2:
            if accuracy_result is not None:
                st.metric("자동 시간 보정", f"{accuracy_result['best_time_shift']}초")
            else:
                st.metric("자동 시간 보정", "-")

            if accuracy_result is not None:
                st.metric("Stability", f"{accuracy_result['stability_score']}점")
            else:
                st.metric("Stability", "계산 실패")

        if estimated_key_shift < 0:
            st.info(
                f"내 보컬은 기준보다 약 {abs(estimated_key_shift)}키 낮게 부른 것으로 추정됩니다. "
                f"보정 그래프에서는 내 보컬 pitch를 내부적으로 {abs(estimated_key_shift)}키 올려서 비교합니다."
            )
        elif estimated_key_shift > 0:
            st.info(
                f"내 보컬은 기준보다 약 {estimated_key_shift}키 높게 부른 것으로 추정됩니다. "
                f"보정 그래프에서는 내 보컬 pitch를 내부적으로 {estimated_key_shift}키 내려서 비교합니다."
            )
        else:
            st.info("내 보컬은 기준과 거의 같은 키로 부른 것으로 추정됩니다.")

        # =========================
        # 코칭 요약
        # =========================
        if accuracy_result is not None:
            st.subheader("코칭 요약")

            if accuracy_result["worst_segment"] is not None:
                worst = accuracy_result["worst_segment"]
                st.write(
                    f"가장 오차가 큰 구간: "
                    f"{worst['start']}초 ~ {worst['end']}초 "
                    f"/ 평균 오차 {worst['avg_error']} cent "
                    f"/ Accuracy {worst['accuracy']}점"
                )

            if accuracy_result["highest_segment"] is not None:
                high = accuracy_result["highest_segment"]
                st.write(
                    f"가장 높게 부른 구간: "
                    f"{high['start']}초 ~ {high['end']}초 "
                    f"/ 평균 {high['avg_cent']:+.2f} cent"
                )

            if accuracy_result["lowest_segment"] is not None:
                low = accuracy_result["lowest_segment"]
                st.write(
                    f"가장 낮게 부른 구간: "
                    f"{low['start']}초 ~ {low['end']}초 "
                    f"/ 평균 {low['avg_cent']:+.2f} cent"
                )

            st.info(get_accuracy_comment(accuracy_result["accuracy_score"]))

        # =========================
        # 그래프 영역
        # =========================
        st.subheader("그래프")

        graph_tab1, graph_tab2, graph_tab3 = st.tabs(
            ["Pitch 비교", "키 보정 비교", "Cent 오차"]
        )

        with graph_tab1:
            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(ref["valid_times"], ref["valid_pitch"], label="Reference Vocal")
            ax.plot(my["valid_times"], my["valid_pitch"], label="My Vocal")

            ax.scatter(ref["max_time"], ref["max_pitch"], s=80, label=f"Ref Max {ref['max_note']}")
            ax.scatter(my["max_time"], my["max_pitch"], s=80, label=f"My Max {my['max_note']}")

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Pitch (Hz)")
            ax.set_title("Reference Vocal vs My Vocal Pitch")
            ax.legend()

            st.pyplot(fig)

        with graph_tab2:
            fig2, ax2 = plt.subplots(figsize=(12, 5))

            ax2.plot(ref["valid_times"], ref["valid_pitch"], label="Reference Vocal")
            ax2.plot(
                my["valid_times"],
                corrected_my_pitch,
                label=f"My Vocal Corrected ({-estimated_key_shift:+} key)"
            )

            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Pitch (Hz)")
            ax2.set_title("Reference Vocal vs Corrected My Vocal Pitch")
            ax2.legend()

            st.pyplot(fig2)

        with graph_tab3:
            if accuracy_result is not None:
                fig3, ax3 = plt.subplots(figsize=(12, 4))

                ax3.plot(
                    accuracy_result["cent_times"],
                    accuracy_result["cent_diff"],
                    label="Cent Error"
                )

                ax3.axhline(0, linestyle="--", linewidth=1)
                ax3.axhline(50, linestyle="--", linewidth=1)
                ax3.axhline(-50, linestyle="--", linewidth=1)
                ax3.axhline(100, linestyle="--", linewidth=1)
                ax3.axhline(-100, linestyle="--", linewidth=1)

                ax3.set_xlabel("Time (seconds)")
                ax3.set_ylabel("Cent Error")
                ax3.set_title("Pitch Error Over Time")
                ax3.legend()

                st.pyplot(fig3)
            else:
                st.warning("Cent 오차 그래프를 표시할 수 없어요.")

        # =========================
        # 상세 분석 접기
        # =========================
        with st.expander("상세 분석 보기"):
            detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(
                ["보컬 분석", "키/비교", "Accuracy", "Stability"]
            )

            with detail_tab1:
                st.subheader("기준 보컬 분석")
                st.write(f"평균 Pitch: {ref['avg_pitch']} Hz / {ref['avg_note']}")
                st.write(f"중앙 Pitch: {ref['median_pitch']} Hz / {ref['median_note']}")
                st.write(f"최고음: {ref['highest_pitch']} Hz / {ref['highest_note']} / {ref['highest_time']}초")
                st.write(f"최저음: {ref['lowest_pitch']} Hz / {ref['lowest_note']} / {ref['lowest_time']}초")
                st.write(f"Pitch 흔들림 정도: {ref['pitch_std']}")

                st.subheader("내 보컬 분석")
                st.write(f"평균 Pitch: {my['avg_pitch']} Hz / {my['avg_note']}")
                st.write(f"중앙 Pitch: {my['median_pitch']} Hz / {my['median_note']}")
                st.write(f"최고음: {my['highest_pitch']} Hz / {my['highest_note']} / {my['highest_time']}초")
                st.write(f"최저음: {my['lowest_pitch']} Hz / {my['lowest_note']} / {my['lowest_time']}초")
                st.write(f"Pitch 흔들림 정도: {my['pitch_std']}")

            with detail_tab2:
                st.subheader("간단 비교")
                st.write(f"내 보컬 평균 pitch가 기준보다 약 {abs(avg_diff):.2f} Hz {'높아요' if avg_diff > 0 else '낮아요'}.")
                st.write(f"내 보컬 중앙 pitch가 기준보다 약 {abs(median_diff):.2f} Hz {'높아요' if median_diff > 0 else '낮아요'}.")

                st.subheader("추정 키 차이")
                st.write(f"평균 기준 추정 차이: {avg_semitone_diff:.2f} semitone")
                st.write(f"중앙값 기준 추정 차이: {median_semitone_diff:.2f} semitone")
                st.write(f"추정 키 차이: {estimated_key_shift}키")
                st.write(f"내부 보정 비율: {correction_ratio:.4f}")

            with detail_tab3:
                st.subheader("Accuracy 상세")

                if accuracy_result is not None:
                    st.write(f"평균 음정 오차: {accuracy_result['avg_abs_cent_error']} cent")
                    st.write(f"최대 음정 오차: {accuracy_result['max_abs_cent_error']} cent")
                    st.write(f"자동 시간 보정: {accuracy_result['best_time_shift']}초")
                    st.write(f"Accuracy 점수: {accuracy_result['accuracy_score']} / 100")

                    st.subheader("구간별 Accuracy")

                    for segment in accuracy_result["segment_results"]:
                        st.write(
                            f"{segment['start']}초 ~ {segment['end']}초 : "
                            f"{segment['accuracy']}점 "
                            f"(평균 오차 {segment['avg_error']} cent / "
                            f"평균 {segment['avg_cent']:+.2f} cent)"
                        )
                else:
                    st.warning("Accuracy를 계산할 수 없었어요.")

            with detail_tab4:
                st.subheader("Stability 분석")

                if accuracy_result is not None:
                    st.write(f"Cent 흔들림 정도: {accuracy_result['cent_stability_std']} cent")
                    st.write(f"Stability 점수: {accuracy_result['stability_score']} / 100")

                    if accuracy_result["stability_score"] >= 85:
                        st.info("음정 유지가 꽤 안정적인 편이에요.")
                    elif accuracy_result["stability_score"] >= 70:
                        st.info("전체적으로 괜찮지만 일부 구간에서 흔들림이 있어요.")
                    elif accuracy_result["stability_score"] >= 55:
                        st.info("음정 흐름은 따라가지만 흔들림이 조금 있는 편이에요.")
                    else:
                        st.info("흔들림이 크게 잡혔어요. 다만 기준 파일 품질이나 시간축 차이도 영향을 줄 수 있어요.")
                else:
                    st.warning("Stability를 계산할 수 없었어요.")

        save_record = {
            "song": record["song"],
            "date": record["date"],
            "reference_avg_pitch": ref["avg_pitch"],
            "reference_avg_note": ref["avg_note"],
            "reference_median_pitch": ref["median_pitch"],
            "reference_median_note": ref["median_note"],
            "my_avg_pitch": my["avg_pitch"],
            "my_avg_note": my["avg_note"],
            "my_median_pitch": my["median_pitch"],
            "my_median_note": my["median_note"],
            "avg_pitch_diff": round(float(avg_diff), 2),
            "median_pitch_diff": round(float(median_diff), 2),
            "avg_semitone_diff": round(float(avg_semitone_diff), 2),
            "median_semitone_diff": round(float(median_semitone_diff), 2),
            "estimated_key_shift": int(estimated_key_shift),
            "correction_ratio": round(float(correction_ratio), 4),
            "reference_highest_pitch": ref["highest_pitch"],
            "reference_highest_note": ref["highest_note"],
            "my_highest_pitch": my["highest_pitch"],
            "my_highest_note": my["highest_note"],
            "reference_lowest_pitch": ref["lowest_pitch"],
            "reference_lowest_note": ref["lowest_note"],
            "my_lowest_pitch": my["lowest_pitch"],
            "my_lowest_note": my["lowest_note"],
            "reference_pitch_std": ref["pitch_std"],
            "my_pitch_std": my["pitch_std"],
            "best_time_shift": accuracy_result["best_time_shift"] if accuracy_result else None,
            "accuracy_score": accuracy_result["accuracy_score"] if accuracy_result else None,
            "avg_abs_cent_error": accuracy_result["avg_abs_cent_error"] if accuracy_result else None,
            "max_abs_cent_error": accuracy_result["max_abs_cent_error"] if accuracy_result else None,
            "stability_score": accuracy_result["stability_score"] if accuracy_result else None,
            "cent_stability_std": accuracy_result["cent_stability_std"] if accuracy_result else None,
            "worst_segment_start": accuracy_result["worst_segment"]["start"] if accuracy_result and accuracy_result["worst_segment"] else None,
            "worst_segment_end": accuracy_result["worst_segment"]["end"] if accuracy_result and accuracy_result["worst_segment"] else None,
            "worst_segment_avg_error": accuracy_result["worst_segment"]["avg_error"] if accuracy_result and accuracy_result["worst_segment"] else None,
            "worst_segment_accuracy": accuracy_result["worst_segment"]["accuracy"] if accuracy_result and accuracy_result["worst_segment"] else None
        }

        if st.button("비교 분석 결과 저장"):
            records_file = "records.json"

            if os.path.exists(records_file):
                with open(records_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            else:
                records = []

            records.append(save_record)

            with open(records_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=4)

            st.success("비교 분석 결과가 records.json에 저장되었습니다.")

with tab_records:
    st.subheader("📊 성장 기록 대시보드")

    records = load_records()

    if len(records) == 0:
        st.warning("아직 저장된 분석 기록이 없습니다.")
    else:
        df = pd.DataFrame(records)

        df["accuracy_score"] = pd.to_numeric(df["accuracy_score"], errors="coerce")
        df["stability_score"] = pd.to_numeric(df["stability_score"], errors="coerce")
        df["estimated_key_shift"] = pd.to_numeric(df["estimated_key_shift"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        song_options = ["전체"] + sorted(df["song"].dropna().unique().tolist())
        selected_song = st.selectbox("곡별 필터", song_options)

        if selected_song != "전체":
            view_df = df[df["song"] == selected_song]
        else:
            view_df = df

        st.markdown("### 🧾 평균 요약")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("평균 Accuracy", f"{view_df['accuracy_score'].mean():.2f}점")

        with col2:
            st.metric("평균 Stability", f"{view_df['stability_score'].mean():.2f}점")

        with col3:
            st.metric("평균 키 차이", f"{view_df['estimated_key_shift'].mean():.2f}키")

        best_accuracy = view_df.loc[view_df["accuracy_score"].idxmax()]
        best_stability = view_df.loc[view_df["stability_score"].idxmax()]
        hardest_song = view_df.loc[view_df["accuracy_score"].idxmin()]

        st.markdown("### 🐾 냐냐치 보컬 프로파일")

        st.info(
            f"""
            평균 Accuracy는 **{view_df['accuracy_score'].mean():.2f}점**입니다.

            평균 Stability는 **{view_df['stability_score'].mean():.2f}점**입니다.

            평균 키 차이는 **{view_df['estimated_key_shift'].mean():.2f}키**입니다.

            가장 Accuracy가 높은 곡은 **{best_accuracy['song']}**입니다.

            가장 Stability가 높은 곡은 **{best_stability['song']}**입니다.

            현재 기준 가장 어려운 곡은 **{hardest_song['song']}**입니다.
            """
        )

        st.markdown("### 📋 저장된 기록 목록")

        display_cols = [
            "date",
            "song",
            "accuracy_score",
            "stability_score",
            "estimated_key_shift",
            "my_highest_note",
            "my_lowest_note",
            "avg_abs_cent_error",
            "max_abs_cent_error"
        ]

        existing_cols = [col for col in display_cols if col in view_df.columns]

        st.dataframe(
            view_df[existing_cols].sort_values("date", ascending=False),
            use_container_width=True
        )

        st.markdown("### 🔁 같은 곡 과거 vs 현재 비교")

        compare_song = st.selectbox(
            "비교할 곡 선택",
            sorted(df["song"].dropna().unique().tolist())
        )

        song_df = df[df["song"] == compare_song].sort_values("date")

        if len(song_df) < 2:
            st.warning("이 곡은 비교할 기록이 2개 이상 필요합니다.")
        else:
            old_record = song_df.iloc[0]
            new_record = song_df.iloc[-1]

            acc_diff = new_record["accuracy_score"] - old_record["accuracy_score"]
            stab_diff = new_record["stability_score"] - old_record["stability_score"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Accuracy 변화",
                    f"{new_record['accuracy_score']:.2f}점",
                    f"{acc_diff:+.2f}"
                )

            with col2:
                st.metric(
                    "Stability 변화",
                    f"{new_record['stability_score']:.2f}점",
                    f"{stab_diff:+.2f}"
                )
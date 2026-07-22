import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import json
import logging
from datetime import datetime
import os
import pandas as pd

from translations import TEXT


logger = logging.getLogger(__name__)


class AudioAnalysisError(Exception):
    pass


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



def get_accuracy_comment(accuracy_score, t):
    if accuracy_score >= 90:
        return t["accuracy_comment_90"]
    elif accuracy_score >= 80:
        return t["accuracy_comment_80"]
    elif accuracy_score >= 70:
        return t["accuracy_comment_70"]
    elif accuracy_score >= 60:
        return t["accuracy_comment_60"]
    else:
        return t["accuracy_comment_low"]


def analyze_pitch(uploaded_file, label):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.read())

        y, sr = librosa.load(temp_path)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6")
        )
    except Exception as exc:
        logger.exception(
            "Audio analysis failed for %s (%s)",
            label,
            getattr(uploaded_file, "name", "unknown file")
        )
        raise AudioAnalysisError from exc
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                logger.exception("Failed to remove temporary audio file: %s", temp_path)

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


if "comparison_record" not in st.session_state:
    st.session_state.comparison_record = None

if "comparison_reference_file_id" not in st.session_state:
    st.session_state.comparison_reference_file_id = None

if "comparison_my_file_id" not in st.session_state:
    st.session_state.comparison_my_file_id = None


language = st.selectbox(
    "🌐 Language / 언어 / 言語",
    ["한국어", "English", "日本語"]
)

t = TEXT[language]

st.title(t["title"])

tab_analyze, tab_records = st.tabs([
    t["tab_analyze"],
    t["tab_records"]
])


with tab_analyze:
    st.write(t["description"])
    st.write(t["wav_notice"])
    st.info(t["recommend"])

    song_name = st.text_input(t["song_input"], value="Unknown Song")

    reference_file = st.file_uploader(t["ref_upload"], type=["wav"], key="reference_file")
    my_file = st.file_uploader(t["my_upload"], type=["wav"], key="my_file")

    reference_file_id = reference_file.file_id if reference_file is not None else None
    my_file_id = my_file.file_id if my_file is not None else None

    if st.session_state.comparison_record is not None:
        uploads_changed = (
            reference_file_id != st.session_state.comparison_reference_file_id
            or my_file_id != st.session_state.comparison_my_file_id
        )

        if uploads_changed:
            st.session_state.comparison_record = None

    if reference_file is not None:
        st.subheader(t["ref_preview"])
        st.audio(reference_file)

    if my_file is not None:
        st.subheader(t["my_preview"])
        st.audio(my_file)

    if st.button(t["start_button"]):
        if reference_file is None or my_file is None:
            st.warning(t["need_files"])
        else:
            with st.spinner(t["analyzing"]):
                ref_error = False
                my_error = False

                try:
                    ref_result = analyze_pitch(reference_file, t["ref_preview"])
                except AudioAnalysisError:
                    ref_result = None
                    ref_error = True
                    st.error(t["audio_processing_error"].format(label=t["ref_preview"]))

                try:
                    my_result = analyze_pitch(my_file, t["my_preview"])
                except AudioAnalysisError:
                    my_result = None
                    my_error = True
                    st.error(t["audio_processing_error"].format(label=t["my_preview"]))

                if ref_error or my_error:
                    pass
                elif ref_result is None:
                    st.warning(t["ref_pitch_not_found"])
                elif my_result is None:
                    st.warning(t["my_pitch_not_found"])
                else:
                    st.session_state.comparison_record = {
                        "song": song_name,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "reference": ref_result,
                        "my_vocal": my_result
                    }
                    st.session_state.comparison_reference_file_id = reference_file_id
                    st.session_state.comparison_my_file_id = my_file_id


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
        st.subheader(t.get("analysis_summary", "분석 요약"))

        def get_performance_label(score):
            if score is None:
                return None

            try:
                score = float(score)
            except (TypeError, ValueError):
                return None

            if not np.isfinite(score):
                return None
            if score >= 90:
                return t["performance_excellent"]
            if score >= 80:
                return t["performance_very_good"]
            if score >= 70:
                return t["performance_good"]
            if score >= 60:
                return t["performance_needs_practice"]
            return t["performance_needs_focus"]

        col1, col2 = st.columns(2)

        with col1:
            if accuracy_result is not None:
                st.metric("Accuracy", f"{accuracy_result['accuracy_score']}{t['point']}")
                accuracy_label = get_performance_label(accuracy_result["accuracy_score"])
                if accuracy_label is not None:
                    st.caption(f"**{accuracy_label}**")
            else:
                st.metric("Accuracy", t["calculation_failed"])

            st.metric(t["estimated_key_shift"], f"{estimated_key_shift}{t['key_unit']}")

        with col2:
            if accuracy_result is not None:
                st.metric(t["auto_time_shift"], f"{accuracy_result['best_time_shift']}{t['sec_unit']}")
            else:
                st.metric(t["auto_time_shift"], "-")

            if accuracy_result is not None:
                st.metric("Stability", f"{accuracy_result['stability_score']}{t['point']}")
                stability_label = get_performance_label(accuracy_result["stability_score"])
                if stability_label is not None:
                    st.caption(f"**{stability_label}**")
            else:
                st.metric("Stability", t["calculation_failed"])

        if estimated_key_shift < 0:
            st.info(t["key_shift_lower"].format(n=abs(estimated_key_shift)))
        elif estimated_key_shift > 0:
            st.info(t["key_shift_higher"].format(n=estimated_key_shift))
        else:
            st.info(t["key_shift_same"])

        # =========================
        # 코칭 요약
        # =========================
        worst_time_range = None

        if accuracy_result is not None:
            st.subheader(t.get("coaching_summary", "코칭 요약"))
            coaching_diagnosis = [
                get_accuracy_comment(accuracy_result["accuracy_score"], t)
            ]

            if accuracy_result["worst_segment"] is not None:
                worst = accuracy_result["worst_segment"]
                start_seconds = int(round(worst["start"]))
                end_seconds = int(round(worst["end"]))
                worst_time_range = (
                    f"{start_seconds // 60:02d}:{start_seconds % 60:02d}–"
                    f"{end_seconds // 60:02d}:{end_seconds % 60:02d}"
                )
                coaching_diagnosis.append(
                    t["coaching_weakest"].format(time_range=worst_time_range)
                )

            if accuracy_result["stability_score"] < 70:
                coaching_diagnosis.append(t["coaching_stability"])

            if worst_time_range is not None:
                coaching_diagnosis.append(t["coaching_priority_weakest"])
            elif accuracy_result["accuracy_score"] < 70:
                coaching_diagnosis.append(t["coaching_priority_accuracy"])
            elif accuracy_result["stability_score"] < 70:
                coaching_diagnosis.append(t["coaching_priority_stability"])

            for diagnosis in coaching_diagnosis:
                st.markdown(f"- {diagnosis}")

        practice_recommendations = []

        if accuracy_result is not None:
            if worst_time_range is not None:
                practice_recommendations.append((
                    t["practice_weakest_title"],
                    t["practice_weakest_body"].format(time_range=worst_time_range)
                ))

            if accuracy_result["accuracy_score"] < 70:
                practice_recommendations.append((
                    t["practice_accuracy_title"],
                    t["practice_accuracy_body"]
                ))

            if accuracy_result["stability_score"] < 70:
                practice_recommendations.append((
                    t["practice_stability_title"],
                    t["practice_stability_body"]
                ))

        if practice_recommendations:
            with st.container(border=True):
                st.subheader(f"🎯 {t['todays_practice']}")

                for priority, (title, body) in enumerate(practice_recommendations[:3], start=1):
                    st.markdown(f"**{t['practice_priority'].format(number=priority)} · {title}**")
                    st.write(body)

        # =========================
        # 그래프 영역
        # =========================
        st.subheader(t.get("graphs", "그래프"))

        graph_tab1, graph_tab2, graph_tab3 = st.tabs(
            [t.get("pitch_graph", "Pitch 비교"), t.get("corrected_graph", "키 보정 비교"), t.get("cent_error_graph", "Cent 오차")]
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
                st.warning(t["cent_graph_unavailable"])

        # =========================
        # 상세 분석 접기
        # =========================
        with st.expander(t.get("detail_analysis", "상세 분석 보기")):
            detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(
                [t.get("vocal_analysis", "보컬 분석"), t.get("key_compare", "키/비교"), "Accuracy", "Stability"]
            )

            with detail_tab1:
                st.subheader(t["reference_vocal_analysis"])
                st.write(f"{t['avg_pitch']}: {ref['avg_pitch']} Hz / {ref['avg_note']}")
                st.write(f"{t['median_pitch']}: {ref['median_pitch']} Hz / {ref['median_note']}")
                st.write(f"{t['highest_note']}: {ref['highest_pitch']} Hz / {ref['highest_note']} / {ref['highest_time']}{t['sec_unit']}")
                st.write(f"{t['lowest_note']}: {ref['lowest_pitch']} Hz / {ref['lowest_note']} / {ref['lowest_time']}{t['sec_unit']}")
                st.write(f"{t['pitch_std']}: {ref['pitch_std']}")

                st.subheader(t["my_vocal_analysis"])
                st.write(f"{t['avg_pitch']}: {my['avg_pitch']} Hz / {my['avg_note']}")
                st.write(f"{t['median_pitch']}: {my['median_pitch']} Hz / {my['median_note']}")
                st.write(f"{t['highest_note']}: {my['highest_pitch']} Hz / {my['highest_note']} / {my['highest_time']}{t['sec_unit']}")
                st.write(f"{t['lowest_note']}: {my['lowest_pitch']} Hz / {my['lowest_note']} / {my['lowest_time']}{t['sec_unit']}")
                st.write(f"{t['pitch_std']}: {my['pitch_std']}")

            with detail_tab2:
                st.subheader(t["simple_comparison"])
                st.write(t["my_avg_pitch_compare"].format(
                    value=abs(avg_diff),
                    direction=t["higher"] if avg_diff > 0 else t["lower"]
                ))
                st.write(t["my_median_pitch_compare"].format(
                    value=abs(median_diff),
                    direction=t["higher"] if median_diff > 0 else t["lower"]
                ))

                st.subheader(t["estimated_key_shift_title"])
                st.write(f"{t['avg_semitone_diff']}: {avg_semitone_diff:.2f} semitone")
                st.write(f"{t['median_semitone_diff']}: {median_semitone_diff:.2f} semitone")
                st.write(f"{t['estimated_key_shift_title']}: {estimated_key_shift}{t['key_unit']}")
                st.write(f"{t['correction_ratio']}: {correction_ratio:.4f}")

            with detail_tab3:
                st.subheader(t["accuracy_detail"])

                if accuracy_result is not None:
                    st.write(f"{t['avg_pitch_error']}: {accuracy_result['avg_abs_cent_error']} cent")
                    st.write(f"{t['max_pitch_error']}: {accuracy_result['max_abs_cent_error']} cent")
                    st.write(f"{t['auto_time_shift']}: {accuracy_result['best_time_shift']}{t['sec_unit']}")
                    st.write(f"{t['accuracy_score']}: {accuracy_result['accuracy_score']} / 100")

                    st.subheader(t["segment_accuracy"])

                    for segment in accuracy_result["segment_results"]:
                        st.write(t["segment_accuracy_line"].format(
                            start=segment["start"],
                            end=segment["end"],
                            accuracy=segment["accuracy"],
                            avg_error=segment["avg_error"],
                            avg_cent=segment["avg_cent"]
                        ))
                else:
                    st.warning(t["accuracy_unavailable"])

            with detail_tab4:
                st.subheader(t["stability_analysis"])

                if accuracy_result is not None:
                    st.write(f"{t['cent_stability']}: {accuracy_result['cent_stability_std']} cent")
                    st.write(f"{t['stability_score']}: {accuracy_result['stability_score']} / 100")

                    if accuracy_result["stability_score"] >= 85:
                        st.info(t["stability_comment_85"])
                    elif accuracy_result["stability_score"] >= 70:
                        st.info(t["stability_comment_70"])
                    elif accuracy_result["stability_score"] >= 55:
                        st.info(t["stability_comment_55"])
                    else:
                        st.info(t["stability_comment_low"])
                else:
                    st.warning(t["stability_unavailable"])

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

        if st.button(t.get("save_button", "비교 분석 결과 저장")):
            records_file = "records.json"

            if os.path.exists(records_file):
                with open(records_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            else:
                records = []

            records.append(save_record)

            with open(records_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=4)

            st.success(t.get("save_success", "비교 분석 결과가 records.json에 저장되었습니다."))



with tab_records:
    st.subheader(t["records_title"])

    records_file = "records.json"

    if not os.path.exists(records_file):
        st.warning(t.get("no_records", "아직 저장된 분석 기록이 없습니다."))
    else:
        try:
            with open(records_file, "r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError:
            records = []

        if len(records) == 0:
            st.warning(t.get("no_records", "아직 저장된 분석 기록이 없습니다."))
        else:
            df = pd.DataFrame(records)

            required_cols = ["song", "date", "accuracy_score", "stability_score", "estimated_key_shift"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None

            df["accuracy_score"] = pd.to_numeric(df["accuracy_score"], errors="coerce")
            df["stability_score"] = pd.to_numeric(df["stability_score"], errors="coerce")
            df["estimated_key_shift"] = pd.to_numeric(df["estimated_key_shift"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            song_options = [t.get("all", "전체")] + sorted(df["song"].dropna().unique().tolist())

            selected_song = st.selectbox(
                t.get("song_filter", "곡별 필터"),
                song_options
            )

            if selected_song != t.get("all", "전체"):
                view_df = df[df["song"] == selected_song]
            else:
                view_df = df

            st.markdown(f"### {t.get('average_summary', '🧾 평균 요약')}")

            col1, col2, col3 = st.columns(3)

            avg_accuracy = view_df["accuracy_score"].mean()
            avg_stability = view_df["stability_score"].mean()
            avg_key_shift = view_df["estimated_key_shift"].mean()

            with col1:
                st.metric("Average Accuracy", f"{avg_accuracy:.2f}{t['point']}")

            with col2:
                st.metric("Average Stability", f"{avg_stability:.2f}{t['point']}")

            with col3:
                st.metric(t["average_key_shift"], f"{avg_key_shift:.2f}{t['key_unit']}")

            st.markdown(f"### {t.get('vocal_profile', '🐾 냐냐치 보컬 프로파일')}")

            if view_df["accuracy_score"].notna().any():
                best_accuracy = view_df.loc[view_df["accuracy_score"].idxmax()]
                hardest_song = view_df.loc[view_df["accuracy_score"].idxmin()]
            else:
                best_accuracy = None
                hardest_song = None

            if view_df["stability_score"].notna().any():
                best_stability = view_df.loc[view_df["stability_score"].idxmax()]
            else:
                best_stability = None

            profile_text = t.get(
                "profile_template",
                "평균 Accuracy는 {avg_accuracy:.2f}점입니다.\n\n평균 Stability는 {avg_stability:.2f}점입니다.\n\n평균 키 차이는 {avg_key_shift:.2f}키입니다.\n\n가장 Accuracy가 높은 곡은 {best_accuracy_song}입니다.\n\n가장 Stability가 높은 곡은 {best_stability_song}입니다.\n\n현재 기준 가장 어려운 곡은 {hardest_song}입니다."
            )

            st.info(profile_text.format(
                avg_accuracy=avg_accuracy,
                avg_stability=avg_stability,
                avg_key_shift=avg_key_shift,
                best_accuracy_song=best_accuracy["song"] if best_accuracy is not None else "-",
                best_stability_song=best_stability["song"] if best_stability is not None else "-",
                hardest_song=hardest_song["song"] if hardest_song is not None else "-"
            ))

            st.markdown(f"### {t.get('record_list', '📋 저장된 기록 목록')}")

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

            st.markdown(f"### {t.get('compare_title', '🔁 같은 곡 과거 vs 현재 비교')}")

            compare_songs = sorted(df["song"].dropna().unique().tolist())

            if len(compare_songs) == 0:
                st.warning(t.get("no_compare_song", "비교할 곡이 없습니다."))
            else:
                compare_song = st.selectbox(
                    t.get("compare_song_select", "비교할 곡 선택"),
                    compare_songs
                )

                song_df = df[df["song"] == compare_song].sort_values("date")

                if len(song_df) < 2:
                    st.warning(t.get("need_two_records", "이 곡은 비교할 기록이 2개 이상 필요합니다."))
                else:
                    old_record = song_df.iloc[0]
                    new_record = song_df.iloc[-1]

                    acc_diff = new_record["accuracy_score"] - old_record["accuracy_score"]
                    stab_diff = new_record["stability_score"] - old_record["stability_score"]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            t["accuracy_change"],
                            f"{new_record['accuracy_score']:.2f}{t['point']}",
                            f"{acc_diff:+.2f}"
                        )

                    with col2:
                        st.metric(
                            t["stability_change"],
                            f"{new_record['stability_score']:.2f}{t['point']}",
                            f"{stab_diff:+.2f}"
                        )

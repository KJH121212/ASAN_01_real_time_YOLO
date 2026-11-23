"""Streamlit 제어 화면에 실시간 좌우 측정 결과 카드를 표시하는 독립 모듈."""

import streamlit as st

QUALITY_POINTS = {"PERFECT": 1.0, "GOOD": 0.5, "BAD": 0.25}
QUALITY_THRESHOLDS = {"PERFECT": 0.70, "GOOD": 0.50, "BAD": 0.35}
QUALITY_CODES = {"PERFECT": 3, "GOOD": 2, "BAD": 1}
QUALITY_COLORS = {"PERFECT": "#16a34a", "GOOD": "#f59e0b", "BAD": "#a855f7"}
QUALITY_LABELS = {"PERFECT": "Perfect", "GOOD": "Good", "BAD": "Normal"}


def quality_result(data, mode="", metric_type=""):
    """측정 결과에서 표시용 등급·라벨·색상·코드·점수를 추출한다."""
    if mode == "CALIBRATION":
        return None
    grade = str(data.get("quality", "")).upper()
    if grade == "NORMAL":
        grade = "BAD"
    if grade not in QUALITY_COLORS:
        return None
    return dict(
        grade=grade,
        label=QUALITY_LABELS[grade],
        color=QUALITY_COLORS[grade],
        code=QUALITY_CODES[grade],
        points=QUALITY_POINTS[grade],
    )


def measurement_values(state):
    """현재 포즈 상태를 받아 좌우 횟수·측정값·품질 배지를 화면에 표시한다."""
    pose = state["pose"]
    for column, side, label in zip(st.columns(2), ("left", "right"), ("왼쪽", "오른쪽")):
        data = pose.get(side, {})
        value = data.get("val")
        unit = (
            "px"
            if pose.get("metric_unit", pose.get("measurement", {}).get("metric_unit")) == "px"
            else ("°" if pose.get("metric_type") == "ANGLE" else "")
        )
        digits = 4 if pose.get("metric_type") in ("COORDINATE", "RAW_COORDINATE") else 2
        text = "--" if value is None else f"{value:.{digits}f}{unit}"
        result = quality_result(state.get("last_results", {}).get(side, data), pose.get("mode", ""), pose.get("metric_type", ""))
        with column.container(border=True):
            st.markdown(f"**{label}** · `{data.get('state', '--')}`")
            reps, metric, score = st.columns(3)
            count = data.get("rep_count", 0)
            reps.metric("횟수", f"{count} / 3" if pose.get("mode") == "CALIBRATION" else count)
            metric.metric("측정값", text)
            total = data.get("total_score")
            score.metric("누적 점수", "--" if pose.get("mode") == "CALIBRATION" or total is None else f"{total:g}")
            if result:
                st.markdown(
                    f'<div style="background:{result["color"]};color:white;border-radius:6px;'
                    f'padding:6px 10px;font-weight:700;text-align:center">'
                    f'최근 판정: {result["label"]} (코드 {result["code"]} · +{result["points"]:g}점)</div>',
                    unsafe_allow_html=True,
                )
            elif pose.get("mode") != "CALIBRATION":
                st.caption("최근 판정: 대기 중 (" + " / ".join(f"{QUALITY_LABELS[k]}({k})" for k in QUALITY_COLORS) + ")")
    if pose.get("mode") == "CALIBRATION":
        st.caption(f"캘리브레이션: 왼쪽 {pose.get('calib_reps_left', 0)}/3 · 오른쪽 {pose.get('calib_reps_right', 0)}/3")

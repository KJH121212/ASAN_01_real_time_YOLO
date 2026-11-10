"""Measurement result cards for the Streamlit control screen."""

import streamlit as st
from utils.quality_display import QUALITY_COLORS, QUALITY_LABELS, quality_result


def measurement_values(state):
    pose = state['pose']
    for column, side, label in zip(st.columns(2), ('left', 'right'), ('왼쪽', '오른쪽')):
        data = pose.get(side, {})
        value = data.get('val')
        unit = ' px' if pose.get('metric_unit') == 'px' else ''
        reps = (state['summary'] or {}).get(f'{side}_reps', data.get('rep_count', 0))
        column.metric(f'{label} 반복 횟수', f'{reps}회')
        column.caption(f"측정값 {'—' if value is None else f'{value:.3f}{unit}'} · 상태 {data.get('state', '대기')}")
        result_data = state.get('last_results', {}).get(side, data)
        result = quality_result(result_data, pose.get('mode', ''), pose.get('metric_type', ''))
        if result:
            column.markdown(f"<div style='border-left:6px solid {result['color']};padding:8px 12px;"
                            f"font-size:24px;font-weight:700;color:{result['color']}'>"
                            f"{result['label']} · +{result['points']:g}점</div>", unsafe_allow_html=True)
            column.caption(f"최근 확정 결과 · score 코드 {result['code']}")
        total = (state['summary'] or {}).get(f'{side}_total_score', result_data.get('total_score', data.get('total_score')))
        if pose.get('mode') == 'MAIN' and total is not None:
            column.metric(f'{label} 누적 점수', f'{total:g}점')
    if (pose.get('mode') == 'MAIN'
            and pose.get('measurement', {}).get('count_on') != 'TARGET_POSITION'):
        st.markdown(' &nbsp; '.join(f"<span style='color:{QUALITY_COLORS[g]};font-weight:700'>{QUALITY_LABELS[g]}</span>"
                                  for g in QUALITY_COLORS), unsafe_allow_html=True)
    if pose.get('is_occluded'):
        st.warning('관절이 가려졌습니다. 카메라에서 몸이 잘 보이는지 확인하세요.')
    if pose.get('mode') == 'CALIBRATION':
        st.caption(f"캘리브레이션: 왼쪽 {pose.get('calib_reps_left', 0)}/3 · 오른쪽 {pose.get('calib_reps_right', 0)}/3")


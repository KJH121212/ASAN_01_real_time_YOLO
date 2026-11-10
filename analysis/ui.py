"""Streamlit 로그 및 영상 분석 탭 (전면 개편 버전)."""

from pathlib import Path
from zipfile import BadZipFile
from datetime import date
import json

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from analysis.log_reader import JOINTS, coordinates, count_events, measurements, read_csv, read_npz, summary
from analysis.metrics import context, diagnostics, repetition_stats, state_segments, threshold_guides, quality_guides
from analysis.skeleton_player import render_skeleton_player_component, smooth_keypoints
from analysis.skeleton_viewer import draw_skeleton_frame, render_skeleton_video
from core.motion_engine import SideFSM
from utils.quality_display import QUALITY_COLORS, QUALITY_LABELS, quality_result

ROOT = Path(__file__).resolve().parent.parent

JOINT_DISPLAY_MAP = {
    'nose': '코 (nose)',
    'left_eye': '왼쪽 눈 (left_eye)',
    'right_eye': '오른쪽 눈 (right_eye)',
    'left_ear': '왼쪽 귀 (left_ear)',
    'right_ear': '오른쪽 귀 (right_ear)',
    'left_shoulder': '왼쪽 어깨 (left_shoulder)',
    'right_shoulder': '오른쪽 어깨 (right_shoulder)',
    'left_elbow': '왼쪽 팔꿈치 (left_elbow)',
    'right_elbow': '오른쪽 팔꿈치 (right_elbow)',
    'left_wrist': '왼쪽 손목 (left_wrist)',
    'right_wrist': '오른쪽 손목 (right_wrist)',
    'left_hip': '왼쪽 골반 (left_hip)',
    'right_hip': '오른쪽 골반 (right_hip)',
    'left_knee': '왼쪽 무릎 (left_knee)',
    'right_knee': '오른쪽 무릎 (right_knee)',
    'left_ankle': '왼쪽 발목 (left_ankle)',
    'right_ankle': '오른쪽 발목 (right_ankle)',
}

STATE_HELP = {
    'BASELINE': '휴식 기준 위치·길이를 수집하는 중입니다.',
    'DOWN': '다음 움직임을 기다리며 저점을 추적합니다.',
    'READY': '다음 움직임을 시작할 수 있는 대기 상태입니다.',
    'RISING': '움직임의 크기 또는 유지 시간을 확인하는 중입니다.',
    'UP': '활성 조건을 충족했습니다. 반전 방식은 피크에서의 하강을, 과거 방식은 복귀를 기다립니다.',
    'PUSHING': '움직임의 피크와 방향 전환을 추적합니다.',
    'RETURNING': '휴식·시작 기준으로 돌아오기를 기다립니다. 이때의 상승은 새 반복으로 인정되지 않을 수 있습니다.',
    'WAITING': '설정된 목표에 도달해 카운트를 대기합니다.',
    'MISSING': '필요한 관절 추적이 끊긴 상태입니다.',
    'FALLING': '하강을 확인했습니다. 다음 상승을 위한 저점을 추적합니다.',
}
DIAGNOSTIC_HELP = {
    '유효 측정 (%)': '선택 구간의 기록 중 측정값이 비어 있지 않은 비율입니다. 판정 정확도를 뜻하지 않습니다.',
    '최솟값': '선택 구간에서 관측된 가장 작은 측정값입니다.',
    '최댓값': '선택 구간에서 관측된 가장 큰 측정값입니다.',
    '변화 폭': '최댓값 - 최솟값입니다. 신체의 실제 가동 각도와 동일한 뜻은 아닙니다.',
    '주요 상태': '선택 구간에서 가장 자주 기록된 판정 상태입니다.',
    '표준편차': '선택 구간에서 측정값이 얼마나 퍼져 있는지 나타냅니다.',
    '5백분위': '유효한 값의 약 5%가 이 값 이하입니다. 극단값을 제외한 범위를 보는 데 사용합니다.',
    '95백분위': '유효한 값의 약 95%가 이 값 이하입니다.',
    '기준 대기 (%)': 'BASELINE 상태로 기록된 프레임 비율입니다.',
    '기준 확보 (%)': '0보다 큰 기준 길이가 저장된 프레임 비율입니다.',
    '활성 연속 유지 (초)': '저장된 측정값이 활성 기준 이상인 가장 긴 연속 구간입니다. 긴 기록 간격과 기준 변경에서 끊습니다.',
    '기준 초과 중 복귀대기 (%)': '활성 기준을 넘은 기록 중 RETURNING 상태인 비율입니다.',
}


def panel_help(title, short, detail):
    st.caption(short)
    with st.expander(f'{title} 읽는 법'):
        st.markdown(detail)


@st.cache_data(show_spinner=False)
def load_file(name, content):
    return read_npz(content) if Path(name).suffix.lower() == '.npz' else read_csv(content)


@st.cache_data(show_spinner=False)
def convert_coordinates(log, mode, confidence):
    return coordinates(log, mode, confidence)


def plot_timeseries(graph, events, title, guides=None, show_points=False):
    """Measurements and threshold categories have independent, explicit colors."""
    time_column = graph.index.name
    values = graph.reset_index().melt(id_vars=time_column, var_name='series', value_name='value')
    color = alt.Color('series:N', title='항목')
    if set(graph.columns).issubset({'left_value', 'right_value'}):
        color = alt.Color('series:N', title='측정값', scale=alt.Scale(
            domain=['left_value', 'right_value'], range=['#2563eb', '#dc2626']))
    line = alt.Chart(values).mark_line(strokeWidth=2.2,
        point=alt.OverlayMarkDef(size=14, opacity=.65) if show_points else False).encode(
        x=alt.X(f'{time_column}:Q', title='시간 (초)' if time_column == 'time_sec' else '행 번호'),
        y=alt.Y('value:Q', title=title, scale=alt.Scale(zero=False)), color=color,
        tooltip=[alt.Tooltip(f'{time_column}:Q', format='.3f'), 'series:N', alt.Tooltip('value:Q', format='.4f')])
    layers = [line]
    if guides is not None and not guides.empty:
        extra = guides.reset_index().melt(id_vars=time_column, var_name='series', value_name='value')
        categories = {'quality_PERFECT': ('Perfect · 70%', QUALITY_COLORS['PERFECT']),
                      'quality_GOOD': ('Good · 50%', QUALITY_COLORS['GOOD']),
                      'quality_BAD': ('Normal · 35%', QUALITY_COLORS['BAD']),
                      'start_val': ('시작 기준', '#64748b'), 'target_val': ('목표 기준', '#111827')}
        extra['kind'] = extra.series.map(lambda name: next((label for suffix, (label, _) in categories.items() if name.endswith(suffix)), '기준선'))
        present = [(label, color) for label, color in categories.values() if label in set(extra.kind)]
        layers.append(alt.Chart(extra).mark_line(interpolate='step-after', strokeWidth=1.8, opacity=.85).encode(
            x=f'{time_column}:Q', y=alt.Y('value:Q', scale=alt.Scale(zero=False)), detail='series:N',
            color=alt.Color('kind:N', title='기준선', scale=alt.Scale(
                domain=[label for label, _ in present], range=[color for _, color in present]),
                legend=alt.Legend(orient='bottom', symbolStrokeWidth=3)),
            tooltip=['kind:N', 'series:N', alt.Tooltip('value:Q', format='.4f')]))
    if not events.empty:
        layers.append(alt.Chart(events).mark_rule(strokeDash=[5, 3], opacity=.75).encode(
            x=f'{time_column}:Q',
            color=alt.Color('side:N', legend=None, scale=alt.Scale(domain=['left', 'right'], range=['#2563eb', '#dc2626'])),
            tooltip=[alt.Tooltip(f'{time_column}:Q', title='카운트 시간', format='.3f'),
                     'side:N', 'rep_count:Q', 'frame_idx:Q']))
    chart = alt.layer(*layers).resolve_scale(color='independent')
    st.altair_chart(chart.properties(height=320).interactive(), width='stretch')


def input_sources():
    method = st.radio('로그 입력', ['경로 선택', '파일 업로드'], horizontal=True, key='log_method')
    if method == '파일 업로드':
        uploaded = st.file_uploader('CSV·NPZ 파일', type=['csv', 'npz'], accept_multiple_files=True,
                                   help='같은 이름의 CSV·NPZ를 함께 올리면 하나의 로그로 표시합니다.')
        sources = {file.name: file.getvalue() for file in uploaded}
        if len(sources) != len(uploaded):
            st.error('같은 이름의 파일이 여러 개입니다. 로그를 나눠서 올리거나 파일명을 구분해 주세요.')
            return {}
    else:
        with st.form('log_path_form'):
            value = st.text_input('폴더 또는 파일 경로', value=str(ROOT / 'data'),
                                 key='log_path',
                                 help='Streamlit이 실행 중인 PC의 경로입니다. 하위 폴더도 검색합니다.')
            submitted = st.form_submit_button('경로 열기', key='log_open')
        if submitted:
            path = Path(value.strip().strip('"')).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            if path.is_file() and path.suffix.lower() in ('.csv', '.npz'):
                files = sorted({path, *(p for suffix in ('.csv', '.npz') if (p := path.with_suffix(suffix)).is_file())})
            elif path.is_dir():
                files = sorted(p for p in path.rglob('*') if p.suffix.lower() in ('.csv', '.npz') and p.is_file())
            else:
                st.error('CSV·NPZ 파일 또는 존재하는 폴더를 입력하세요.')
                files = []
            st.session_state['log_paths'] = [str(p.resolve()) for p in files]
        sources = {name: None for name in st.session_state.get('log_paths', [])}
    return sources


def log_filters(groups):
    """Independent filters: empty selections mean all available logs."""
    identities = {}
    for source in groups:
        parts = Path(source).parts
        identity = dict(patient='미지정', date='미지정', exercise='미지정')
        if len(parts) >= 4:
            try:
                date.fromisoformat(parts[-3])
            except ValueError:
                pass
            else:
                identity = dict(patient=parts[-4], date=parts[-3], exercise=parts[-2])
        identities[source] = identity
    config = ROOT / 'configs/exercise_kpt_config_coco.json'
    exercises = json.loads(config.read_text(encoding='utf-8'))['exercises'] if config.exists() else {}
    keys = ('patient', 'date', 'exercise')

    def reset():
        for key in keys:
            st.session_state[f'log_filter_{key}'] = []

    st.caption('각 필터는 독립적으로 여러 값을 선택할 수 있습니다. 비워 두면 전체입니다. 서로 다른 필터는 모두 만족하는 로그를 표시합니다.')
    selected = {}
    for column, key, label in zip(st.columns(3), keys, ('환자 필터', '날짜 필터', '동작 필터')):
        options = sorted({item[key] for item in identities.values()}, reverse=key == 'date')
        state_key = f'log_filter_{key}'
        if state_key in st.session_state:
            st.session_state[state_key] = [value for value in st.session_state[state_key] if value in options]
        selected[key] = column.multiselect(label, options, key=state_key,
            format_func=(lambda value: exercises.get(value, {}).get('display_name_kr', value)) if key == 'exercise' else str,
            placeholder='전체')
    st.button('필터 초기화', on_click=reset)
    filtered = {source: files for source, files in groups.items()
                if all(not selected[key] or identities[source][key] in selected[key] for key in keys)}
    st.caption(f'전체 {len(groups):,}개 로그 · 필터 결과 {len(filtered):,}개 (같은 이름의 CSV·NPZ는 1개 로그)')
    return filtered


def select_files():
    with st.expander('로그 열기 · 변경', expanded=not st.session_state.get('analysis_loaded', False)):
        st.caption('폴더는 하위 로그를 검색하고, 파일은 같은 이름의 CSV·NPZ를 함께 읽습니다. 다른 PC의 파일은 업로드를 사용하세요.')
        sources = input_sources()
    if not sources:
        st.info('분석할 경로를 열거나 CSV·NPZ 파일을 업로드하세요.')
        return {}, ''
    groups = {}
    for name in sources:
        groups.setdefault(str(Path(name).with_suffix('')), []).append(name)
    groups = log_filters(groups)
    if not groups:
        st.info('조건에 맞는 로그가 없습니다. 필터를 변경하거나 초기화하세요.')
        return {}, ''
    selected = st.selectbox('분석할 로그', list(groups), key='selected_log',
                           format_func=lambda value: ' / '.join(Path(value).parts[-4:]))
    result = {}
    for name in groups[selected]:
        try:
            content = sources[name] if sources[name] is not None else Path(name).read_bytes()
            result[Path(name).suffix.lower()] = load_file(name, content)
        except (OSError, ValueError, EOFError, BadZipFile) as exc:
            st.warning(f'{Path(name).name} 읽기 실패: {exc}')
    st.session_state['analysis_loaded'] = True
    return result, selected


def overview(table, events, visible_events, ctx, time_column, sides, show_thresholds, show_quality=False):
    value_columns = [f'{side}_value' for side in sides if f'{side}_value' in table]
    if value_columns:
        graph = table.set_index(time_column)[value_columns] * ctx['factor']
        guides = None
        if show_thresholds:
            guides = threshold_guides(table, ctx, time_column, sides)
        if show_quality:
            grades = quality_guides(table, ctx, time_column, sides)
            guides = pd.concat([guides, grades], axis=1) if guides is not None else grades
        graph_sides = [side for side in sides if f'{side}_value' in graph]
        if guides is not None and not guides.empty and len(graph_sides) > 1:
            for panel, side in zip(st.columns(len(graph_sides)), graph_sides):
                with panel:
                    st.write('왼쪽' if side == 'left' else '오른쪽')
                    plot_timeseries(graph[[f'{side}_value']], visible_events.loc[visible_events.side == side],
                                    ctx['unit'], guides[[name for name in guides if name.startswith(side+'_')]])
        else:
            plot_timeseries(graph, visible_events, ctx['unit'], guides)
        if show_quality and ctx['counting_method'] != 'TARGET_POSITION':
            st.caption('품질 기준: 시작→목표 범위의 Perfect 70% · Good 50% · Normal 35%. 카운트 조건과는 별개입니다.')
        st.caption('threshold 출처: ' + ctx['threshold_source'])
    else:
        st.info('측정값 열이 없습니다. 관절 좌표 그래프에서 상세 좌표를 확인하세요.')

    with st.expander('측정 방식 · 핵심 관절'):
        st.write(ctx['description'])
        st.caption(f'{ctx["settings_source"]} · 핵심 관절: ' + ', '.join(JOINTS[j] for j in ctx['joints']))
        st.json(ctx['evaluation'])


def score_overview(events, ctx, info):
    if info.get('mode') == 'CALIBRATION' or ctx.get('counting_method') == 'TARGET_POSITION':
        return
    if events.empty or 'quality' not in events:
        return
    grades = events.quality.astype(str).str.upper().replace({'NORMAL': 'BAD'})
    if not grades.isin(QUALITY_COLORS).any():
        return
    for column, grade in zip(st.columns(3), QUALITY_COLORS):
        result = quality_result(dict(quality=grade))
        count = int(grades.eq(grade).sum())
        column.markdown(f"<div style='border-top:4px solid {result['color']};padding:8px'>"
                        f"<strong style='color:{result['color']};font-size:20px'>{result['label']}</strong>"
                        f"<br><b>{count}회 · +{count*result['points']:g}점</b>"
                        f"<br>score 코드 {result['code']} · 1회당 +{result['points']:g}점</div>",
                        unsafe_allow_html=True)


def tracking(table, log, ctx, time_column, bounds, sides):
    st.write('**추적 및 판정 상태**')
    segments = state_segments(table, time_column)
    if not segments.empty:
        segments = segments.loc[segments.side.isin(sides)]
        chart = alt.Chart(segments).mark_bar().encode(
            x=alt.X('start:Q', title='시간 (초)' if time_column == 'time_sec' else '행 번호'), x2='end:Q',
            y=alt.Y('side:N', title=None), color=alt.Color('state:N', title='상태'),
            tooltip=['side:N', 'state:N', alt.Tooltip('start:Q', format='.3f'), alt.Tooltip('end:Q', format='.3f')])
        st.altair_chart(chart.properties(height=90), width='stretch')
    report = diagnostics(table, ctx, time_column)
    if not report.empty:
        labels = ['왼쪽' if side == 'left' else '오른쪽' for side in sides]
        report = report.loc[report.side.isin(labels)].copy()
        report[['minimum', 'maximum', 'span', 'std', 'p05', 'p95']] *= ctx['factor']
        report = report.rename(columns={'side': '좌우', 'valid_pct': '유효 측정 (%)', 'minimum': '최솟값',
            'maximum': '최댓값', 'span': '변화 폭', 'std': '표준편차', 'main_state': '주요 상태'})
        compact = [name for name in ('좌우', '유효 측정 (%)', '최솟값', '최댓값', '변화 폭', '주요 상태') if name in report]
        st.dataframe(report[compact].round(3), hide_index=True)


def calibration_details(info, table):
    reference = info.get('calibration', {})
    if reference:
        label = '생성한' if info.get('mode') == 'CALIBRATION' else '사용한'
        st.caption(f"{label} 캘리브레이션: {reference['file']} · v{int(reference['version']):02d} · {reference['created_at']}")
        with st.expander('캘리브레이션 기준 · 출처'):
            st.code(reference.get('path', reference['file']), language=None)
            st.json(reference.get('thresholds', {}))
        return
    fields = ['calibration_file', 'calibration_version', 'calibration_created_at']
    if all(name in table for name in fields):
        records = table[fields].dropna(subset=['calibration_file']).drop_duplicates()
        records = records.loc[records.calibration_file.astype(str).str.strip().ne('')]
        if not records.empty:
            st.caption('사용한 캘리브레이션: ' + ' · '.join(records.calibration_file.astype(str)))
            with st.expander('캘리브레이션 기준 · 출처'):
                st.dataframe(records, hide_index=True)
            return
    st.caption('캘리브레이션 연결 정보가 없는 로그입니다.')


# ==============================================================================
# TAB 1: 📊 신호 및 관절 좌표 그래프
# ==============================================================================
def render_graph_tab(filtered, table, log, ctx, time_column, bounds, sides, markers, show_thresholds, show_quality, source_id):
    st.subheader('📊 시계열 신호 및 관절 좌표 분석')
    st.caption('측정값(Left/Right Value), 17개 COCO 관절 좌표, 또는 두 관절 간 상대 변위(Delta)를 선택하여 정규화(Norm) 여부와 함께 분석합니다.')

    data_type = st.radio(
        '그래프 데이터 종류 선택',
        ['운동 측정 신호 (Left / Right Value)', '개별 관절 좌표 (COCO 17 Joints)', '관절 간 상대 변위 및 길이 (Relative Delta)'],
        horizontal=True,
        key='graph_main_data_type'
    )

    if data_type == '운동 측정 신호 (Left / Right Value)':
        score_overview(markers, ctx, summary(log) if log is not None else {})
        overview(filtered, markers, markers, ctx, time_column, sides, show_thresholds, show_quality)
        with st.expander(f'카운트 기록 · 선택 구간 {len(markers)}건'):
            if len(markers):
                st.dataframe(markers, hide_index=True)
                stats = repetition_stats(table, markers, time_column)
                if not stats.empty:
                    stats = stats.loc[stats[time_column].between(*bounds) & stats.side.isin(sides)].copy()
                    stats[['minimum', 'maximum', 'span']] *= ctx['factor']
                    st.dataframe(stats, hide_index=True)
                    st.download_button('반복 통계 CSV', stats.to_csv(index=False).encode('utf-8-sig'), 'repetition_stats.csv', 'text/csv')
                st.download_button('카운트 시점 CSV', markers.to_csv(index=False).encode('utf-8-sig'), 'count_events.csv', 'text/csv')
            else:
                st.caption('선택한 구간에 카운트 기록이 없습니다.')
        tracking(filtered, log, ctx, time_column, bounds, sides)

    elif data_type == '개별 관절 좌표 (COCO 17 Joints)':
        if log is None or 'keypoints' not in log:
            st.info('개별 관절 좌표를 확인하려면 NPZ 로그가 필요합니다.')
            return

        c1, c2, c3 = st.columns([1.5, 1.2, 1.3])
        norm_mode = c1.selectbox(
            '좌표 정규화 모드',
            ['신체 정규화', '영상 비율', '픽셀'],
            index=0,
            help='신체 정규화(Norm): 골반 중심(0,0), 몸통길이=1.0, 위쪽=+Y (신체/거리 불변)\n영상 비율: 0~1 비율\n픽셀: 원본 영상 픽셀 (px)'
        )
        axis_choice = c2.selectbox('좌표 축 선택', ['Y축 (수직)', 'X축 (수평)', '둘 다 (X & Y)', '신뢰도 (Confidence)'], index=0)
        confidence = c3.slider('관절 최소 신뢰도', 0.0, 1.0, 0.35, 0.05, key='joint_plot_conf')

        default_joints = [JOINTS[j] for j in ctx.get('joints', []) if any(JOINTS[j].startswith(s) for s in sides)]
        if not default_joints:
            default_joints = [JOINTS[j] for j in ctx.get('joints', [])]
        if not default_joints:
            default_joints = ['right_knee', 'right_ankle']

        selected_joints = st.multiselect(
            '표시할 관절 선택',
            JOINTS,
            default=default_joints,
            format_func=lambda j: JOINT_DISPLAY_MAP.get(j, j),
            key='joint_plot_select'
        )

        if not selected_joints:
            st.warning('표시할 관절을 1개 이상 선택해 주세요.')
            return

        try:
            points = convert_coordinates(log, norm_mode, confidence)
            times = log['timestamps']
            mask = (times >= bounds[0]) & (times <= bounds[1])

            axis_map = {'X축 (수평)': [0], 'Y축 (수직)': [1], '둘 다 (X & Y)': [0, 1], '신뢰도 (Confidence)': [2]}
            target_axes = axis_map[axis_choice]
            axis_labels = {0: 'X', 1: 'Y', 2: 'Conf'}

            graph_dict = {}
            for j_name in selected_joints:
                idx = JOINTS.index(j_name)
                for ax in target_axes:
                    lbl = f'{j_name}_{axis_labels[ax]}'
                    graph_dict[lbl] = points[mask, idx, ax]

            graph_df = pd.DataFrame(graph_dict, index=times[mask])
            graph_df.index.name = time_column

            valid_count = int(graph_df.notna().any(axis=1).sum())
            total_count = len(graph_df)
            st.caption(f'📊 유효 좌표 시점: {valid_count:,} / {total_count:,} 프레임 | 모드: {norm_mode} | 축: {axis_choice}')

            if valid_count > 0:
                plot_timeseries(graph_df, markers, f'{axis_choice} ({norm_mode})', show_points=True)
                with st.expander('선택 관절 좌표 요약 통계'):
                    stats_df = graph_df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
                    stats_df['span'] = stats_df['max'] - stats_df['min']
                    st.dataframe(stats_df.round(4))
                st.download_button(
                    '선택 관절 좌표 시계열 CSV 다운로드',
                    graph_df.reset_index().to_csv(index=False).encode('utf-8-sig'),
                    'joint_coordinates.csv',
                    'text/csv'
                )
            else:
                st.warning('선택한 조건에서 유효한 관절 좌표가 없습니다. 신뢰도 임계값이나 확인 구간을 확인하세요.')
        except Exception as exc:
            st.error(f'관절 좌표 변환 실패: {exc}')

    elif data_type == '관절 간 상대 변위 및 길이 (Relative Delta)':
        if log is None or 'keypoints' not in log:
            st.info('관절 좌표를 확인하려면 NPZ 로그가 필요합니다.')
            return

        c1, c2, c3, c4 = st.columns(4)
        norm_mode = c1.selectbox('좌표 모드', ['신체 정규화', '영상 비율', '픽셀'], index=0, key='delta_norm_mode')
        ref_default = JOINTS.index('right_knee') if 'right_knee' in JOINTS else 14
        tgt_default = JOINTS.index('right_ankle') if 'right_ankle' in JOINTS else 16
        ref_joint = c2.selectbox('기준 관절 (Reference)', JOINTS, index=ref_default, format_func=lambda j: JOINT_DISPLAY_MAP.get(j, j), key='delta_ref')
        tgt_joint = c3.selectbox('대상 관절 (Target)', JOINTS, index=tgt_default, format_func=lambda j: JOINT_DISPLAY_MAP.get(j, j), key='delta_tgt')
        metric_choice = c4.selectbox(
            '계산 변위/거리',
            ['Y축 상대 변위 (Target Y - Ref Y)', 'X축 상대 변위 (Target X - Ref X)', '유클리디안 2D 거리 (Distance)'],
            index=0,
            key='delta_metric'
        )
        confidence = st.slider('관절 최소 신뢰도', 0.0, 1.0, 0.35, 0.05, key='delta_conf')

        try:
            points = convert_coordinates(log, norm_mode, confidence)
            times = log['timestamps']
            mask = (times >= bounds[0]) & (times <= bounds[1])

            ref_idx = JOINTS.index(ref_joint)
            tgt_idx = JOINTS.index(tgt_joint)

            ref_pts = points[mask, ref_idx, :2]
            tgt_pts = points[mask, tgt_idx, :2]

            col_label = f'{tgt_joint} - {ref_joint}'
            if metric_choice.startswith('Y축'):
                series_vals = tgt_pts[:, 1] - ref_pts[:, 1]
                val_unit = f'ΔY ({norm_mode})'
            elif metric_choice.startswith('X축'):
                series_vals = tgt_pts[:, 0] - ref_pts[:, 0]
                val_unit = f'ΔX ({norm_mode})'
            else:
                series_vals = np.sqrt(np.sum((tgt_pts - ref_pts) ** 2, axis=1))
                val_unit = f'거리 2D ({norm_mode})'

            delta_df = pd.DataFrame({col_label: series_vals}, index=times[mask])
            delta_df.index.name = time_column

            valid_count = int(delta_df.notna().sum().iloc[0])
            st.caption(f'📐 계산 대상: {col_label} ({metric_choice}) | 모드: {norm_mode} | 유효 시점: {valid_count:,}개')

            if valid_count > 0:
                plot_timeseries(delta_df, markers, val_unit, show_points=True)
                with st.expander('상대 변위 요약 통계'):
                    stats_df = delta_df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
                    stats_df['span'] = stats_df['max'] - stats_df['min']
                    st.dataframe(stats_df.round(4))
                st.download_button(
                    '상대 변위 CSV 다운로드',
                    delta_df.reset_index().to_csv(index=False).encode('utf-8-sig'),
                    'relative_delta.csv',
                    'text/csv'
                )
            else:
                st.warning('선택한 두 관절 모두가 검출된 유효 시점이 없습니다.')
        except Exception as exc:
            st.error(f'상대 변위 계산 실패: {exc}')


# ==============================================================================
# TAB 2: 🦴 스켈레톤 단독 모션 (검은 배경)
# ==============================================================================
def render_skeleton_motion_tab(log, source_id):
    st.subheader('🦴 스켈레톤 단독 모션 뷰어 (검은 배경)')
    st.caption('별도의 비디오/GIF 생성 대기 없이, 브라우저에서 실시간으로 즉시 재생되는 스켈레톤 애니메이션입니다. 재생/일시정지, 속도 조절, 탐색 슬라이더를 바로 이용할 수 있습니다.')

    if log is None or 'keypoints' not in log:
        st.warning('스켈레톤 모션을 렌더링하려면 관절 좌표가 포함된 NPZ 파일이 필요합니다.')
        return

    kpts = log['keypoints']
    total_frames = len(kpts)
    times = log.get('timestamps', np.arange(total_frames) / 30.0)

    c1, c2, c3, c4 = st.columns([1.3, 1.1, 1.2, 1.0])
    norm_choice = c1.radio(
        '스켈레톤 정규화 모드',
        ['신체 정규화 (Norm)', '원본 영상 좌표 (Unnorm)'],
        horizontal=True,
        key=f'skel_norm_{source_id}'
    )
    is_norm = (norm_choice == '신체 정규화 (Norm)')
    confidence = c2.slider('관절 최소 신뢰도', 0.0, 1.0, 0.35, 0.05, key=f'skel_conf_{source_id}')
    hide_face = c3.checkbox('얼굴 관절 숨김 (눈·코·귀)', value=True, key=f'skel_hide_face_{source_id}', help='얼굴 5개 관절(코, 눈, 귀)을 숨기고 어깨/몸통/팔/다리 골격만 표시합니다.')
    apply_smoothing = c4.checkbox('평활화 (Smoothing)', value=True, key=f'skel_smooth_{source_id}', help='고주파 지터와 일시적 가림을 보정하여 부드러운 궤적으로 만듭니다.')

    smooth_window = 5
    compare_raw = False
    if apply_smoothing:
        s_col1, s_col2 = st.columns([1.5, 2.5])
        smooth_window = s_col1.select_slider('평활화 강도 (Window Size)', options=[3, 5, 7, 9, 11], value=5, key=f'skel_win_{source_id}')
        compare_raw = s_col2.checkbox('하단 그래프에 원본(Raw) 곡선 함께 비교 표시 (점선)', value=True, key=f'skel_comp_raw_{source_id}')

    # 하단 연동 시계열 그래프 설정
    st.write('**📈 하단 연동 시계열 그래프 관절 선택** (그래프를 클릭/드래그하면 해당 시점의 스켈레톤으로 즉시 이동합니다)')
    col_g1, col_g2, col_g3 = st.columns([1.5, 1.3, 1.2])
    chart_preset = col_g1.selectbox(
        '그래프 분석 항목',
        [
            '무릎 신전 핵심 (발목 Y + 무릎 Y)',
            '무릎-발목 상대변위 (ΔY: 무릎 대비 발목 높이)',
            '사용자 관절 직접 선택 (Custom Joints)',
        ],
        key=f'skel_chart_preset_{source_id}'
    )

    raw_pts = coordinates(log, mode='신체 정규화' if is_norm else '영상 비율', confidence=confidence)
    pts = smooth_keypoints(raw_pts, window=smooth_window) if apply_smoothing else raw_pts

    chart_series = []
    chart_title = chart_preset + (" (평활화 적용)" if apply_smoothing else " (원본 Raw)")

    if chart_preset == '무릎 신전 핵심 (발목 Y + 무릎 Y)':
        colors = {'right_ankle': '#38bdf8', 'right_knee': '#fb923c', 'left_ankle': '#facc15', 'left_knee': '#10b981'}
        target_side = col_g2.radio('표시할 다리', ['오른쪽', '왼쪽', '양쪽'], horizontal=True, key=f'skel_chart_side_{source_id}')
        sides_to_plot = ['right'] if target_side == '오른쪽' else (['left'] if target_side == '왼쪽' else ['right', 'left'])

        for s in sides_to_plot:
            side_kr = '오른쪽' if s == 'right' else '왼쪽'
            ank_idx = JOINTS.index(f'{s}_ankle')
            knee_idx = JOINTS.index(f'{s}_knee')

            chart_series.append({
                'name': f'{side_kr} 발목 Y',
                'color': colors[f'{s}_ankle'],
                'values': pts[:, ank_idx, 1].tolist(),
                'dashed': False
            })
            if apply_smoothing and compare_raw:
                chart_series.append({
                    'name': f'{side_kr} 발목 Y (Raw)',
                    'color': '#64748b',
                    'values': raw_pts[:, ank_idx, 1].tolist(),
                    'dashed': True
                })

            chart_series.append({
                'name': f'{side_kr} 무릎 Y',
                'color': colors[f'{s}_knee'],
                'values': pts[:, knee_idx, 1].tolist(),
                'dashed': False
            })

    elif chart_preset == '무릎-발목 상대변위 (ΔY: 무릎 대비 발목 높이)':
        target_side = col_g2.radio('표시할 다리', ['오른쪽', '왼쪽', '양쪽'], horizontal=True, key=f'skel_delta_side_{source_id}')
        sides_to_plot = ['right'] if target_side == '오른쪽' else (['left'] if target_side == '왼쪽' else ['right', 'left'])

        for s in sides_to_plot:
            side_kr = '오른쪽' if s == 'right' else '왼쪽'
            ank_idx = JOINTS.index(f'{s}_ankle')
            knee_idx = JOINTS.index(f'{s}_knee')

            delta_vals = pts[:, ank_idx, 1] - pts[:, knee_idx, 1]
            color = '#38bdf8' if s == 'right' else '#facc15'
            chart_series.append({
                'name': f'{side_kr} 발목-무릎 ΔY',
                'color': color,
                'values': delta_vals.tolist(),
                'dashed': False
            })
            if apply_smoothing and compare_raw:
                raw_delta = raw_pts[:, ank_idx, 1] - raw_pts[:, knee_idx, 1]
                chart_series.append({
                    'name': f'{side_kr} ΔY (Raw)',
                    'color': '#64748b',
                    'values': raw_delta.tolist(),
                    'dashed': True
                })

    else:
        default_custom = ['right_ankle', 'right_knee']
        selected_custom = col_g2.multiselect(
            '표시할 관절',
            JOINTS,
            default=default_custom,
            format_func=lambda j: JOINT_DISPLAY_MAP.get(j, j),
            key=f'skel_custom_joints_{source_id}'
        )
        axis_choice = col_g3.selectbox('좌표 축', ['Y축 (수직)', 'X축 (수평)', '신뢰도 (Confidence)'], index=0, key=f'skel_custom_axis_{source_id}')
        axis_idx = {'Y축 (수직)': 1, 'X축 (수평)': 0, '신뢰도 (Confidence)': 2}[axis_choice]
        axis_label = {'Y축 (수직)': 'Y', 'X축 (수평)': 'X', '신뢰도 (Confidence)': 'Conf'}[axis_choice]

        palette = ['#38bdf8', '#fb923c', '#10b981', '#facc15', '#a855f7', '#ec4899', '#94a3b8']
        for i, j_name in enumerate(selected_custom):
            j_idx = JOINTS.index(j_name)
            chart_series.append({
                'name': f'{JOINT_DISPLAY_MAP.get(j_name, j_name).split()[0]} {axis_label}',
                'color': palette[i % len(palette)],
                'values': pts[:, j_idx, axis_idx].tolist(),
                'dashed': False
            })
            if apply_smoothing and compare_raw:
                chart_series.append({
                    'name': f'{JOINT_DISPLAY_MAP.get(j_name, j_name).split()[0]} (Raw)',
                    'color': '#64748b',
                    'values': raw_pts[:, j_idx, axis_idx].tolist(),
                    'dashed': True
                })

    if is_norm:
        st.info('💡 **신체 정규화 (Norm)**: 골반을 화면 중앙에 고정하고 신체 크기를 통일합니다. 피사체 이동이나 카메라 거리에 무관하게 순수 관절 각도/신전 움직임만 즉시 관찰할 수 있습니다.')
    else:
        st.info('💡 **원본 영상 좌표 (Unnorm)**: 카메라 화각 내 실제 위치와 크기 비율을 그대로 반영하여 검은 배경에 스켈레톤을 그립니다.')

    # 1) 즉각 렌더링되는 인터랙티브 HTML5 캔버스 플레이어 + 연동 그래프
    render_skeleton_player_component(
        log,
        is_norm=is_norm,
        confidence=confidence,
        apply_smoothing=apply_smoothing,
        smooth_window=smooth_window,
        hide_face=hide_face,
        chart_series=chart_series,
        chart_title=chart_title,
        height=780
    )

    # 2) 특정 프레임 세부 관절 좌표 확인
    with st.expander('🔍 특정 프레임 관절 좌표 수치표 보기 (정밀 검토)'):
        frame_idx = st.slider(
            '확인할 프레임 선택',
            min_value=0,
            max_value=total_frames - 1,
            value=0,
            format=f'프레임 %d / {total_frames}',
            key=f'skel_inspect_slider_{source_id}'
        )
        t_sec = float(times[frame_idx]) if frame_idx < len(times) else frame_idx / 30.0
        st.caption(f'프레임 #{frame_idx + 1} ({t_sec:.3f}초) 주요 관절 좌표')
        frame_kpt = kpts[frame_idx]
        key_indices = [11, 12, 13, 14, 15, 16, 5, 6]
        data_rows = []
        for kidx in key_indices:
            x, y, conf = frame_kpt[kidx]
            data_rows.append({
                '관절': JOINT_DISPLAY_MAP.get(JOINTS[kidx], JOINTS[kidx]),
                'X': round(float(x), 4),
                'Y': round(float(y), 4),
                '신뢰도': round(float(conf), 2)
            })
        st.dataframe(pd.DataFrame(data_rows), hide_index=True)

    # 3) 오프라인 비디오 파일 내보내기 (선택사항)
    with st.expander('💾 오프라인 MP4 비디오 파일로 내보내기 (선택사항)'):
        st.caption('브라우저 재생 외에 로컬 보관용 MP4 파일이 필요할 때만 인코딩합니다.')
        out_dir = ROOT / 'data' / 'outputs' / 'skeleton_videos'
        stem = Path(source_id).stem
        mode_tag = 'norm' if is_norm else 'unnorm'
        out_video_path = out_dir / f'{stem}_skeleton_{mode_tag}.mp4'

        if st.button('MP4 비디오 파일 렌더링', key=f'btn_offline_mp4_{source_id}_{mode_tag}'):
            prog = st.progress(0.0, text='MP4 인코딩 중...')
            try:
                render_skeleton_video(
                    log=log,
                    output_path=out_video_path,
                    is_norm=is_norm,
                    fps=30.0,
                    confidence=confidence,
                    on_progress=prog.progress
                )
                prog.progress(1.0, text='완료!')
                st.success('MP4 생성 완료!')
            except Exception as exc:
                st.error(f'인코딩 실패: {exc}')

        if out_video_path.is_file():
            st.download_button(
                '생성된 MP4 다운로드',
                out_video_path.read_bytes(),
                file_name=out_video_path.name,
                mime='video/mp4',
                key=f'dl_offline_{source_id}_{mode_tag}'
            )


# ==============================================================================
# TAB 3: 🎥 오버레이 비디오 생성
# ==============================================================================
def render_overlay_tab(log, table, files, source_id):
    st.subheader('🎥 원본 영상 스켈레톤 오버레이 합성')
    from analysis.video_ui import video_panel
    video_panel(log, table if '.csv' in files else None, source_id)


# ==============================================================================
# TAB 4: 🎯 카운트 시뮬레이션 실험
# ==============================================================================
def render_count_tab(table, log, ctx, time_column):
    st.subheader('🎯 카운트 판정 파라미터 시뮬레이션')
    count_simulation_panel(table, log, ctx, time_column)


def count_simulation_panel(table, log, ctx, time_column):
    panel_help('카운트 실험', '추출된 관절 신호(Relative Y, 각도 등)를 기반으로 FSM 카운트 파라미터를 조절하여 반복 횟수가 정상 판정되는지 시뮬레이션합니다.',
               '- **신호 곡선**: NPZ에 저장된 프레임별 실시간 측정값(좌/우)입니다.\n'
               '- **시작값(Start)**: 편안한 휴식 상태의 기준값입니다.\n'
               '- **목표값(Target)**: 신전 또는 굴곡 완료 시의 도달 목표값입니다.\n'
               '- **카운트 조건**: 40% 이상 가동 범위 도달 후 정점에서 5% 이상 하강(반전) 시 1회로 카운트됩니다.\n'
               '- **품질 등급**: 70% 이상 PERFECT(+1점), 50%~70% GOOD(+0.5점), 35%~50% NORMAL(+0.25점)')

    col_side, col_dir = st.columns(2)
    side = col_side.radio('실험할 다리/팔 (Side)', ['right', 'left'],
                          format_func=lambda s: {'right': '오른쪽 (Right)', 'left': '왼쪽 (Left)'}[s],
                          horizontal=True, key='sim_side')

    val_col = f'{side}_value'
    if val_col not in table:
        st.warning(f'데이터에 {side} 측정값이 없습니다.')
        return

    series = table[val_col].dropna()
    if series.empty:
        st.warning(f'{side} 측정값에 유효한 데이터가 없습니다.')
        return

    v_min, v_max = float(series.min()), float(series.max())
    cfg_direction = ctx.get('motion_direction', 'INCREASING')
    motion_dir = col_dir.selectbox('동작 방향 (Motion Direction)', ['INCREASING', 'DECREASING'],
                                   index=0 if cfg_direction == 'INCREASING' else 1, key='sim_direction')

    default_start = v_min if motion_dir == 'INCREASING' else v_max
    default_target = v_max if motion_dir == 'INCREASING' else v_min
    cfg_th = ctx.get('default_thresholds', {}).get(side, {})
    if 'start_val' in cfg_th and 'target_val' in cfg_th:
        default_start = float(cfg_th['start_val'])
        default_target = float(cfg_th['target_val'])

    p1, p2 = st.columns(2)
    start_val = p1.number_input('시작 기준값 (start_val)', value=round(default_start, 4), format='%.4f', key=f'sim_start_{side}')
    target_val = p2.number_input('목표 도달값 (target_val)', value=round(default_target, 4), format='%.4f', key=f'sim_target_{side}')

    with st.expander('세부 판정 비율 설정'):
        c1, c2, c3 = st.columns(3)
        act_ratio = c1.slider('동작 개시 비율 (Activation Ratio)', 0.10, 0.80, 0.40, 0.05, key=f'sim_act_{side}')
        rev_ratio = c2.slider('변곡점 확인 폭 (Reversal Ratio)', 0.01, 0.20, 0.05, 0.01, key=f'sim_rev_{side}')
        ret_ratio = c3.slider('복귀 판정 비율 (Return Ratio)', 0.05, 0.50, 0.30, 0.05, key=f'sim_ret_{side}')

    run_sim = st.button('🎯 카운트 시뮬레이션 실행', type='primary', use_container_width=True, key=f'run_sim_btn_{side}')

    state_key = f'sim_results_{side}'
    if run_sim:
        fsm = SideFSM(
            side_name=side,
            motion_direction=motion_dir,
            thresholds={'start_val': start_val, 'target_val': target_val},
            target_reps=1000,
            scoring=True,
            min_activation_ratio=act_ratio,
            return_ratio=ret_ratio
        )

        sim_events = []
        valid_df = table.dropna(subset=[val_col]).copy()

        for _, row in valid_df.iterrows():
            t = row[time_column]
            v = row[val_col]
            res = fsm.update(v, allow_count=True)
            if res['is_count_updated']:
                rep = fsm.completed_reps_history[-1]
                sim_events.append({
                    time_column: round(float(t), 3),
                    'rep_count': int(rep['rep_num']),
                    'peak_val': round(float(rep['max_val'] if motion_dir == 'INCREASING' else rep['min_val']), 4),
                    'achieved_rom': round(float(rep['achieved_rom']), 4),
                    'quality': rep['quality'],
                    'score_delta': float(rep['score_delta']),
                    'total_score': float(rep['total_score']),
                    'side': side
                })

        st.session_state[state_key] = {
            'events': sim_events,
            'rep_count': fsm.rep_count,
            'total_score': fsm.total_score,
            'start_val': start_val,
            'target_val': target_val,
        }

    res_data = st.session_state.get(state_key)
    if res_data:
        sim_events = res_data['events']
        rep_count = res_data['rep_count']
        total_score = res_data['total_score']

        st.write('---')
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('총 감지 횟수', f'{rep_count}회')
        perfs = sum(1 for e in sim_events if e['quality'] == 'PERFECT')
        goods = sum(1 for e in sim_events if e['quality'] == 'GOOD')
        normals = sum(1 for e in sim_events if e['quality'] == 'BAD')
        m2.metric('PERFECT (+1.0)', f'{perfs}회')
        m3.metric('GOOD (+0.5)', f'{goods}회')
        m4.metric('NORMAL (+0.25)', f'{normals}회')
        m5.metric('총 획득 점수', f'{total_score:.2f}점')

        valid_df = table.dropna(subset=[val_col]).copy()
        graph_df = valid_df[[time_column, val_col]].copy()
        events_df = pd.DataFrame(sim_events)

        chart_line = alt.Chart(graph_df).mark_line(color='#2563eb', size=2).encode(
            x=alt.X(f'{time_column}:Q', title='시간 (초)'),
            y=alt.Y(f'{val_col}:Q', title=f'{side} 신호값'),
            tooltip=[f'{time_column}:Q', alt.Tooltip(f'{val_col}:Q', format='.4f')]
        )

        rule_start = alt.Chart(pd.DataFrame({'y': [res_data['start_val']]})).mark_rule(
            strokeDash=[4, 4], color='#16a34a', size=1.5
        ).encode(y='y:Q')
        rule_target = alt.Chart(pd.DataFrame({'y': [res_data['target_val']]})).mark_rule(
            strokeDash=[4, 4], color='#9333ea', size=1.5
        ).encode(y='y:Q')

        layers = [chart_line, rule_start, rule_target]

        if not events_df.empty:
            rule_events = alt.Chart(events_df).mark_rule(
                strokeDash=[6, 3], size=2, opacity=0.8
            ).encode(
                x=f'{time_column}:Q',
                color=alt.Color('quality:N', scale=alt.Scale(
                    domain=['PERFECT', 'GOOD', 'BAD'],
                    range=['#16a34a', '#2563eb', '#ca8a04']
                ), legend=alt.Legend(title='판정 등급')),
                tooltip=[
                    alt.Tooltip(f'{time_column}:Q', title='카운트 시간(초)', format='.2f'),
                    alt.Tooltip('rep_count:Q', title='반복 번호'),
                    alt.Tooltip('quality:N', title='품질 등급'),
                    alt.Tooltip('peak_val:Q', title='정점 값', format='.4f'),
                    alt.Tooltip('achieved_rom:Q', title='달성 ROM', format='.4f'),
                ]
            )
            points_events = alt.Chart(events_df).mark_circle(size=80).encode(
                x=f'{time_column}:Q',
                y=alt.Y('peak_val:Q'),
                color=alt.Color('quality:N', legend=None),
                tooltip=[f'{time_column}:Q', 'rep_count:Q', 'quality:N', 'peak_val:Q']
            )
            layers.extend([rule_events, points_events])

        combined_chart = alt.layer(*layers).properties(
            title=f'카운트 시뮬레이션 결과 ({side}) | 초록점선: Start ({res_data["start_val"]}) | 보라점선: Target ({res_data["target_val"]})',
            height=340
        ).interactive()

        st.altair_chart(combined_chart, width='stretch')

        if not events_df.empty:
            st.caption('📋 감지된 반복 횟수 상세 내역:')
            st.dataframe(events_df, hide_index=True)
            st.download_button('카운트 시뮬레이션 결과 CSV 다운로드', events_df.to_csv(index=False).encode('utf-8-sig'),
                               f'{side}_count_simulation.csv', 'text/csv')
        else:
            st.warning('설정한 임계값 조건으로 감지된 반복 동작이 없습니다. start_val과 target_val 범위를 조절해 보세요.')


# ==============================================================================
# TAB 5: 📋 세션 상세 데이터
# ==============================================================================
def render_data_tab(filtered, table, time_column, markers, source_id, info):
    st.subheader('📋 세션 상세 데이터 및 요약')
    panel_help('상세 데이터', '원본 열과 수치를 확인하거나 필요한 시계열을 직접 선택해 분석·다운로드합니다.',
        '- **원본 표**: 선택 구간의 데이터프레임입니다.\n'
        '- **추가 시계열·통계**: 원하는 숫자 열을 선택하여 그래프를 그립니다.\n'
        '- **세션 요약**: 측정 동작·모드·저장 경로 등 전체 세션 메타데이터를 확인합니다.\n'
        '- **다운로드**: 선택 결과를 별도 CSV로 저장합니다.')

    st.dataframe(filtered, hide_index=True)

    with st.expander('추가 시계열 및 통계 분석'):
        numeric = [name for name in table.select_dtypes(include='number') if name != time_column]
        columns = st.multiselect('추가로 볼 열 선택', numeric)
        if columns:
            plot_timeseries(filtered.set_index(time_column)[columns], markers, '선택 열')
            st.dataframe(filtered[columns].describe().T.round(4))

    with st.expander('세션 요약 · 원본 경로'):
        st.code(source_id, language=None)
        if info:
            st.json(info)

    st.download_button(
        '선택 구간 전체 로그 CSV 다운로드',
        filtered.to_csv(index=False).encode('utf-8-sig'),
        f'{Path(source_id).stem}_log.csv',
        'text/csv'
    )


# ==============================================================================
# MAIN ANALYSIS ENTRY
# ==============================================================================
def show_analysis():
    try:
        files, source_id = select_files()
        if not files:
            return
        log = files.get('.npz')
        table, time_column = files['.csv'] if '.csv' in files else (measurements(log), 'time_sec')
        if 'video' in table and table['video'].nunique() > 1:
            video = st.selectbox('영상', table['video'].dropna().unique())
            table = table.loc[table['video'] == video].copy()
        table = table.sort_values(time_column)
        ctx = context(table, log)
        info = summary(log) if log is not None else {}

        st.subheader(ctx['name'])
        calibration_details(info, table)
        st.caption(f'{ctx["label"]} · {table[time_column].nunique():,}개 시점')

        first, last = float(table[time_column].min()), float(table[time_column].max())
        cards = st.columns(3)
        cards[0].metric('기록 구간', f'{last-first:.2f}초' if time_column == 'time_sec' else f'{last-first:g}행')
        for card, side, label in zip(cards[1:], ('left', 'right'), ('왼쪽 카운트', '오른쪽 카운트')):
            counts = pd.to_numeric(table.get(f'{side}_rep_count', pd.Series(dtype=float)), errors='coerce').dropna()
            card.metric(label, int(counts.iloc[-1]) if len(counts) else info.get(f'{side}_reps', '—'),
                        help='전체 로그의 마지막 누적 횟수입니다.')

        with st.expander('구간 및 필터 설정'):
            st.caption('선택한 시간 구간과 좌우가 아래 분석 탭에 적용됩니다.')
            bounds = st.slider('확인 구간 (초)' if time_column == 'time_sec' else '확인 구간 (행)',
                first, last, (first, last), key=f'analysis_range_{source_id}') if last > first else (first, last)
            side = st.radio('좌우 선택', ['양쪽', '왼쪽', '오른쪽'], horizontal=True, key='analysis_side')
            show_thresholds = st.checkbox('설정 threshold 표시', value=True)
            show_quality = st.checkbox('Perfect · Good · Normal 기준 표시', value=True,
                                       disabled=ctx['counting_method'] == 'TARGET_POSITION')
            show_events = st.checkbox('카운트 시점 표시', value=True, key='show_count_events')

        sides = ['left', 'right'] if side == '양쪽' else ['left' if side == '왼쪽' else 'right']
        filtered = table.loc[table[time_column].between(*bounds)]
        full_events = count_events(table, time_column)
        events = full_events.loc[full_events[time_column].between(*bounds) & full_events.side.isin(sides)]
        markers = events if show_events else events.iloc[:0]

        # 5 Revamped Tabs
        tab_graph, tab_skel_motion, tab_overlay, tab_count, tab_data = st.tabs([
            '📊 신호 및 관절 좌표 그래프',
            '🦴 스켈레톤 단독 모션 (검은 배경)',
            '🎥 오버레이 비디오 생성',
            '🎯 카운트 시뮬레이션 실험',
            '📋 세션 상세 데이터'
        ])

        with tab_graph:
            render_graph_tab(filtered, table, log, ctx, time_column, bounds, sides, markers, show_thresholds, show_quality, source_id)

        with tab_skel_motion:
            render_skeleton_motion_tab(log, source_id)

        with tab_overlay:
            render_overlay_tab(log, table, files, source_id)

        with tab_count:
            render_count_tab(table, log, ctx, time_column)

        with tab_data:
            render_data_tab(filtered, table, time_column, markers, source_id, info)

    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        st.error(f'로그를 읽을 수 없습니다: {exc}')

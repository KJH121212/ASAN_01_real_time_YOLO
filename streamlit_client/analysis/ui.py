"""Streamlit 로그 및 영상 분석 탭 (컴팩트 버전: 카운트 시뮬레이션 · 실전 캘리브레이션/본측정 · 스켈레톤 모션 · 그래프 · 오버레이 영상)."""

from datetime import date
import json
from pathlib import Path
from zipfile import BadZipFile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analysis.log_reader import JOINTS, NUM_KEYPOINTS, coordinates, count_events, measurements, read_csv, read_npz, summary
from analysis.metrics import context, quality_guides, state_segments, threshold_guides
from analysis.simulator import (
    CalibrationCycleDetector,
    MotionEngine,
    OneEuroKeypointFilter,
    SessionController,
    SideFSM,
    config_for_mode,
    find_saved_threshold,
    save_threshold_file,
)
from analysis.skeleton_player import render_skeleton_player_component, smooth_keypoints
from analysis.video_overlay import video_panel
from client.live_cards import QUALITY_COLORS

CLIENT_ROOT = Path(__file__).resolve().parent.parent
ROOT = CLIENT_ROOT.parent

JOINT_DISPLAY_MAP = {
    'nose': '코 (0)',
    'left_eye': '왼눈 (1)', 'right_eye': '오른눈 (2)',
    'left_ear': '왼귀 (3)', 'right_ear': '오른귀 (4)',
    'left_shoulder': '왼어깨 (5)', 'right_shoulder': '오른어깨 (6)',
    'left_elbow': '왼팔꿈치 (7)', 'right_elbow': '오른팔꿈치 (8)',
    'left_wrist': '왼손목 (9)', 'right_wrist': '오른손목 (10)',
    'left_hip': '왼골반 (11)', 'right_hip': '오른골반 (12)',
    'left_knee': '왼무릎 (13)', 'right_knee': '오른무릎 (14)',
    'left_ankle': '왼발목 (15)', 'right_ankle': '오른발목 (16)',
    'left_big_toe': '왼엄지발가락 (17)', 'right_big_toe': '오른엄지발가락 (20)',
    'left_small_toe': '왼새끼발가락 (18)', 'right_small_toe': '오른새끼발가락 (21)',
    'left_heel': '왼발뒤꿈치 (19)', 'right_heel': '오른발뒤꿈치 (22)',
}

SYMMETRIC_JOINT_PAIRS = {
    'big_toe': {'label': '엄지발가락 (Big Toe · L:17 / R:20)', 'left': 17, 'right': 20},
    'small_toe': {'label': '새끼발가락 (Small Toe · L:18 / R:21)', 'left': 18, 'right': 21},
    'heel': {'label': '발뒤꿈치 (Heel · L:19 / R:22)', 'left': 19, 'right': 22},
    'ankle': {'label': '발목 (Ankle · L:15 / R:16)', 'left': 15, 'right': 16},
    'knee': {'label': '무릎 (Knee · L:13 / R:14)', 'left': 13, 'right': 14},
    'hip': {'label': '동측 골반 (Hip · L:11 / R:12)', 'left': 11, 'right': 12},
    'opp_hip': {'label': '반대측 골반 (Opposite Hip · L:12 / R:11)', 'left': 12, 'right': 11},
    'wrist': {'label': '손목 (Wrist · L:9 / R:10)', 'left': 9, 'right': 10},
    'elbow': {'label': '팔꿈치 (Elbow · L:7 / R:8)', 'left': 7, 'right': 8},
    'shoulder': {'label': '동측 어깨 (Shoulder · L:5 / R:6)', 'left': 5, 'right': 6},
    'opp_shoulder': {'label': '반대측 어깨 (Opposite Shoulder · L:6 / R:5)', 'left': 6, 'right': 5},
    'nose': {'label': '코 (Nose · 0)', 'left': 0, 'right': 0},
}


@st.cache_data(show_spinner=False)
def load_file(name, content):
    return read_npz(content) if Path(name).suffix.lower() == '.npz' else read_csv(content)


@st.cache_data(show_spinner=False)
def convert_coordinates(log, mode, confidence):
    return coordinates(log, mode, confidence)


def _render_altair_chart(chart, height=320, interactive=True):
    """Altair 차트를 렌더링한다. Windows 정책으로 pyarrow DLL이 차단된 경우 HTML 폴백으로 표시한다."""
    sized = chart.properties(height=height)
    if interactive:
        sized = sized.interactive()
    try:
        st.altair_chart(sized, width='stretch')
    except ImportError:
        alt.data_transformers.disable_max_rows()
        html = sized.properties(width='container').to_html()
        components.html(html, height=height + 90, scrolling=False)


def _render_dataframe(df):
    """데이터프레임을 표로 렌더링한다. pyarrow DLL이 차단된 경우 HTML 테이블로 표시한다."""
    try:
        st.dataframe(df, hide_index=True)
    except ImportError:
        html_table = df.to_html(index=False, border=0)
        st.markdown(
            f'<div style="max-height:360px;overflow:auto;font-size:13px;">{html_table}</div>',
            unsafe_allow_html=True,
        )


def plot_timeseries(graph, events, title, guides=None, show_points=False, height=300):
    """시계열 측정값·관절 좌표 및 기준선·카운트 마커를 렌더링합니다."""
    time_column = graph.index.name
    values = graph.reset_index().melt(id_vars=time_column, var_name='series', value_name='value')
    color = alt.Color('series:N', title='항목')
    if set(graph.columns).issubset({'left_value', 'right_value'}):
        color = alt.Color('series:N', title='측정값', scale=alt.Scale(
            domain=['left_value', 'right_value'], range=['#2563eb', '#dc2626']))
    line = alt.Chart(values).mark_line(
        strokeWidth=2.0,
        point=alt.OverlayMarkDef(size=12, opacity=.6) if show_points else False,
    ).encode(
        x=alt.X(f'{time_column}:Q', title='시간 (초)' if time_column == 'time_sec' else '행 번호'),
        y=alt.Y('value:Q', title=title, scale=alt.Scale(zero=False)),
        color=color,
        tooltip=[alt.Tooltip(f'{time_column}:Q', format='.3f'), 'series:N', alt.Tooltip('value:Q', format='.4f')],
    )
    layers = [line]
    if guides is not None and not guides.empty:
        extra = guides.reset_index().melt(id_vars=time_column, var_name='series', value_name='value')
        categories = {
            'quality_PERFECT': ('Perfect (70%)', QUALITY_COLORS['PERFECT']),
            'quality_GOOD': ('Good (50%)', QUALITY_COLORS['GOOD']),
            'quality_BAD': ('Normal (35%)', QUALITY_COLORS['BAD']),
            'start_val': ('시작 기준', '#64748b'),
            'target_val': ('목표 기준', '#111827'),
        }
        extra['kind'] = extra.series.map(
            lambda name: next((lbl for sfx, (lbl, _) in categories.items() if name.endswith(sfx)), '기준선')
        )
        present = [(lbl, clr) for lbl, clr in categories.values() if lbl in set(extra.kind)]
        layers.append(alt.Chart(extra).mark_line(interpolate='step-after', strokeWidth=1.6, opacity=.85).encode(
            x=f'{time_column}:Q',
            y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
            detail='series:N',
            color=alt.Color(
                'kind:N', title='기준선',
                scale=alt.Scale(domain=[lbl for lbl, _ in present], range=[clr for _, clr in present]),
                legend=alt.Legend(orient='bottom', symbolStrokeWidth=3),
            ),
            tooltip=['kind:N', 'series:N', alt.Tooltip('value:Q', format='.4f')],
        ))
    if events is not None and not events.empty:
        layers.append(alt.Chart(events).mark_rule(strokeDash=[5, 3], opacity=.75).encode(
            x=f'{time_column}:Q',
            color=alt.Color('side:N', legend=None, scale=alt.Scale(domain=['left', 'right'], range=['#2563eb', '#dc2626'])),
            tooltip=[alt.Tooltip(f'{time_column}:Q', title='카운트 시간', format='.3f'), 'side:N', 'rep_count:Q'],
        ))
    chart = alt.layer(*layers).resolve_scale(color='independent')
    _render_altair_chart(chart, height=height)


# ==============================================================================
# COMPACT FILE & FILTER BAR
# ==============================================================================
def select_files():
    with st.expander('로그 폴더 열기 / 필터 설정', expanded=not st.session_state.get('analysis_loaded', False)):
        c_mode, c_path, c_btn = st.columns([1.0, 2.8, 0.7])
        method = c_mode.radio('입력 방식', ['경로 선택', '파일 업로드'], horizontal=True, key='log_method', label_visibility='collapsed')
        if method == '파일 업로드':
            uploaded = st.file_uploader('CSV/NPZ 파일 업로드', type=['csv', 'npz'], accept_multiple_files=True)
            sources = {f.name: f.getvalue() for f in uploaded}
        else:
            value = c_path.text_input('폴더/파일 경로', value=str(ROOT / 'data'), key='log_path', label_visibility='collapsed')
            if c_btn.button('경로 열기', key='log_open', use_container_width=True) or not st.session_state.get('log_paths'):
                path = Path(value.strip().strip('"')).expanduser()
                if not path.is_absolute():
                    path = ROOT / path
                if path.is_file() and path.suffix.lower() in ('.csv', '.npz'):
                    files = sorted({path, *(p for sfx in ('.csv', '.npz') if (p := path.with_suffix(sfx)).is_file())})
                elif path.is_dir():
                    files = sorted(p for p in path.rglob('*') if p.suffix.lower() in ('.csv', '.npz') and p.is_file())
                else:
                    files = []
                st.session_state['log_paths'] = [str(p.resolve()) for p in files]
            sources = {name: None for name in st.session_state.get('log_paths', [])}

        if not sources:
            st.info('분석할 경로를 열거나 CSV/NPZ 파일을 업로드하세요.')
            return {}, ''

        groups = {}
        for name in sources:
            groups.setdefault(str(Path(name).with_suffix('')), []).append(name)

        identities = {}
        for src in groups:
            parts = Path(src).parts
            ident = dict(patient='미지정', date='미지정', exercise='미지정')
            if len(parts) >= 4:
                try:
                    date.fromisoformat(parts[-3])
                    ident = dict(patient=parts[-4], date=parts[-3], exercise=parts[-2])
                except ValueError:
                    pass
            identities[src] = ident

        config_p = CLIENT_ROOT / 'configs/exercise_kpt_config_coco.json'
        exercises = json.loads(config_p.read_text(encoding='utf-8'))['exercises'] if config_p.exists() else {}
        keys = ('patient', 'date', 'exercise')
        selected_filters = {}
        for col, k, lbl in zip(st.columns(3), keys, ('환자 필터', '날짜 필터', '동작 필터')):
            opts = sorted({item[k] for item in identities.values()}, reverse=(k == 'date'))
            s_key = f'log_filter_{k}'
            if s_key in st.session_state:
                st.session_state[s_key] = [v for v in st.session_state[s_key] if v in opts]
            selected_filters[k] = col.multiselect(
                lbl, opts, key=s_key, placeholder='전체',
                format_func=(lambda v: exercises.get(v, {}).get('display_name_kr', v)) if k == 'exercise' else str,
            )
        groups = {
            src: fls for src, fls in groups.items()
            if all(not selected_filters[k] or identities[src][k] in selected_filters[k] for k in keys)
        }

    if not groups:
        st.info('조건에 맞는 로그가 없습니다.')
        return {}, ''

    selected = st.selectbox(
        f'분석할 로그 ({len(groups):,}개)',
        list(groups),
        key='selected_log',
        format_func=lambda v: ' / '.join(Path(v).parts[-4:]),
    )
    result = {}
    for name in groups[selected]:
        try:
            content = sources[name] if sources[name] is not None else Path(name).read_bytes()
            result[Path(name).suffix.lower()] = load_file(name, content)
        except (OSError, ValueError, EOFError, BadZipFile) as exc:
            st.warning(f'{Path(name).name} 읽기 실패: {exc}')
    st.session_state['analysis_loaded'] = True
    return result, selected


# ==============================================================================
# TAB 1: 카운트 시뮬레이션
# ==============================================================================
def _describe_eval_config(eval_cfg: dict) -> str:
    mtype = eval_cfg.get('metric_type', 'COORDINATE')
    default_norm = False if mtype == 'RAW_COORDINATE' else True
    use_norm = eval_cfg.get('use_normalization', default_norm)
    norm_txt = 'Norm(골반=0,몸통=1,위=+Y)' if use_norm else 'Raw(영상비율 0~1,아래=+Y)'
    direction = eval_cfg.get('motion_direction', 'DECREASING' if mtype == 'RAW_COORDINATE' else 'INCREASING')

    if mtype in ('COORDINATE', 'RAW_COORDINATE'):
        axis = eval_cfg.get('axis', 'y').upper()
        tgt = eval_cfg.get('target_kpt', {})
        t_id = tgt.get('left')
        if t_id is None:
            pk = eval_cfg.get('primary_kpts', {}).get('left', [])
            t_id = pk[0] if isinstance(pk, list) and len(pk) == 1 else None
        t_l = JOINTS[t_id] if isinstance(t_id, int) and 0 <= t_id < len(JOINTS) else str(t_id)
        formula = f'{axis}좌표 위치 ({t_l})'
    elif mtype == 'ANGLE':
        pk = eval_cfg.get('primary_kpts', {}).get('left', [])
        names = '-'.join(JOINTS[i] for i in pk if isinstance(i, int) and 0 <= i < len(JOINTS))
        formula = f'3점 각도 ({names}°)'
    else:
        pk = eval_cfg.get('primary_kpts', {}).get('left', [])
        names = '-'.join(JOINTS[i] for i in pk if isinstance(i, int) and 0 <= i < len(JOINTS))
        formula = f'2점 길이 ({names})'

    return f"**{mtype}** · {formula} · `{norm_txt}` · `{direction}`"


def _compute_dynamic_simulation_table(log, base_table, sim_config, sim_mode, apply_smoothing=False, smooth_window=5):
    df = base_table.copy().reset_index(drop=True)
    if not log or 'keypoints' not in log:
        df['left_posture_ok'] = True
        df['right_posture_ok'] = True
        return df

    kpts = log['keypoints'].astype(float).copy()
    if apply_smoothing and smooth_window > 1:
        kpts = smooth_keypoints(kpts, window=smooth_window)

    aspects = log.get('image_aspect_ratio')
    frame_sizes = log.get('frame_sizes')

    engine = MotionEngine(config=sim_config, mode=sim_mode)
    engine.exercise_name = sim_config.get('exercise_name', 'custom')
    controller = SessionController(motion_engine=engine)

    n_frames = min(len(df), len(kpts))
    left_vals = np.full(len(df), np.nan, dtype=float)
    right_vals = np.full(len(df), np.nan, dtype=float)
    left_posture = np.ones(len(df), dtype=bool)
    right_posture = np.ones(len(df), dtype=bool)

    for idx in range(n_frames):
        mat = kpts[idx]
        aspect = float(aspects[idx]) if (aspects is not None and idx < len(aspects) and np.isfinite(aspects[idx]) and aspects[idx] > 0) else 1.0
        w_px, h_px = (int(frame_sizes[idx][0]), int(frame_sizes[idx][1])) if (frame_sizes is not None and idx < len(frame_sizes) and np.all(np.isfinite(frame_sizes[idx]))) else (1280, 720)

        raw_list = [
            {'id': j, 'x': float(mat[j, 0]), 'y': float(mat[j, 1]), 'score': float(mat[j, 2]) if np.isfinite(mat[j, 2]) else 0.0,
             'image_aspect_ratio': aspect, 'image_width': w_px, 'image_height': h_px}
            for j in range(min(NUM_KEYPOINTS, len(mat))) if np.isfinite(mat[j, 0]) and np.isfinite(mat[j, 1])
        ]
        if not raw_list:
            left_posture[idx] = right_posture[idx] = False
            continue
        measured = controller.measurement_points(raw_list)
        if not measured:
            left_posture[idx] = right_posture[idx] = False
            continue
        pts_map = {p['id']: p for p in measured}
        vl = engine._compute_side_value('left', pts_map)
        vr = engine._compute_side_value('right', pts_map)
        left_vals[idx] = np.nan if vl is None else float(vl)
        right_vals[idx] = np.nan if vr is None else float(vr)
        left_posture[idx] = bool(engine.count_posture_valid('left', pts_map))
        right_posture[idx] = bool(engine.count_posture_valid('right', pts_map))

    df['left_value'] = left_vals
    df['right_value'] = right_vals
    df['left_posture_ok'] = left_posture
    df['right_posture_ok'] = right_posture
    return df


def _estimate_smart_thresholds(series: pd.Series, motion_dir: str, target_fraction: float = 0.85):
    valid = series.dropna().to_numpy(dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) < 5:
        return (0.0, 1.0, 0.85) if motion_dir == 'INCREASING' else (1.0, 0.0, 0.15)
    base = float(np.percentile(valid, 10 if motion_dir == 'INCREASING' else 90))
    peak = float(np.percentile(valid, 95 if motion_dir == 'INCREASING' else 5))
    target = base + target_fraction * (peak - base)
    return round(base, 4), round(peak, 4), round(target, 4)


def render_count_tab(table, log, ctx, time_column, source_id=''):
    config_path = CLIENT_ROOT / 'configs/exercise_kpt_config_coco.json'
    exercises_cfg = json.loads(config_path.read_text(encoding='utf-8'))['exercises'] if config_path.exists() else {}
    ex_keys = list(exercises_cfg.keys())
    detected_ex = ctx.get('exercise') if ctx.get('exercise') in ex_keys else 'knee_extension'
    has_npz_kpts = bool(log and 'keypoints' in log)
    joint_keys = list(SYMMETRIC_JOINT_PAIRS.keys())

    with st.container(border=True):
        c_ex, c_preset, c_post, c_sm = st.columns([1.3, 1.8, 0.9, 0.9])
        if 'sim_global_ex' not in st.session_state:
            st.session_state['sim_global_ex'] = detected_ex if detected_ex in ex_keys else ex_keys[0]
        sel_ex = c_ex.selectbox(
            '운동 동작',
            ex_keys,
            format_func=lambda k: f"{exercises_cfg[k].get('display_name_kr', k)}",
            key='sim_global_ex',
        )
        preset_options = ['MAIN_OFFICIAL', 'CALIB_OFFICIAL', 'CUSTOM_FEATURE', 'LOG_SAVED'] if has_npz_kpts else ['LOG_SAVED']
        preset_labels = {
            'MAIN_OFFICIAL': '본 측정 방식 (MAIN)',
            'CALIB_OFFICIAL': '캘리브레이션 방식 (CALIB)',
            'CUSTOM_FEATURE': '4종 측정 방식 / 관절 직접 선택',
            'LOG_SAVED': '로그 저장값 그대로 사용',
        }
        sel_preset = c_preset.selectbox('측정 방식', preset_options, format_func=lambda k: preset_labels[k], key='sim_global_preset')

        base_ex_cfg = exercises_cfg.get(sel_ex, {})
        if sel_preset == 'MAIN_OFFICIAL':
            active_sim_mode = 'MAIN'
            active_sim_cfg = config_for_mode(base_ex_cfg, 'MAIN')
            active_eval_cfg = dict(active_sim_cfg.get('eval_config', {}))
            custom_sig = f"MAIN_{active_eval_cfg.get('metric_type', 'COORDINATE')}"
        elif sel_preset == 'CALIB_OFFICIAL':
            active_sim_mode = 'CALIBRATION'
            active_sim_cfg = config_for_mode(base_ex_cfg, 'CALIBRATION')
            active_eval_cfg = dict(active_sim_cfg.get('eval_config', {}))
            custom_sig = f"CALIB_{active_eval_cfg.get('metric_type', 'COORDINATE')}"
        elif sel_preset == 'CUSTOM_FEATURE':
            active_sim_mode = 'MAIN'
            custom_mtype = st.radio(
                '측정 방식 (4종)',
                ['COORDINATE', 'RAW_COORDINATE', 'ANGLE', 'LENGTH'],
                format_func=lambda m: {
                    'COORDINATE': '정규화 Y좌표 (COORDINATE)',
                    'RAW_COORDINATE': '비정규화 Y좌표 (RAW_COORDINATE)',
                    'ANGLE': '3점 각도 (ANGLE)',
                    'LENGTH': '2점 길이 (LENGTH)',
                }[m],
                horizontal=True,
                key='sim_custom_mtype',
            )
            if custom_mtype in ('COORDINATE', 'RAW_COORDINATE'):
                is_raw = (custom_mtype == 'RAW_COORDINATE')
                tgt_key = st.selectbox('대상 관절', joint_keys, index=joint_keys.index('ankle'),
                                       format_func=lambda k: SYMMETRIC_JOINT_PAIRS[k]['label'], key='sim_coord_target')
                tp = SYMMETRIC_JOINT_PAIRS[tgt_key]
                active_eval_cfg = {
                    'metric_type': custom_mtype, 'axis': 'y',
                    'target_kpt': {'left': tp['left'], 'right': tp['right']},
                    'primary_kpts': {'left': [tp['left']], 'right': [tp['right']]},
                    'motion_direction': 'DECREASING' if is_raw else 'INCREASING',
                    'use_normalization': not is_raw, 'counting_method': 'PEAK_REVERSAL', 'posture_constraints': [],
                }
                custom_sig = f"{custom_mtype}_{tgt_key}"
            elif custom_mtype == 'ANGLE':
                ac1, ac2, ac3 = st.columns(3)
                p1_k = ac1.selectbox('관절 1 (시작)', joint_keys, index=joint_keys.index('hip'), format_func=lambda k: SYMMETRIC_JOINT_PAIRS[k]['label'], key='sim_angle_p1')
                p2_k = ac2.selectbox('관절 2 (중심각)', joint_keys, index=joint_keys.index('knee'), format_func=lambda k: SYMMETRIC_JOINT_PAIRS[k]['label'], key='sim_angle_p2')
                p3_k = ac3.selectbox('관절 3 (끝)', joint_keys, index=joint_keys.index('ankle'), format_func=lambda k: SYMMETRIC_JOINT_PAIRS[k]['label'], key='sim_angle_p3')
                j1, j2, j3 = SYMMETRIC_JOINT_PAIRS[p1_k], SYMMETRIC_JOINT_PAIRS[p2_k], SYMMETRIC_JOINT_PAIRS[p3_k]
                active_eval_cfg = {
                    'metric_type': 'ANGLE',
                    'primary_kpts': {'left': [j1['left'], j2['left'], j3['left']], 'right': [j1['right'], j2['right'], j3['right']]},
                    'motion_direction': 'INCREASING', 'use_normalization': True, 'counting_method': 'PEAK_REVERSAL', 'posture_constraints': [],
                }
                custom_sig = f"ANGLE_{p1_k}_{p2_k}_{p3_k}"
            else:
                lc1, lc2 = st.columns(2)
                l1_k = lc1.selectbox('관절 1 (시작)', joint_keys, index=joint_keys.index('knee'), format_func=lambda k: SYMMETRIC_JOINT_PAIRS[k]['label'], key='sim_len_p1')
                l2_k = lc2.selectbox('관절 2 (끝)', joint_keys, index=joint_keys.index('ankle'), format_func=lambda k: SYMMETRIC_JOINT_PAIRS[k]['label'], key='sim_len_p2')
                j1, j2 = SYMMETRIC_JOINT_PAIRS[l1_k], SYMMETRIC_JOINT_PAIRS[l2_k]
                active_eval_cfg = {
                    'metric_type': 'LENGTH',
                    'primary_kpts': {'left': [j1['left'], j2['left']], 'right': [j1['right'], j2['right']]},
                    'motion_direction': 'DECREASING', 'use_normalization': True, 'counting_method': 'PEAK_REVERSAL', 'posture_constraints': [],
                }
                custom_sig = f"LEN_{l1_k}_{l2_k}"
            active_sim_cfg = {'exercise_name': sel_ex, 'eval_config': active_eval_cfg, 'default_thresholds': base_ex_cfg.get('default_thresholds', {})}
        else:
            active_sim_mode = 'MAIN'
            active_sim_cfg = dict(base_ex_cfg)
            active_eval_cfg = dict(active_sim_cfg.get('eval_config', {}))
            custom_sig = 'LOG_SAVED'

        c_post.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        use_posture = c_post.checkbox('자세 제약', value=bool(active_eval_cfg.get('posture_constraints')), disabled=(sel_preset == 'LOG_SAVED'), key='sim_global_posture')
        c_sm.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        use_smooth = c_sm.checkbox('평활화(5f)', value=False, disabled=(sel_preset == 'LOG_SAVED'), key='sim_global_smooth')

        sim_table = (
            _compute_dynamic_simulation_table(log, table, active_sim_cfg, active_sim_mode, apply_smoothing=use_smooth, smooth_window=5)
            if (sel_preset != 'LOG_SAVED' and has_npz_kpts)
            else table.assign(left_posture_ok=True, right_posture_ok=True).reset_index(drop=True)
        )

        r1, r2, r3, r4, r5, r6 = st.columns([1.0, 1.3, 1.0, 1.0, 1.0, 1.1])
        side_choice = r1.selectbox('측정 방향', ['right', 'left', 'both'], format_func=lambda s: {'right': '오른쪽 (R)', 'left': '왼쪽 (L)', 'both': '양쪽 (L+R)'}[s], key='sim_global_side')
        target_sides = ['left', 'right'] if side_choice == 'both' else [side_choice]
        primary_side = 'right' if side_choice == 'both' else side_choice

        algo_choice = r2.selectbox(
            '판정 알고리즘',
            ['PEAK_REVERSAL', 'CALIBRATION_CYCLE'],
            format_func=lambda a: '본측정 FSM (반전)' if a == 'PEAK_REVERSAL' else '캘리브레이션 (30% 상승)',
            key='sim_global_algo',
        )
        cfg_dir = active_eval_cfg.get('motion_direction', ctx.get('motion_direction', 'INCREASING'))
        motion_dir = r3.selectbox('신호 방향', ['INCREASING', 'DECREASING'], index=0 if cfg_dir == 'INCREASING' else 1, key=f'sim_global_dir_{sel_ex}_{custom_sig}')

        series_primary = sim_table[f'{primary_side}_value'].dropna() if f'{primary_side}_value' in sim_table else pd.Series(dtype=float)
        if series_primary.empty:
            st.warning('유효한 측정 신호가 없습니다.')
            return

        est_start, est_peak, est_target = _estimate_smart_thresholds(series_primary, motion_dir, 0.85)
        param_sig = f"{sel_ex}_{custom_sig}_{motion_dir}_{primary_side}"
        start_key, target_key = f"sim_start_{param_sig}", f"sim_target_{param_sig}"
        if start_key not in st.session_state or target_key not in st.session_state:
            st.session_state[start_key] = float(est_start)
            st.session_state[target_key] = float(est_target)

        def _on_autofit(s_k=start_key, t_k=target_key, s_v=float(est_start), t_v=float(est_target)):
            st.session_state[s_k] = s_v
            st.session_state[t_k] = t_v

        start_val = r4.number_input('Start (0%)', format='%.4f', step=0.01, key=start_key)
        target_val = r5.number_input('Target (100%)', format='%.4f', step=0.01, key=target_key)
        r6.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        r6.button('현재 영상 맞춤', use_container_width=True, key=f'btn_autofit_{param_sig}', on_click=_on_autofit)

        with st.expander(f"수식 요약: {_describe_eval_config(active_eval_cfg)} | 현재 영상 참고치: Baseline {est_start:.4f} · Peak {est_peak:.4f} · 85% Target {est_target:.4f}"):
            p_c1, p_c2, p_c3 = st.columns(3)
            act_ratio = p_c1.slider('동작 개시 비율 (Activation)', 0.10, 0.80, 0.30 if algo_choice == 'CALIBRATION_CYCLE' else 0.40, 0.05, key=f'sim_act_{algo_choice}')
            rev_ratio = p_c2.slider('변곡점 반전 폭 (Reversal)', 0.01, 0.20, 0.05, 0.01, disabled=(algo_choice != 'PEAK_REVERSAL'), key='sim_rev_global')
            ret_ratio = p_c3.slider('복귀 판정 비율 (Return)', 0.05, 0.50, 0.15 if algo_choice == 'CALIBRATION_CYCLE' else 0.30, 0.05, key=f'sim_ret_{algo_choice}')

    all_events, side_summaries, learned_main_thresholds = [], {}, {}
    for s in target_sides:
        s_col, p_col = f'{s}_value', f'{s}_posture_ok'
        if s_col not in sim_table:
            continue
        s_start, s_target = float(start_val), float(target_val)
        if algo_choice == 'PEAK_REVERSAL':
            fsm = SideFSM(s, motion_dir, {'start_val': s_start, 'target_val': s_target}, target_reps=1000, scoring=True,
                          min_activation_ratio=act_ratio, return_ratio=max(0.01, min(ret_ratio, act_ratio - 0.01)), reversal_ratio=rev_ratio)
            s_events = []
            for _, row in sim_table.iterrows():
                v = row[s_col]
                if pd.isna(v):
                    continue
                res = fsm.update(float(v), allow_count=bool(row.get(p_col, True)) if use_posture else True)
                if res['is_count_updated']:
                    rep = fsm.completed_reps_history[-1]
                    s_events.append({
                        time_column: round(float(row[time_column]), 3), 'side': s, 'rep_count': int(rep['rep_num']),
                        'peak_val': round(float(rep['max_val'] if motion_dir == 'INCREASING' else rep['min_val']), 4),
                        'achieved_rom': round(float(rep['achieved_rom']), 4), 'quality': rep['quality'],
                        'score_delta': float(rep['score_delta']), 'total_score': float(rep['total_score']),
                    })
            all_events.extend(s_events)
            side_summaries[s] = {'rep_count': fsm.rep_count, 'total_score': fsm.total_score, 'start_val': s_start, 'target_val': s_target, 'events': s_events}
        else:
            if (motion_dir == 'INCREASING' and s_target <= s_start) or (motion_dir == 'DECREASING' and s_target >= s_start):
                s_start, _, s_target = _estimate_smart_thresholds(sim_table[s_col], motion_dir, 0.85)
                if s_target == s_start:
                    s_target = s_start + (0.35 if motion_dir == 'INCREASING' else -0.35)
            detector = CalibrationCycleDetector(
                rest_seconds=float(base_ex_cfg.get('calibration_config', {}).get('rest_seconds', 0.5)),
                metric_type=active_eval_cfg.get('metric_type', 'COORDINATE'), direction=motion_dir,
                start_val=s_start, target_val=s_target, activation_fraction=act_ratio, rearm_fraction=max(0.01, min(ret_ratio, act_ratio - 0.01)),
            )
            s_events = []
            for _, row in sim_table.iterrows():
                t, v = float(row[time_column]), row[s_col]
                val_f = None if pd.isna(v) else float(v)
                if detector.update(val_f, t, allow_count=bool(row.get(p_col, True)) if use_posture else True):
                    s_events.append({
                        time_column: round(t, 3), 'side': s, 'rep_count': len(detector.peaks),
                        'peak_val': round(float(val_f), 4) if val_f is not None else np.nan,
                        'achieved_rom': round(float(detector.peaks[-1]), 4), 'quality': 'CALIBRATING', 'score_delta': 0.0, 'total_score': 0.0,
                    })
            all_events.extend(s_events)
            if detector.ready and detector.peaks:
                sign = -1.0 if motion_dir == 'DECREASING' else 1.0
                learned_main_thresholds[s] = {
                    'start_val': round(float(detector.baseline), 4),
                    'target_val': round(float(detector.baseline + sign * 0.85 * float(np.median(detector.peaks[-3:]))), 4),
                }
            side_summaries[s] = {'rep_count': len(detector.peaks), 'total_score': 0.0, 'start_val': s_start, 'target_val': s_target, 'events': s_events}

    p_info = side_summaries.get(primary_side, {'rep_count': 0, 'total_score': 0.0, 'start_val': start_val, 'target_val': target_val, 'events': []})
    ev_list = p_info['events']
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f'{primary_side.upper()} 감지 횟수', f"{p_info['rep_count']}회")
    m2.metric('PERFECT (+1.0)', f"{sum(1 for e in ev_list if e['quality'] == 'PERFECT')}회")
    m3.metric('GOOD (+0.5)', f"{sum(1 for e in ev_list if e['quality'] == 'GOOD')}회")
    m4.metric('NORMAL (+0.25)', f"{sum(1 for e in ev_list if e['quality'] == 'BAD')}회")
    m5.metric('총 점수', f"{p_info['total_score']:.2f}점")

    if primary_side in learned_main_thresholds:
        l_th = learned_main_thresholds[primary_side]
        def _apply_calib(s_k=start_key, t_k=target_key, s_v=float(l_th['start_val']), t_v=float(l_th['target_val'])):
            st.session_state[s_k] = s_v
            st.session_state[t_k] = t_v
        c_cal1, c_cal2 = st.columns([3, 1])
        c_cal1.info(f"산출된 캘리브레이션 임계값(85%): Start={l_th['start_val']:.4f} → Target={l_th['target_val']:.4f}")
        c_cal2.button('현재 임계값에 적용', key=f'btn_copy_calib_{param_sig}', use_container_width=True, on_click=_apply_calib)

    plot_cols = [f'{s}_value' for s in target_sides if f'{s}_value' in sim_table]
    melted = sim_table.dropna(subset=plot_cols, how='all')[[time_column] + plot_cols].melt(id_vars=time_column, var_name='series', value_name='value').dropna()
    melted['side'] = melted['series'].str.replace('_value', '', regex=False)

    s_v, t_v = p_info['start_val'], p_info['target_val']
    act_v = s_v + act_ratio * (t_v - s_v)
    thresholds_df = pd.DataFrame([
        {'y': s_v, 'label': f'Start (0%: {s_v:.3f})', 'color': '#16a34a'},
        {'y': act_v, 'label': f'Act ({int(act_ratio*100)}%: {act_v:.3f})', 'color': '#f59e0b'},
        {'y': t_v, 'label': f'Target (100%: {t_v:.3f})', 'color': '#9333ea'},
    ])
    layers = [
        alt.Chart(melted).mark_line(size=2.0).encode(
            x=alt.X(f'{time_column}:Q', title='시간 (초)'),
            y=alt.Y('value:Q', title='측정 신호값', scale=alt.Scale(zero=False)),
            color=alt.Color('side:N', title='방향', scale=alt.Scale(domain=['left', 'right'], range=['#2563eb', '#dc2626'])),
            tooltip=[alt.Tooltip(f'{time_column}:Q', format='.3f'), 'side:N', alt.Tooltip('value:Q', format='.4f')],
        ),
        alt.Chart(thresholds_df).mark_rule(strokeDash=[4, 4], size=1.5).encode(
            y='y:Q',
            color=alt.Color('label:N', title='기준선', scale=alt.Scale(domain=thresholds_df['label'].tolist(), range=thresholds_df['color'].tolist())),
            tooltip=['label:N', alt.Tooltip('y:Q', format='.4f')],
        ),
    ]
    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        layers.append(alt.Chart(events_df).mark_rule(strokeDash=[6, 3], size=2, opacity=0.85).encode(
            x=f'{time_column}:Q',
            color=alt.Color('quality:N', scale=alt.Scale(domain=['PERFECT', 'GOOD', 'BAD', 'CALIBRATING'], range=['#16a34a', '#2563eb', '#ca8a04', '#06b6d4'])),
            tooltip=[alt.Tooltip(f'{time_column}:Q', format='.2f'), 'side:N', 'rep_count:Q', 'quality:N', alt.Tooltip('peak_val:Q', format='.4f')],
        ))
        layers.append(alt.Chart(events_df).mark_circle(size=80).encode(
            x=f'{time_column}:Q', y='peak_val:Q', color=alt.Color('quality:N', legend=None),
        ))
    _render_altair_chart(alt.layer(*layers).resolve_scale(color='independent'), height=320)

    if not events_df.empty:
        with st.expander(f'감지된 카운트 상세 내역 ({len(events_df)}건)'):
            _render_dataframe(events_df)


# ==============================================================================
# TAB 2: 실전 캘리브레이션 및 본 측정 테스트 (NPZ 기반)
# ==============================================================================
def render_offline_pipeline_tab(log, ctx, source_id):
    """NPZ 포즈 데이터를 사용하여 실시간 세션과 동일한 캘리브레이션 및 본 측정(TEST) 파이프라인을 실행합니다."""
    if not log or 'keypoints' not in log:
        st.warning('실전 캘리브레이션 / 본 측정 테스트를 수행하려면 keypoints가 포함된 NPZ 파일이 필요합니다.')
        return

    kpts = log['keypoints'].astype(float).copy()
    times = log.get('timestamps')
    if times is None or len(times) != len(kpts):
        times = np.arange(len(kpts)) * (1.0 / 30.0)

    aspects = log.get('image_aspect_ratio')
    frame_sizes = log.get('frame_sizes')

    config_path = CLIENT_ROOT / 'configs/exercise_kpt_config_coco.json'
    exercises_cfg = json.loads(config_path.read_text(encoding='utf-8'))['exercises'] if config_path.exists() else {}
    ex_keys = list(exercises_cfg.keys())
    detected_ex = ctx.get('exercise') if ctx.get('exercise') in ex_keys else 'knee_extension'

    source_path = Path(source_id)
    patient_id = ctx.get('patient') or '0000'
    round_no = 1
    if 'round' in source_path.stem.lower():
        import re
        m = re.search(r'round(\d+)', source_path.stem.lower())
        if m:
            round_no = int(m.group(1))

    c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.0, 1.3])
    sel_exercise = c1.selectbox(
        '운동 동작',
        ex_keys,
        index=ex_keys.index(detected_ex) if detected_ex in ex_keys else 0,
        format_func=lambda k: exercises_cfg[k].get('display_name_kr', k),
        key='pipe_exercise',
    )
    test_mode = c2.radio('실행 모드', ['CALIBRATION (캘리브레이션)', 'MAIN (본 측정)'], key='pipe_test_mode')
    is_calib = test_mode.startswith('CALIBRATION')

    patient_val = c3.text_input('환자번호', value=str(patient_id).zfill(4), max_chars=4, key='pipe_patient')
    round_val = c4.number_input('라운드 번호', min_value=1, value=round_no, step=1, key='pipe_round')

    saved_thresholds = find_saved_threshold(ROOT / 'data', patient_val, sel_exercise, int(round_val))

    if is_calib:
        st.markdown("##### 캘리브레이션 테스트 (3회 안정 수행 및 임계값 산출)")
        st.caption("양측 발/팔의 안정 상태(Baseline)와 3회 가동 범위(ROM)를 추적하여 85% 맞춤형 임계값을 산출합니다.")
        ex_cfg = config_for_mode(exercises_cfg[sel_exercise], 'CALIBRATION')
        engine = MotionEngine(config=ex_cfg, mode='CALIBRATION')
        engine.exercise_name = sel_exercise
        controller = SessionController(motion_engine=engine)
        filter_engine = OneEuroKeypointFilter()

        run_btn = st.button('캘리브레이션 분석 실행', type='primary', key='btn_run_calib')
        if run_btn:
            with st.spinner('NPZ 프레임 분석 중...'):
                calib_history = []
                left_values, right_values = [], []
                timestamps = []
                for idx in range(len(kpts)):
                    t = float(times[idx])
                    mat = kpts[idx]
                    aspect = float(aspects[idx]) if (aspects is not None and idx < len(aspects) and np.isfinite(aspects[idx]) and aspects[idx] > 0) else 1.0
                    w_px, h_px = (int(frame_sizes[idx][0]), int(frame_sizes[idx][1])) if (frame_sizes is not None and idx < len(frame_sizes) and np.all(np.isfinite(frame_sizes[idx]))) else (1280, 720)

                    raw_list = [
                        {'id': j, 'x': float(mat[j, 0]), 'y': float(mat[j, 1]), 'score': float(mat[j, 2]) if np.isfinite(mat[j, 2]) else 0.0,
                         'image_aspect_ratio': aspect, 'image_width': w_px, 'image_height': h_px}
                        for j in range(min(NUM_KEYPOINTS, len(mat))) if np.isfinite(mat[j, 0]) and np.isfinite(mat[j, 1])
                    ]
                    filtered = filter_engine.update(raw_list, timestamp=t)
                    is_occ = controller.check_occlusion(filtered) if filtered else True
                    cal_res = controller.process_calibration_frame(filtered, is_occ, now=t)
                    timestamps.append(t)
                    left_values.append(controller.last_left_data.get('val'))
                    right_values.append(controller.last_right_data.get('val'))
                    calib_history.append(cal_res)

                st.session_state['pipe_calib_result'] = {
                    'controller': controller,
                    'timestamps': timestamps,
                    'left_values': left_values,
                    'right_values': right_values,
                    'last_status': calib_history[-1] if calib_history else {},
                }

        calib_res = st.session_state.get('pipe_calib_result')
        if calib_res:
            ctrl = calib_res['controller']
            rep_l = ctrl.calib_reps_left
            rep_r = ctrl.calib_reps_right
            final_step = ctrl.calib_step

            m1, m2, m3, m4 = st.columns(4)
            m1.metric('최종 상태', final_step)
            m2.metric('왼쪽 감지 횟수', f"{rep_l} / 3회")
            m3.metric('오른쪽 감지 횟수', f"{rep_r} / 3회")
            is_success = (final_step in ('FINISHED', 'COOLDOWN') or (rep_l >= 3 and rep_r >= 3))
            m4.metric('캘리브레이션 판정', '성공' if is_success else '미완료')

            if is_success:
                try:
                    ctrl.calib_step = 'FINISHED'
                    th_out = ctrl.compute_thresholds()
                    st.success(f"임계값 산출 성공 ({sel_exercise})")
                    st.json(th_out)

                    save_col1, save_col2 = st.columns([2, 1])
                    save_path_preview = ROOT / f"data/{patient_val}/{date.today().isoformat()}/{sel_exercise}/round{str(round_val).zfill(2)}_thresholds_v01.json"
                    save_col1.caption(f"저장 예정 위치: `{save_path_preview}`")
                    if save_col2.button('이 임계값을 정규 파일로 저장 (확정)', key='btn_save_calib_json'):
                        saved_p = save_threshold_file(ROOT / 'data', patient_val, sel_exercise, int(round_val), th_out)
                        st.success(f"정규 임계값 파일이 저장되었습니다: `{saved_p}`")
                except Exception as exc:
                    st.warning(f"임계값 생성 중 주의: {exc}")
            else:
                st.info("양쪽 모두 안정 상태에서 3회 이상 가동 범위 동작이 완료되어야 캘리브레이션이 확정됩니다.")

    else:
        st.markdown("##### 본 측정(MAIN) 카운트 및 점수 테스트")
        st.caption("저장된 캘리브레이션 임계값 또는 사용자 설정 임계값을 바탕으로 본 측정을 실행하고 카운트/품질을 평가합니다.")

        custom_th = saved_thresholds
        th_src_label = "저장된 캘리브레이션 파일 발견" if custom_th else "저장된 캘리브레이션 없음 (기본값 사용)"
        st.info(f"임계값 상태: {th_src_label}")

        use_default = False
        if not custom_th:
            use_default = st.checkbox('동작 기본 임계값(default_thresholds)으로 테스트 진행', value=True)
            if use_default:
                custom_th = exercises_cfg[sel_exercise].get('default_thresholds')

        run_main_btn = st.button('본 측정(MAIN) 분석 실행', type='primary', disabled=(not custom_th), key='btn_run_main')
        if run_main_btn:
            with st.spinner('NPZ 프레임 본 측정 분석 중...'):
                ex_cfg = config_for_mode(exercises_cfg[sel_exercise], 'MAIN')
                engine = MotionEngine(
                    config=ex_cfg,
                    custom_thresholds=custom_th,
                    mode='MAIN',
                    target_reps=1000,
                )
                engine.exercise_name = sel_exercise
                controller = SessionController(motion_engine=engine)
                filter_engine = OneEuroKeypointFilter()

                rep_events = []
                left_vals, right_vals = [], []
                timestamps = []
                for idx in range(len(kpts)):
                    t = float(times[idx])
                    mat = kpts[idx]
                    aspect = float(aspects[idx]) if (aspects is not None and idx < len(aspects) and np.isfinite(aspects[idx]) and aspects[idx] > 0) else 1.0
                    w_px, h_px = (int(frame_sizes[idx][0]), int(frame_sizes[idx][1])) if (frame_sizes is not None and idx < len(frame_sizes) and np.all(np.isfinite(frame_sizes[idx]))) else (1280, 720)

                    raw_list = [
                        {'id': j, 'x': float(mat[j, 0]), 'y': float(mat[j, 1]), 'score': float(mat[j, 2]) if np.isfinite(mat[j, 2]) else 0.0,
                         'image_aspect_ratio': aspect, 'image_width': w_px, 'image_height': h_px}
                        for j in range(min(NUM_KEYPOINTS, len(mat))) if np.isfinite(mat[j, 0]) and np.isfinite(mat[j, 1])
                    ]
                    filtered = filter_engine.update(raw_list, timestamp=t)
                    is_occ = controller.check_occlusion(filtered) if filtered else True
                    l_data, r_data, _ = controller.process_main_frame(filtered, is_occ, now=t)
                    timestamps.append(t)
                    left_vals.append(l_data.get('val'))
                    right_vals.append(r_data.get('val'))

                    for side, d in (('left', l_data), ('right', r_data)):
                        if d.get('is_count_updated'):
                            fsm = engine.fsm_left if side == 'left' else engine.fsm_right
                            rep = fsm.completed_reps_history[-1]
                            rep_events.append({
                                'time_sec': round(t, 3),
                                'side': side,
                                'rep_count': int(rep['rep_num']),
                                'quality': rep['quality'],
                                'achieved_rom': round(float(rep['achieved_rom']), 4),
                                'score_delta': float(rep['score_delta']),
                                'total_score': float(rep['total_score']),
                            })

                st.session_state['pipe_main_result'] = {
                    'engine': engine,
                    'events': rep_events,
                    'timestamps': timestamps,
                    'left_vals': left_vals,
                    'right_vals': right_vals,
                }

        main_res = st.session_state.get('pipe_main_result')
        if main_res:
            eng = main_res['engine']
            events = main_res['events']
            ev_df = pd.DataFrame(events)

            c_l1, c_l2, c_r1, c_r2 = st.columns(4)
            c_l1.metric('왼쪽(L) 카운트', f"{eng.fsm_left.rep_count}회")
            c_l2.metric('왼쪽 총 점수', f"{eng.fsm_left.total_score:.2f}점")
            c_r1.metric('오른쪽(R) 카운트', f"{eng.fsm_right.rep_count}회")
            c_r2.metric('오른쪽 총 점수', f"{eng.fsm_right.total_score:.2f}점")

            if not ev_df.empty:
                _render_dataframe(ev_df)
            else:
                st.info('본 측정 동안 감지된 카운트가 없습니다.')


# ==============================================================================
# TAB 3: 스켈레톤 모션 및 연동 그래프
# ==============================================================================
def render_skeleton_motion_tab(log, ctx, source_id):
    if log is None or 'keypoints' not in log:
        st.warning('스켈레톤 모션을 보려면 NPZ 파일이 필요합니다.')
        return

    c1, c2, c3, c4 = st.columns([1.2, 1.5, 0.8, 0.8])
    norm_choice = c1.radio('좌표계', ['신체 정규화 (Norm)', '원본 영상 (Raw)'], horizontal=True, key='skel_norm_global')
    is_norm = (norm_choice == '신체 정규화 (Norm)')

    default_joints = [JOINTS[j] for j in ctx.get('joints', []) if j < len(JOINTS)] or ['right_ankle', 'right_knee']
    selected_joints = c2.multiselect('연동 그래프 표시 관절 (Y축)', JOINTS, default=default_joints[:2],
                                     format_func=lambda j: JOINT_DISPLAY_MAP.get(j, j), key='skel_joints_global')
    c3.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
    apply_smoothing = c3.checkbox('평활화 (5f)', value=True, key='skel_smooth_global')
    c4.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
    hide_face = c4.checkbox('얼굴 숨김', value=True, key='skel_hide_face_global')

    raw_pts = convert_coordinates(log, '신체 정규화' if is_norm else '영상 비율', 0.35)
    pts = smooth_keypoints(raw_pts, window=5) if apply_smoothing else raw_pts

    palette = ['#38bdf8', '#fb923c', '#10b981', '#facc15', '#a855f7', '#ec4899']
    chart_series = []
    for i, j_name in enumerate(selected_joints):
        j_idx = JOINTS.index(j_name)
        chart_series.append({
            'name': f"{JOINT_DISPLAY_MAP.get(j_name, j_name)} Y",
            'color': palette[i % len(palette)],
            'values': pts[:, j_idx, 1].tolist(),
            'dashed': False,
        })

    render_skeleton_player_component(
        log, is_norm=is_norm, confidence=0.35, apply_smoothing=apply_smoothing,
        smooth_window=5, hide_face=hide_face, chart_series=chart_series,
        chart_title=f"관절 Y좌표 궤적 ({'Norm' if is_norm else 'Raw'}) - 그래프 클릭 시 해당 프레임 이동",
        height=720,
    )


# ==============================================================================
# TAB 4: 신호 및 관절 그래프
# ==============================================================================
def render_graph_tab(table, log, ctx, time_column, source_id):
    first, last = float(table[time_column].min()), float(table[time_column].max())
    c_type, c_side, c_range = st.columns([1.8, 0.9, 1.5])
    data_type = c_type.radio(
        '그래프 종류',
        ['운동 측정 신호 (Value)', '개별 관절 좌표 (23 Joints)', '2점 관절 길이/변위'],
        horizontal=True, key='graph_main_data_type',
    )
    side_sel = c_side.radio('좌우', ['양쪽', '오른쪽', '왼쪽'], horizontal=True, key='analysis_side')
    sides = ['left', 'right'] if side_sel == '양쪽' else (['right'] if side_sel == '오른쪽' else ['left'])
    bounds = c_range.slider('구간 (초)', first, last, (first, last), key=f'analysis_range_{source_id}') if last > first else (first, last)

    filtered = table.loc[table[time_column].between(*bounds)]
    full_events = count_events(table, time_column)
    markers = full_events.loc[full_events[time_column].between(*bounds) & full_events.side.isin(sides)]

    if data_type == '운동 측정 신호 (Value)':
        value_cols = [f'{s}_value' for s in sides if f'{s}_value' in filtered]
        if value_cols:
            graph = filtered.set_index(time_column)[value_cols] * ctx['factor']
            guides = pd.concat([threshold_guides(filtered, ctx, time_column, sides), quality_guides(filtered, ctx, time_column, sides)], axis=1)
            plot_timeseries(graph, markers, ctx['unit'], guides, height=320)
        else:
            st.info('측정값 열이 없습니다.')

        with st.expander(f'추적 상태 및 카운트 기록 ({len(markers)}건)'):
            segments = state_segments(filtered, time_column)
            if not segments.empty:
                segments = segments.loc[segments.side.isin(sides)]
                _render_altair_chart(
                    alt.Chart(segments).mark_bar().encode(
                        x='start:Q', x2='end:Q', y='side:N', color='state:N',
                    ),
                    height=80,
                    interactive=False,
                )
            if not markers.empty:
                _render_dataframe(markers)

    elif data_type == '개별 관절 좌표 (23 Joints)':
        if log is None or 'keypoints' not in log:
            st.info('NPZ 파일이 필요합니다.')
            return
        k1, k2, k3 = st.columns([1.2, 1.0, 2.0])
        norm_mode = k1.selectbox('좌표계', ['신체 정규화', '영상 비율', '픽셀'], index=0, key='joint_norm_mode')
        axis_choice = k2.selectbox('축', ['Y축', 'X축', '둘 다 (X&Y)'], index=0, key='joint_axis_choice')
        default_j = [JOINTS[j] for j in ctx.get('joints', []) if j < len(JOINTS)] or ['right_ankle', 'right_knee']
        sel_joints = k3.multiselect('관절 선택', JOINTS, default=default_j, format_func=lambda j: JOINT_DISPLAY_MAP.get(j, j), key='joint_plot_select')
        if sel_joints:
            points = convert_coordinates(log, norm_mode, 0.35)
            times = log['timestamps']
            mask = (times >= bounds[0]) & (times <= bounds[1])
            axes = [1] if axis_choice == 'Y축' else ([0] if axis_choice == 'X축' else [0, 1])
            ax_lbl = {0: 'X', 1: 'Y'}
            graph_df = pd.DataFrame(
                {f'{j}_{ax_lbl[a]}': points[mask, JOINTS.index(j), a] for j in sel_joints for a in axes},
                index=pd.Index(times[mask], name=time_column),
            )
            plot_timeseries(graph_df, markers, f'{axis_choice} ({norm_mode})', show_points=True, height=320)

    else:
        if log is None or 'keypoints' not in log:
            st.info('NPZ 파일이 필요합니다.')
            return
        d1, d2, d3, d4 = st.columns(4)
        norm_mode = d1.selectbox('좌표계', ['신체 정규화', '영상 비율', '픽셀'], index=0, key='delta_norm_mode')
        ref_j = d2.selectbox('관절 1 (기준)', JOINTS, index=JOINTS.index('right_knee'), format_func=lambda j: JOINT_DISPLAY_MAP.get(j, j), key='delta_ref')
        tgt_j = d3.selectbox('관절 2 (대상)', JOINTS, index=JOINTS.index('right_ankle'), format_func=lambda j: JOINT_DISPLAY_MAP.get(j, j), key='delta_tgt')
        m_choice = d4.selectbox('측정 항목', ['2D 유클리드 길이 (Distance)', 'Y축 차이 (Target Y - Ref Y)', 'X축 차이 (Target X - Ref X)'], key='delta_metric')

        points = convert_coordinates(log, norm_mode, 0.35)
        times = log['timestamps']
        mask = (times >= bounds[0]) & (times <= bounds[1])
        p_ref = points[mask, JOINTS.index(ref_j), :2]
        p_tgt = points[mask, JOINTS.index(tgt_j), :2]
        if m_choice.startswith('2D'):
            vals = np.sqrt(np.sum((p_tgt - p_ref) ** 2, axis=1))
        elif m_choice.startswith('Y축'):
            vals = p_tgt[:, 1] - p_ref[:, 1]
        else:
            vals = p_tgt[:, 0] - p_ref[:, 0]
        delta_df = pd.DataFrame({f'{tgt_j}-{ref_j}': vals}, index=pd.Index(times[mask], name=time_column))
        plot_timeseries(delta_df, markers, f'{m_choice} ({norm_mode})', show_points=True, height=320)


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
        table = table.sort_values(time_column)
        ctx = context(table, log)
        info = summary(log) if log is not None else {}

        first, last = float(table[time_column].min()), float(table[time_column].max())
        l_cnt = pd.to_numeric(table.get('left_rep_count', pd.Series(dtype=float)), errors='coerce').dropna()
        r_cnt = pd.to_numeric(table.get('right_rep_count', pd.Series(dtype=float)), errors='coerce').dropna()
        l_reps = int(l_cnt.iloc[-1]) if len(l_cnt) else info.get('left_reps', 0)
        r_reps = int(r_cnt.iloc[-1]) if len(r_cnt) else info.get('right_reps', 0)

        st.caption(
            f"**{ctx['name']}** · `{ctx['metric']}` ({ctx['label']}) · "
            f"기록 구간 **{last - first:.1f}초** ({len(table):,} 프레임) · "
            f"기록 카운트 **L {l_reps}회 / R {r_reps}회**"
        )

        tab_count, tab_pipe, tab_skel, tab_graph, tab_overlay = st.tabs([
            '카운트 시뮬레이션',
            '실전 캘리브레이션/본측정 테스트',
            '스켈레톤 모션 및 연동 그래프',
            '신호 및 관절 그래프',
            '오버레이 비디오 생성',
        ])

        with tab_count:
            render_count_tab(table, log, ctx, time_column, source_id)

        with tab_pipe:
            render_offline_pipeline_tab(log, ctx, source_id)

        with tab_skel:
            render_skeleton_motion_tab(log, ctx, source_id)

        with tab_graph:
            render_graph_tab(table, log, ctx, time_column, source_id)

        with tab_overlay:
            video_panel(log, table if '.csv' in files else None, source_id)

    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        st.error(f'로그를 읽을 수 없습니다: {exc}')

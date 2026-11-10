"""Streamlit controls for exporting NPZ measurements on MP4.

Automatically associates the video corresponding to the selected analysis log
without requiring manual path input or re-uploading.
"""

from pathlib import Path
import os
import re
import tempfile
from typing import Optional

import cv2
import streamlit as st

from analysis.log_reader import summary
from analysis.video_overlay import export_video


def find_associated_video(source_id: str, log: dict = None, table = None) -> Optional[Path]:
    """Automatically find the video file corresponding to the selected log."""
    ROOT = Path(__file__).resolve().parent.parent
    source_path = Path(source_id)

    # 1. Direct match with same stem in same folder or subfolder
    for ext in ('.mp4', '.MP4', '.avi', '.mov', '.mkv'):
        cand = source_path.with_suffix(ext)
        if cand.is_file():
            return cand

    # 2. Check summary_json inside the NPZ log
    if log is not None:
        try:
            info = summary(log)
            for k in ('source_video', 'video_mp4', 'video_path', 'video'):
                v = info.get(k)
                if v and Path(v).is_file():
                    return Path(v)
        except Exception:
            pass

    # 3. Check CSV table if column exists
    if table is not None:
        for col in ('video', 'video_path', 'source_video'):
            if col in table and len(table[col].dropna()):
                v = str(table[col].dropna().iloc[0])
                if Path(v).is_file():
                    return Path(v)

    # 4. Check Streamlit session_state for recently extracted/used video
    for key in ('last_extracted_video', 'offline_selected_video'):
        v = st.session_state.get(key)
        if v and Path(v).is_file():
            # If log has same stem or matches frame count, use it
            return Path(v)

    # 5. Search in data/test_video/ matching exercise, mode, and set number
    log_name = source_path.name.lower()
    exercise = None
    for part in source_path.parts:
        if part in ('knee_extension', 'biceps_curl', 'shoulder_press', 'slr', 'clamshell', 'hip_knee_flexion'):
            exercise = part
            break

    search_dirs = []
    if exercise:
        search_dirs.extend((ROOT / 'data' / 'test_video').rglob(f'*{exercise}*'))
    search_dirs.append(ROOT / 'data' / 'test_video')
    search_dirs.append(source_path.parent)

    total_frames = len(log['keypoints']) if log is not None and 'keypoints' in log else None

    cand_files = []
    for d in search_dirs:
        if d.is_dir():
            for f in d.glob('*.mp4'):
                cand_files.append(f)
            for f in d.glob('*.MP4'):
                cand_files.append(f)

    # Remove duplicates
    unique_cands = []
    seen = set()
    for f in cand_files:
        p = f.resolve()
        if p not in seen and p.is_file():
            seen.add(p)
            unique_cands.append(p)

    is_calib = 'calib' in log_name
    m_set = re.search(r'set(\d+)', log_name)
    set_num = int(m_set.group(1)) if m_set else None

    scored = []
    for cand in unique_cands:
        score = 0
        cand_name = cand.name.lower()
        if is_calib and ('cali' in cand_name or 'calib' in cand_name):
            score += 10
        if not is_calib and 'cali' not in cand_name and 'calib' not in cand_name:
            score += 5
        if set_num is not None:
            if f'test{set_num}' in cand_name or f'set{set_num}' in cand_name or f'set{set_num:02d}' in cand_name:
                score += 20
        if exercise and exercise in str(cand).lower():
            score += 5
        scored.append((score, cand))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Frame count exact verification
    if total_frames:
        for score, cand in scored:
            if score > 0:
                try:
                    cap = cv2.VideoCapture(str(cand))
                    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if abs(cnt - total_frames) <= 2:
                        return cand
                except Exception:
                    pass

    if scored and scored[0][0] > 0:
        return scored[0][1]

    return None


def video_panel(log, table, source_id):
    st.caption('상단에서 선택한 로그(NPZ)와 원본 영상을 자동으로 연동하여 스켈레톤 및 실시간 측정값 오버레이 영상을 합성합니다.')
    if log is None:
        st.info('로그 열기에서 NPZ 파일을 선택하세요.')
        return

    # Automatically detect the video corresponding to the log selected above
    auto_video = find_associated_video(source_id, log, table)

    if auto_video and auto_video.is_file():
        st.success(f"🎬 **선택된 로그 연동 원본 영상**: `{auto_video.name}`")
        v_col1, v_col2 = st.columns([2.5, 1.5])
        v_col1.caption(f"📁 파일 경로: `{auto_video}`")

        try:
            cap = cv2.VideoCapture(str(auto_video))
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vfps = cap.get(cv2.CAP_PROP_FPS)
            vcnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            v_col2.caption(f"📐 {vw}x{vh} · {vfps:.1f} FPS · {vcnt} 프레임")
        except Exception:
            pass

        with st.expander("👁️ 원본 영상 미리보기"):
            st.video(str(auto_video))

        default_path = auto_video
    else:
        st.warning("⚠️ 선택된 로그와 일치하는 원본 영상을 자동으로 찾지 못했습니다. 아래 옵션에서 영상 경로를 지정해 주세요.")
        default_path = None

    # Optional override section
    with st.expander("⚙️ 다른 영상으로 변경 또는 수동 경로 / 업로드 (선택사항)"):
        method = st.radio(
            '영상 선택 모드',
            ['자동 연동 영상 사용', '직접 파일 경로 입력', '파일 업로드'],
            index=0 if default_path else 1,
            horizontal=True,
            key=f'overlay_method_{source_id}'
        )
        uploaded = None
        if method == '직접 파일 경로 입력':
            val = st.text_input('MP4 파일 경로', value=str(default_path) if default_path else '', key=f'override_path_{source_id}')
            target_path = Path(val.strip().strip('"')).expanduser() if val.strip() else None
        elif method == '파일 업로드':
            uploaded = st.file_uploader('합성할 MP4 파일', type=['mp4', 'mov', 'avi'], key=f'override_up_{source_id}')
            target_path = None
        else:
            target_path = default_path

        c_dir, c_off = st.columns([2, 1])
        root = Path(__file__).resolve().parent / 'outputs' / 'overlay_videos'
        directory = c_dir.text_input('결과 저장 폴더', value=str(root), key=f'overlay_dir_{source_id}')
        offset = c_off.number_input('로그 표시 지연 (초)', value=0., step=.1, help='양수면 로그 표시를 영상에서 더 늦게 시작합니다.')

    if method == '자동 연동 영상 사용' and target_path is None:
        target_path = default_path

    st.write("---")
    result_key = f'overlay_result_{source_id}'

    if st.button('🚀 오버레이 영상 합성 시작', type='primary', use_container_width=True, key=f'btn_overlay_run_{source_id}'):
        if target_path is None and uploaded is None or (target_path is not None and not target_path.is_file()):
            st.error('합성할 원본 영상 파일을 찾을 수 없습니다. 경로를 확인해 주세요.')
            return

        st.session_state.pop(result_key, None)
        upload_path = None
        try:
            output_dir = Path(directory.strip().strip('"')).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            name = target_path.stem if target_path is not None else Path(uploaded.name).stem
            output = output_dir / f'{name}_overlay.mp4'
            take = 2
            while output.exists():
                output = output_dir / f'{name}_overlay_{take:02d}.mp4'
                take += 1

            bar = st.progress(0., text='스켈레톤 및 실시간 측정값 오버레이 합성 중...')
            with st.spinner('오버레이 영상을 합성하는 중입니다. 잠시만 기다려 주세요...'):
                if uploaded is not None:
                    handle, tmp_name = tempfile.mkstemp(prefix='.overlay_upload_', suffix='.mp4', dir=output_dir)
                    os.close(handle)
                    upload_path = Path(tmp_name)
                    upload_path.write_bytes(uploaded.getbuffer())
                    target_path = upload_path

                result = export_video(target_path, log, output, table, offset, progress=bar.progress)
                bar.progress(1.0, text='합성 완료!')

            st.session_state[result_key] = result
        except Exception as exc:
            st.error(f'영상 합성 실패: {exc}')
        finally:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)

    result = st.session_state.get(result_key)
    if result and Path(result['path']).is_file():
        st.success(f'🎉 오버레이 영상 합성 완료! ({result["frames"]:,} 프레임)')
        st.code(result['path'], language=None)
        if not result['counts_available']:
            st.info('이 로그에는 반복 횟수 발생 시점이 없어 카운트는 --로 표시됩니다.')

        if result['browser_compatible']:
            st.video(result['path'])
        else:
            st.info('브라우저 호환 코덱 변환을 확인하세요.')

        with Path(result['path']).open('rb') as file:
            st.download_button(
                '📥 오버레이 MP4 영상 다운로드',
                file,
                file_name=Path(result['path']).name,
                mime='video/mp4',
                key=f'dl_overlay_{source_id}'
            )

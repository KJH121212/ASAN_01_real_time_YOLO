"""23개 관절(COCO-23: 몸통 17 + 발 6) 오버레이 MP4 영상 합성 및 Streamlit UI 통합 모듈."""

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from analysis.log_reader import NUM_KEYPOINTS, measurements, summary
from analysis.simulator import measurement_progress
from client.live_cards import QUALITY_COLORS, QUALITY_THRESHOLDS, quality_result

COCO23_BODY_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
COCO23_FOOT_EDGES = (
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
)


def find_ffmpeg():
    """PATH 또는 streamlit_client/configs/bin/ffmpeg.exe에서 ffmpeg 실행 파일을 찾는다."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    binaries = Path(__file__).resolve().parent.parent / "configs" / "bin"
    for path in sorted(binaries.glob("ffmpeg*")):
        if path.is_file() and (path.suffix.lower() == ".exe" if os.name == "nt" else os.access(path, os.X_OK)):
            return str(path)
    return None


def motion_progress(pose, side):
    data = pose.get(side, {})
    if pose.get("is_occluded") or data.get("val") is None:
        return None
    metric = pose.get("metric_type", "")
    display = pose.get("display_thresholds", {}).get(side)
    if display is not None:
        return measurement_progress(data["val"], "COORDINATE", display.get("start_val"), display.get("target_val"))
    start = pose.get("thresholds", {}).get(side, {}).get("start_val")
    target = pose.get("thresholds", {}).get(side, {}).get("target_val")
    if start is not None and target is not None:
        return measurement_progress(data["val"], metric, start, target)
    return measurement_progress(data.get("progress_ratio"), "COORDINATE", 0.0, 1.0)


def render_overlay(frame, keypoints, pose, exercise_name, elapsed_sec):
    """영상 프레임 위에 COCO-23 스켈레톤(몸통+발)과 오른쪽 정보 패널을 그린다."""
    image = frame.copy()
    height, width = image.shape[:2]
    unit = min(width / 640, height / 360)
    thickness = max(1, round(2 * unit))
    points = {}
    for p in keypoints or []:
        x, y = p.get("x"), p.get("y")
        if (
            p["id"] >= 5
            and p.get("score", 0) >= 0.35
            and x is not None
            and y is not None
            and math.isfinite(x)
            and math.isfinite(y)
            and 0 <= x <= 1
            and 0 <= y <= 1
        ):
            points[p["id"]] = (min(width - 1, round(x * width)), min(height - 1, round(y * height)))

    for first, second in COCO23_BODY_EDGES:
        if first in points and second in points:
            cv2.line(image, points[first], points[second], (70, 230, 100), thickness, cv2.LINE_AA)
    for first, second in COCO23_FOOT_EDGES:
        if first in points and second in points:
            cv2.line(image, points[first], points[second], (255, 140, 0), thickness, cv2.LINE_AA)
    for pid, point in points.items():
        pt_color = (255, 80, 255) if pid >= 17 else (0, 210, 255)
        cv2.circle(image, point, max(2, round(3 * unit)), pt_color, -1, cv2.LINE_AA)

    panel_width = min(width - 1, max(round(170 * unit), round(width * 0.27)))
    edge = width - panel_width
    margin = max(4, round(12 * unit))
    image[:, edge:] = (image[:, edge:] * 0.15).astype("uint8")

    def text(value, x, y, space, size=0.5, color=(255, 255, 255), bold=False):
        scale = max(0.15, size * unit)
        weight = max(1, round((2 if bold else 1) * unit))
        text_width = cv2.getTextSize(str(value), cv2.FONT_HERSHEY_SIMPLEX, scale, weight)[0][0]
        scale *= min(1.0, max(1, space) / max(1, text_width))
        cv2.putText(image, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)

    header = max(1, round(58 * unit))
    image[:header, :edge] = (image[:header, :edge] * 0.3).astype("uint8")
    text(f'{exercise_name} | {pose.get("mode", "")}', margin, round(22 * unit), edge - 2 * margin)
    text(f'{elapsed_sec:.1f}s  {pose.get("fps", 0):.1f} FPS', margin, round(45 * unit), edge - 2 * margin, 0.45)

    for index, side in enumerate(("left", "right")):
        data = pose.get(side, {})
        top = index * (height // 2)
        left, right = edge + margin, width - margin
        bar_width = max(8, round(18 * unit))
        bar_left = right - bar_width
        space = max(1, bar_left - left - margin)
        color = (255, 190, 80) if side == "left" else (100, 220, 255)
        text(f"{side.upper()} REPS", left, top + round(25 * unit), space, 0.6, color, True)
        count = data.get("rep_count")
        count_text = "--" if count is None else str(int(count))
        if pose.get("mode") == "CALIBRATION" and count is not None:
            count_text += " / 3"
        text(count_text, left, top + round(61 * unit), space, 1.4, color, True)
        result = quality_result(data, pose.get("mode", ""), pose.get("metric_type", ""))
        if result:
            hex_color = result["color"].lstrip("#")
            grade_color = tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))
            text(f"{result['label'].upper()} [{result['code']}] +{result['points']:g}", left, top + round(87 * unit), space, 0.55, grade_color, True)
        elif pose.get("mode") == "CALIBRATION":
            text("CALIBRATION", left, top + round(87 * unit), space, 0.4, color)
        else:
            text("RESULT --", left, top + round(87 * unit), space, 0.4)

        total = data.get("total_score")
        total_text = f"{total:g}" if total is not None and math.isfinite(total) else "--"
        text("TARGET 3" if pose.get("mode") == "CALIBRATION" else f"TOTAL {total_text}", left, top + round(110 * unit), space, 0.55, color, True)
        value = data.get("val")
        suffix = " px" if pose.get("metric_unit") == "px" else (" deg" if pose.get("metric_type") == "ANGLE" else "")
        digits = 4 if pose.get("metric_type") in ("COORDINATE", "RAW_COORDINATE") else 2
        value_text = "--" if value is None or not math.isfinite(value) else f"{value:.{digits}f}{suffix}"
        text(f"Value {value_text}", left, top + round(131 * unit), space, 0.42)

        progress = motion_progress(pose, side)
        bar_top = top + margin
        bar_bottom = top + height // 2 - margin
        cv2.rectangle(image, (bar_left, bar_top), (right, bar_bottom), (85, 85, 85), 1)
        if progress is not None:
            fill = round((bar_bottom - bar_top - 2) * progress)
            cv2.rectangle(image, (bar_left + 1, bar_bottom - 1 - fill), (right - 1, bar_bottom - 1), color, -1)

    cv2.line(image, (edge, 0), (edge, height), (90, 90, 90), 1)
    cv2.line(image, (edge, height // 2), (width, height // 2), (90, 90, 90), 1)
    return image


def export_video(video_path, log, output_path, table=None, offset=0.0, progress=None):
    """원본 영상에 NPZ 23관절 좌표와 측정값 오버레이를 합성해 H.264 MP4로 내보낸다."""
    video_path, output_path = Path(video_path).resolve(), Path(output_path).resolve()
    if output_path == video_path or output_path.exists():
        raise ValueError("원본 또는 기존 파일을 덮어쓸 수 없습니다. 새 출력 경로를 선택하세요.")
    times = np.asarray(log["timestamps"], dtype=float)
    info = summary(log)
    values = measurements(log)
    if table is not None and "time_sec" in table and len(table) == len(times):
        for column in table:
            if column != "time_sec":
                values[column] = table[column].to_numpy()
    stored = info.get("measurement", {})
    method = stored.get("measurement", {}).get("count_on", "")
    points = log.get("keypoints")
    if points is None or points.ndim != 3 or points.shape[0] != len(times):
        raise ValueError("NPZ에 프레임별 관절 좌표가 필요합니다.")

    capture = cv2.VideoCapture(str(video_path))
    writer = None
    intermediate = None
    try:
        if not capture.isOpened():
            raise ValueError("MP4 파일을 열 수 없습니다.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("영상 FPS를 확인할 수 없습니다.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.asarray(log.get("video_frame_idx", []))
        if len(indices) == len(times) and np.all(indices >= 0) and np.all(np.diff(indices) >= 0):
            video_times = indices / fps + offset
        else:
            video_times = times - float(info.get("video_time_origin_sec") or 0.0) + offset

        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".overlay_render_", suffix=".mp4", dir=output_path.parent)
        os.close(handle)
        intermediate = Path(name)
        frame_index = 0
        rows = values.to_dict("records")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] > 1920:
                scale_factor = 1920.0 / frame.shape[1]
                new_w = 1920
                new_h = (int(round(frame.shape[0] * scale_factor)) // 2) * 2
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(intermediate),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame.shape[1], frame.shape[0]),
                )
            t = frame_index / fps
            index = int(np.searchsorted(video_times, t, side="right") - 1)
            valid = index >= 0 and t - video_times[index] <= 1.0
            row = rows[max(index, 0)] if index >= 0 else {}
            pose = dict(
                mode=info.get("mode", ""),
                fps=fps,
                metric_type=row.get("metric_type", stored.get("metric_type", "")),
                measurement=stored.get("measurement", {}),
                metric_unit=row.get("metric_unit", stored.get("metric_unit", "")),
                is_occluded=not valid or str(row.get("is_occluded", "")).lower() == "true",
                thresholds={},
                display_thresholds=stored.get("display_thresholds", {}),
            )
            for side in ("left", "right"):
                data = {k[len(side) + 1:]: v for k, v in row.items() if k.startswith(side + "_") and pd.notna(v)}
                data["val"] = data.get("value") if valid else None
                data["counting_method"] = row.get("counting_method", method)
                pose[side] = data
                pose["thresholds"][side] = {k: data.get(k) for k in ("start_val", "target_val")}
                if not any(v is not None for v in pose["thresholds"][side].values()):
                    pose["thresholds"][side] = stored.get("thresholds", {}).get(side, {})
            if info.get("mode") == "MAIN":
                pose["display_thresholds"] = pose["thresholds"]
            keypoints = (
                [
                    dict(id=i, x=float(x), y=float(y), score=float(score))
                    for i, (x, y, score) in enumerate(points[index])
                ]
                if valid
                else []
            )
            image = render_overlay(
                frame,
                keypoints,
                pose,
                info.get("motion_name", str(log.get("exercise_name", ""))),
                float(times[index]) if index >= 0 else t,
            )
            writer.write(image)
            frame_index += 1
            if progress and (frame_index % max(1, round(fps)) == 0):
                progress(min(0.95, frame_index / max(total, frame_index) * 0.95))
        if writer is None or frame_index == 0:
            raise ValueError("영상에서 읽을 수 있는 프레임이 없습니다.")
        writer.release()
        writer = None
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            result = subprocess.run(
                [
                    ffmpeg, "-v", "error", "-i", str(intermediate),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags", "+faststart", "-an", str(output_path),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                output_path.unlink(missing_ok=True)
                raise ValueError(f"H.264 변환 실패: {result.stderr[-500:]}")
        else:
            intermediate.replace(output_path)
        if progress:
            progress(1.0)
        return dict(
            path=str(output_path),
            frames=frame_index,
            fps=fps,
            browser_compatible=bool(ffmpeg),
            counts_available=all(f"{side}_rep_count" in values for side in ("left", "right")),
        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if intermediate is not None:
            intermediate.unlink(missing_ok=True)


def find_associated_video(source_id: str, log: dict = None, table=None) -> Optional[Path]:
    """선택된 NPZ 로그와 짝이 맞는 원본 영상을 자동 탐색한다."""
    data_root = Path(__file__).resolve().parents[2] / "data"
    source_path = Path(source_id)

    for ext in (".mp4", ".MP4", ".avi", ".mov", ".mkv"):
        cand = source_path.with_suffix(ext)
        if cand.is_file():
            return cand

    if log is not None:
        try:
            info = summary(log)
            for k in ("source_video", "video_mp4", "video_path", "video"):
                v = info.get(k)
                if v and Path(v).is_file():
                    return Path(v)
        except Exception:
            pass

    if table is not None:
        for col in ("video", "video_path", "source_video"):
            if col in table and len(table[col].dropna()):
                v = str(table[col].dropna().iloc[0])
                if Path(v).is_file():
                    return Path(v)

    for key in ("last_extracted_video", "offline_selected_video"):
        v = st.session_state.get(key)
        if v and Path(v).is_file():
            return Path(v)

    log_name = source_path.name.lower()
    exercise = next(
        (
            p
            for p in source_path.parts
            if p in ("knee_extension", "biceps_curl", "shoulder_press", "slr", "clamshell", "hip_knee_flexion")
        ),
        None,
    )
    search_dirs = []
    if exercise and (data_root / "test_video").is_dir():
        search_dirs.extend((data_root / "test_video").rglob(f"*{exercise}*"))
    if (data_root / "test_video").is_dir():
        search_dirs.append(data_root / "test_video")
    search_dirs.append(source_path.parent)

    cand_files = []
    for d in search_dirs:
        if d.is_dir():
            cand_files.extend(d.glob("*.mp4"))
            cand_files.extend(d.glob("*.MP4"))

    unique_cands, seen = [], set()
    for f in cand_files:
        p = f.resolve()
        if p not in seen and p.is_file():
            seen.add(p)
            unique_cands.append(p)

    is_calib = "calib" in log_name
    m_set = re.search(r"set(\d+)", log_name)
    set_num = int(m_set.group(1)) if m_set else None

    scored = []
    for cand in unique_cands:
        score = 0
        cand_name = cand.name.lower()
        if is_calib and ("cali" in cand_name or "calib" in cand_name):
            score += 10
        if not is_calib and "cali" not in cand_name and "calib" not in cand_name:
            score += 5
        if set_num is not None and any(k in cand_name for k in (f"test{set_num}", f"set{set_num}", f"set{set_num:02d}")):
            score += 20
        if exercise and exercise in str(cand).lower():
            score += 5
        scored.append((score, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def video_panel(log, table, source_id):
    """오버레이 MP4 합성 UI 패널을 렌더링한다."""
    st.caption("선택한 로그(NPZ)와 원본 영상을 연동해 23개 관절 스켈레톤 및 측정값 오버레이 영상을 합성합니다.")
    if log is None:
        st.info("로그 열기에서 NPZ 파일을 선택하세요.")
        return

    auto_video = find_associated_video(source_id, log, table)
    if auto_video and auto_video.is_file():
        st.success(f"🎬 **선택된 로그 연동 원본 영상**: `{auto_video.name}`")
        with st.expander("👁️ 원본 영상 미리보기"):
            st.video(str(auto_video))
        default_path = auto_video
    else:
        st.warning("⚠️ 선택된 로그와 일치하는 원본 영상을 자동으로 찾지 못했습니다. 아래에서 경로를 지정해 주세요.")
        default_path = None

    with st.expander("⚙️ 다른 영상으로 변경 또는 수동 경로 / 업로드"):
        method = st.radio(
            "영상 선택 모드",
            ["자동 연동 영상 사용", "직접 파일 경로 입력", "파일 업로드"],
            index=0 if default_path else 1,
            horizontal=True,
            key=f"overlay_method_{source_id}",
        )
        uploaded = None
        if method == "직접 파일 경로 입력":
            val = st.text_input("MP4 파일 경로", value=str(default_path) if default_path else "", key=f"override_path_{source_id}")
            target_path = Path(val.strip().strip('"')).expanduser() if val.strip() else None
        elif method == "파일 업로드":
            uploaded = st.file_uploader("합성할 MP4 파일", type=["mp4", "mov", "avi"], key=f"override_up_{source_id}")
            target_path = None
        else:
            target_path = default_path

        c_dir, c_off = st.columns([2, 1])
        root = Path(__file__).resolve().parent / "outputs" / "overlay_videos"
        directory = c_dir.text_input("결과 저장 폴더", value=str(root), key=f"overlay_dir_{source_id}")
        offset = c_off.number_input("로그 표시 지연 (초)", value=0.0, step=0.1)

    if method == "자동 연동 영상 사용" and target_path is None:
        target_path = default_path

    result_key = f"overlay_result_{source_id}"
    if st.button("🚀 오버레이 영상 합성 시작", type="primary", use_container_width=True, key=f"btn_overlay_run_{source_id}"):
        if (target_path is None and uploaded is None) or (target_path is not None and not target_path.is_file()):
            st.error("합성할 원본 영상 파일을 찾을 수 없습니다.")
            return
        st.session_state.pop(result_key, None)
        upload_path = None
        try:
            output_dir = Path(directory.strip().strip('"')).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            name = target_path.stem if target_path is not None else Path(uploaded.name).stem
            output = output_dir / f"{name}_overlay.mp4"
            take = 2
            while output.exists():
                output = output_dir / f"{name}_overlay_{take:02d}.mp4"
                take += 1
            bar = st.progress(0.0, text="23관절 스켈레톤 및 측정값 오버레이 합성 중...")
            if uploaded is not None:
                handle, tmp_name = tempfile.mkstemp(prefix=".overlay_upload_", suffix=".mp4", dir=output_dir)
                os.close(handle)
                upload_path = Path(tmp_name)
                upload_path.write_bytes(uploaded.getbuffer())
                target_path = upload_path
            result = export_video(target_path, log, output, table, offset, progress=bar.progress)
            bar.progress(1.0, text="합성 완료!")
            st.session_state[result_key] = result
        except Exception as exc:
            st.error(f"영상 합성 실패: {exc}")
        finally:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)

    result = st.session_state.get(result_key)
    if result and Path(result["path"]).is_file():
        st.success(f"🎉 오버레이 영상 합성 완료! ({result['frames']:,} 프레임)")
        st.code(result["path"], language=None)
        if result["browser_compatible"]:
            st.video(result["path"])
        with Path(result["path"]).open("rb") as file:
            st.download_button(
                "📥 오버레이 MP4 영상 다운로드",
                file,
                file_name=Path(result["path"]).name,
                mime="video/mp4",
                key=f"dl_overlay_{source_id}",
            )

"""23개 관절(COCO-23) 및 기존 17개 관절 호환 CSV·NPZ 로그 읽기와 좌표 변환 모듈."""

from io import BytesIO
import json

import numpy as np
import pandas as pd

from client.live_cards import QUALITY_CODES, QUALITY_POINTS

JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]
NUM_KEYPOINTS = len(JOINTS)


class PoseNormalizer:
    """골반 중심 원점·몸통 길이=1.0 기준 등방성 2D 포즈 정규화 클래스 (독립 내장)."""

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(self, conf_threshold: float = 0.35, invert_y: bool = False):
        self.conf_threshold = conf_threshold
        self.invert_y = invert_y

    def normalize(self, keypoints: list) -> list:
        if not keypoints or len(keypoints) < 17:
            return None
        n_kpts = len(keypoints)
        kpt_matrix = np.array(
            [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in keypoints],
            dtype=np.float32,
        )
        coords = kpt_matrix[:, :2]
        scores = kpt_matrix[:, 2]
        aspect_ratio = float(keypoints[0].get("image_aspect_ratio", 1.0))
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            aspect_ratio = 1.0
        coords[:, 0] *= aspect_ratio

        has_hips = (
            scores[self.LEFT_HIP] >= self.conf_threshold
            and scores[self.RIGHT_HIP] >= self.conf_threshold
        )
        has_shoulders = (
            scores[self.LEFT_SHOULDER] >= self.conf_threshold
            and scores[self.RIGHT_SHOULDER] >= self.conf_threshold
        )
        if not has_hips:
            return None

        hip_center = (coords[self.LEFT_HIP] + coords[self.RIGHT_HIP]) / 2.0
        if has_shoulders:
            shoulder_center = (coords[self.LEFT_SHOULDER] + coords[self.RIGHT_SHOULDER]) / 2.0
            torso_length = float(np.linalg.norm(shoulder_center - hip_center))
        else:
            torso_length = float(np.linalg.norm(coords[self.LEFT_HIP] - coords[self.RIGHT_HIP])) * 1.5
        if torso_length < 1e-4:
            torso_length = 1.0

        norm_coords = (coords - hip_center) / torso_length
        if self.invert_y:
            norm_coords[:, 1] = -norm_coords[:, 1]

        return [
            {
                "id": i,
                "x": round(float(norm_coords[i, 0]), 4),
                "y": round(float(norm_coords[i, 1]), 4),
                "score": round(float(scores[i]), 3),
            }
            for i in range(n_kpts)
        ]


def count_events(table, time_column="time_sec"):
    """카운트 발생 프레임의 시각·횟수·측정값·품질을 표로 반환한다."""
    rows = []
    for side in ("left", "right"):
        flag, count = f"{side}_count_updated", f"{side}_rep_count"
        if flag in table:
            mask = table[flag].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
            source = "count_updated"
        elif count in table:
            mask = pd.to_numeric(table[count], errors="coerce").diff().gt(0)
            source = "count_increase"
        else:
            continue
        for _, row in table.loc[mask].iterrows():
            rows.append({
                time_column: row[time_column],
                "side": side,
                "rep_count": row.get(count, np.nan),
                "frame_idx": row.get("frame_idx", np.nan),
                "value": row.get(f"{side}_value", np.nan),
                "source": source,
                "quality": row.get(f"{side}_quality", ""),
                "score": row.get(
                    f"{side}_score",
                    QUALITY_CODES.get(str(row.get(f"{side}_quality", "")).upper(), np.nan),
                ),
                "score_delta": row.get(f"{side}_score_delta", np.nan),
                "total_score": row.get(f"{side}_total_score", np.nan),
            })
    return pd.DataFrame(
        rows,
        columns=[
            time_column, "side", "rep_count", "frame_idx", "value",
            "source", "quality", "score", "score_delta", "total_score",
        ],
    ).sort_values(time_column)


def read_csv(content):
    table = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    if table.empty:
        raise ValueError("CSV에 데이터가 없습니다.")
    for column, scale in (("time_sec", 1), ("time", 1), ("timestamp_ms", 0.001)):
        if column in table:
            times = pd.to_numeric(table[column], errors="coerce") * scale
            if not np.isfinite(times).all():
                raise ValueError(f"{column} 열에 유효하지 않은 시간이 있습니다.")
            table = table.copy()
            table["time_sec"] = times
            return table, "time_sec"
    table["row_index"] = np.arange(len(table))
    return table, "row_index"


def read_npz(content):
    """23개 관절(또는 기존 17개 관절) NPZ 파일을 읽어 23개 관절 배열로 통일한다."""
    with np.load(BytesIO(content), allow_pickle=False) as data:
        log = {key: data[key].copy() for key in data.files}
    times = np.asarray(log.get("timestamps", []), dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError("NPZ의 timestamps가 비어 있거나 올바르지 않습니다.")
    log["timestamps"] = times
    for key in ("keypoints", "keypoints_px"):
        if key in log:
            arr = log[key]
            if arr.ndim != 3 or arr.shape[0] != len(times) or arr.shape[1] not in (17, 23) or arr.shape[2] != 3:
                raise ValueError(f"{key}의 크기는 프레임 수 × 23(또는 17) × 3이어야 합니다.")
            if arr.shape[1] == 17:
                padded = np.full((len(times), NUM_KEYPOINTS, 3), np.nan, dtype=arr.dtype)
                padded[:, :, 2] = 0.0
                padded[:, :17, :] = arr
                log[key] = padded
    for key in ("values", "total_scores", "score_deltas", "rep_counts", "progress_ratios", "states"):
        if key in log and log[key].shape != (len(times), 2):
            raise ValueError(f"{key}의 크기는 프레임 수 × 2이어야 합니다.")
    return log


def summary(log):
    value = log.get("summary_json")
    info = json.loads(str(value.item())) if value is not None else {}
    if not isinstance(info, dict):
        raise ValueError("summary_json은 JSON 객체여야 합니다.")
    return info


def measurements(log):
    table = pd.DataFrame({"time_sec": log["timestamps"]})
    for key, suffix in (
        ("values", "value"),
        ("total_scores", "total_score"),
        ("score_deltas", "score_delta"),
        ("rep_counts", "rep_count"),
        ("progress_ratios", "progress_ratio"),
        ("states", "state"),
    ):
        if key in log:
            for i, side in enumerate(("left", "right")):
                table[f"{side}_{suffix}"] = log[key][:, i]
    raw_history = log.get("reps_json")
    history = json.loads(str(raw_history.item())) if raw_history is not None else []
    for side in ("left", "right"):
        grades = pd.Series(index=table.index, dtype=object)
        count = f"{side}_rep_count"
        if count in table:
            by_rep = {
                row.get("rep_num"): row.get("quality")
                for row in history
                if isinstance(row, dict) and row.get("side") == side
            }
            grades = table[count].map(by_rep)
        delta = f"{side}_score_delta"
        if delta in table:
            mapped = table[delta].map({points: grade for grade, points in QUALITY_POINTS.items()})
            inferred = mapped.astype("string").ffill().astype(object)
            grades = grades.astype("string").fillna(inferred.astype("string")).astype(object)
        table[f"{side}_quality"] = grades
    return table


def coordinates(log, mode="영상 비율", confidence=0.35):
    """관절 좌표를 영상 비율(0~1), 픽셀, 신체 정규화(골반=0, 몸통=1) 모드로 변환한다."""
    if "keypoints" in log:
        points = log["keypoints"].astype(float).copy()
    elif "keypoints_px" in log and "frame_sizes" in log:
        points = log["keypoints_px"].astype(float).copy()
        points[:, :, :2] /= log["frame_sizes"][:, None, :]
    else:
        raise ValueError("관절 좌표 또는 영상 크기 정보가 없습니다.")

    if mode == "픽셀":
        if "keypoints_px" in log:
            points = log["keypoints_px"].astype(float).copy()
        elif "frame_sizes" in log:
            points[:, :, :2] *= log["frame_sizes"][:, None, :]
        elif "video_width" in log and "video_height" in log:
            points[:, :, :2] *= [float(log["video_width"]), float(log["video_height"])]
        else:
            raise ValueError("픽셀 변환에 필요한 영상 크기가 없습니다.")
    elif mode == "신체 정규화":
        normalizer = PoseNormalizer(conf_threshold=confidence, invert_y=True)
        result = np.full_like(points, np.nan)
        result[:, :, 2] = points[:, :, 2]
        aspects = np.broadcast_to(log.get("image_aspect_ratio", 1.0), (len(points),))
        for i, frame in enumerate(points):
            if not np.isfinite(aspects[i]) or aspects[i] <= 0:
                continue
            if not np.isfinite(frame[[11, 12]]).all() or (frame[[11, 12], 2] < confidence).any():
                continue
            keypoints = [
                dict(
                    id=j,
                    x=float(x) if np.isfinite(x) else 0.0,
                    y=float(y) if np.isfinite(y) else 0.0,
                    score=float(score) if np.isfinite([x, y, score]).all() else 0.0,
                    image_aspect_ratio=float(aspects[i]),
                )
                for j, (x, y, score) in enumerate(frame)
            ]
            normalized = normalizer.normalize(keypoints)
            if normalized:
                result[i, :, :2] = [[p["x"], p["y"]] for p in normalized]
        points = result

    points[(~np.isfinite(points[:, :, 2])) | (points[:, :, 2] < confidence), :2] = np.nan
    return points

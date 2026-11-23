"""23개 관절(COCO-23) 로그의 측정 문맥, 핵심 관절, 기준선 및 진단 지표를 계산하는 독립 모듈."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.log_reader import NUM_KEYPOINTS, summary
from client.live_cards import QUALITY_THRESHOLDS

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "exercise_kpt_config_coco.json"

LABELS = {
    "ANGLE": ("관절 각도", "각도 (°)", "설정된 세 관절의 각도 변화를 봅니다."),
    "COORDINATE": ("정규화 관절 위치", "정규화 Y좌표 (무차원)", "골반 중심을 원점으로 하고 몸통 길이를 1로 맞춘 대상 관절의 Y좌표 위치입니다."),
    "RAW_COORDINATE": ("비정규화 관절 위치", "영상 높이 대비 Y좌표 (0~1)", "몸통 정규화를 적용하지 않은 원본 영상 비율(0~1) 기준 대상 관절의 Y좌표 위치입니다."),
    "LENGTH": ("관절 사이 길이", "정규화 길이 (무차원)", "몸통 길이를 1로 맞춘 좌표에서 계산한 두 관절 사이 유클리드 길이입니다."),
}


def context(table, log=None):
    """로그에 기록된 운동 종류, 측정 방식, 대상 관절(0~22), 기준값 이력을 정리한다."""
    configs = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["exercises"] if CONFIG_PATH.exists() else {}
    info = summary(log) if log is not None else {}
    stored = info.get("measurement", {})
    name = info.get("motion_name", stored.get("exercise_name", ""))
    if not name and "motion_name" in table:
        names = table["motion_name"].dropna()
        name = str(names.iloc[0]) if len(names) else ""
    if not name and log is not None:
        name = str(log.get("exercise_name", ""))
        if not name:
            source = str(log.get("source_video", ""))
            name = next((key for key in configs if key in source), "")

    config = configs.get(name, {})
    evaluation = dict(config.get("eval_config", {}))
    metric = stored.get("metric_type", evaluation.get("metric_type", "UNKNOWN"))
    if "metric_type" in table and table["metric_type"].notna().any():
        metric = str(table["metric_type"].dropna().iloc[0])
    recorded = stored.get("measurement", {})
    if recorded:
        evaluation = dict(recorded)
    for key in ("axis", "target_kpt", "primary_kpts", "coordinate_unit"):
        if key in recorded:
            evaluation[key] = recorded[key]
    if "movement_direction" in recorded:
        evaluation["motion_direction"] = recorded["movement_direction"]

    by_side = {}
    for side in ("left", "right"):
        fields = ("primary_kpts",) if metric in ("ANGLE", "LENGTH") else ("target_kpt", "primary_kpts")
        ids = []
        for field in fields:
            value = evaluation.get(field, {}).get(side, [])
            ids.extend(value if isinstance(value, list) else [value])
        by_side[side] = sorted({j for j in ids if type(j) is int and 0 <= j < NUM_KEYPOINTS})
    joints = sorted(set(by_side["left"] + by_side["right"])) or [13, 14, 15, 16, 17, 20]

    label, unit, description = LABELS.get(metric, ("저장된 측정값", "로그 값", "저장된 수치 그대로 표시합니다."))
    default_norm = False if metric == "RAW_COORDINATE" else True
    normalized = recorded.get("use_normalization", evaluation.get("use_normalization", default_norm))
    if "use_normalization" in table and table.use_normalization.notna().any():
        normalized = str(table.use_normalization.dropna().iloc[0]).lower() == "true"
    counting_method = recorded.get("count_on", "RETURN")
    if "counting_method" in table and table.counting_method.notna().any():
        counting_method = str(table.counting_method.dropna().iloc[0])

    if not normalized and metric in ("COORDINATE", "LENGTH"):
        unit = "영상 높이 대비 Y좌표" if metric == "COORDINATE" else "영상 높이 대비 길이"
    coordinate_unit = recorded.get("coordinate_unit", "image" if recorded else evaluation.get("coordinate_unit", "image"))
    evaluation["coordinate_unit"] = coordinate_unit
    evaluation["use_normalization"] = normalized

    versions = pd.to_numeric(table.get("threshold_version", pd.Series(0, index=table.index)), errors="coerce").ffill().fillna(0)
    history = {}
    for version in versions.unique():
        bounds = {}
        for side in ("left", "right"):
            bounds[side] = {}
            for key in ("start_val", "target_val"):
                saved = stored.get("thresholds", {}).get(side, {}).get(key) if version == 0 else None
                values = (
                    pd.to_numeric(table.loc[versions.eq(version), f"{side}_{key}"], errors="coerce").dropna()
                    if f"{side}_{key}" in table
                    else pd.Series(dtype=float)
                )
                bounds[side][key] = saved if saved is not None else (float(values.iloc[0]) if len(values) else np.nan)
        history[str(float(version))] = bounds

    return dict(
        exercise=name,
        name=config.get("display_name_kr", name or "동작 정보 없음"),
        metric=metric,
        label=label,
        unit=unit,
        description=description,
        evaluation=evaluation,
        joints=joints,
        by_side=by_side,
        factor=1,
        counting_method=counting_method,
        threshold_history=history,
    )


def threshold_guides(table, ctx, time_column, sides):
    versions = pd.to_numeric(table.get("threshold_version", pd.Series(0, index=table.index)), errors="coerce").ffill().fillna(0)
    guides = pd.DataFrame(index=table.index)
    for side in sides:
        for key in ("start_val", "target_val"):
            column = f"{side}_{key}"
            guides[column] = versions.map(
                lambda v: ctx["threshold_history"].get(str(float(v)), {}).get(side, {}).get(key, np.nan)
            )
    guides.index = table[time_column]
    return guides.dropna(axis=1, how="all") * ctx["factor"]


def quality_guides(table, ctx, time_column, sides):
    bounds = threshold_guides(table, ctx, time_column, sides)
    result = pd.DataFrame(index=bounds.index)
    if ctx.get("counting_method") == "TARGET_POSITION":
        return result
    for side in sides:
        start, target = f"{side}_start_val", f"{side}_target_val"
        if start in bounds and target in bounds:
            for grade, fraction in QUALITY_THRESHOLDS.items():
                result[f"{side}_quality_{grade}"] = bounds[start] + (bounds[target] - bounds[start]) * fraction
    return result


def state_segments(table, time_column):
    rows = []
    times = table[time_column]
    delta = times.diff().loc[lambda s: s > 0].median()
    delta = float(delta) if np.isfinite(delta) else 0.001
    for side in ("left", "right"):
        name = f"{side}_state"
        if name not in table:
            continue
        values = table[name].fillna("UNKNOWN")
        starts = np.flatnonzero((values.ne(values.shift()) | times.diff().gt(max(0.2, delta * 1.5))).to_numpy())
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(table)
            rows.append(
                dict(
                    side=side,
                    state=str(values.iloc[start]),
                    start=float(times.iloc[start]),
                    end=min(float(times.iloc[end]), float(times.iloc[end - 1]) + delta)
                    if end < len(table)
                    else float(times.iloc[-1]) + delta,
                )
            )
    return pd.DataFrame(rows)

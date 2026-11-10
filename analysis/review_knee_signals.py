"""Compare knee signals in existing NPZ logs without loading a pose model.

Run: python analysis/review_knee_signals.py
Outputs are separate from the original logs; count markers are recorded events,
not manually verified repetitions. The median is for offline inspection only.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def inspect(path):
    with np.load(path, allow_pickle=False) as log:
        keypoints = log["keypoints"].copy()
        times = log["timestamps"].copy()
        counts = log["rep_counts"].copy()
        summary = json.loads(str(log["summary_json"]))
    video = cv2.VideoCapture(str(path.with_suffix(".mp4")))
    width, height = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Video dimensions unavailable: {path}")
    xy = keypoints[:, :, :2] * [width, height]
    torso = np.linalg.norm((xy[:, 5] + xy[:, 6] - xy[:, 11] - xy[:, 12]) / 2, axis=1)
    torso_valid = (keypoints[:, [5, 6, 11, 12], 2] >= .35).all(axis=1) & (torso > 1)
    frames, stats = [], []
    for side, hip, knee, ankle, column in [("left", 11, 13, 15, 0), ("right", 12, 14, 16, 1)]:
        def valid(ids):
            p = keypoints[:, ids]
            return ((p[:, :, 2] >= .35).all(axis=1)
                    & np.isfinite(p).all(axis=(1, 2))
                    & ((p[:, :, :2] >= 0) & (p[:, :, :2] <= 1)).all(axis=(1, 2)))

        leg_valid = valid([knee, ankle])
        all_valid = valid([hip, knee, ankle])
        shin = xy[:, ankle] - xy[:, knee]
        thigh = xy[:, hip] - xy[:, knee]
        length = np.linalg.norm(shin, axis=1)
        denominator = length * np.linalg.norm(thigh, axis=1)
        cosine = np.divide((shin * thigh).sum(axis=1), denominator,
                           out=np.full(len(times), np.nan), where=denominator > 1)
        signals = {
            "vertical_gap_px": shin[:, 1],
            "shin_length_px": length,
            "torso_normalized_length": np.divide(length, torso, out=np.full(len(times), np.nan), where=torso > 1),
            "knee_angle_deg": np.degrees(np.arccos(np.clip(cosine, -1, 1))),
            "hip_knee_gap_px": xy[:, knee, 1] - xy[:, hip, 1],
        }
        for name, values in signals.items():
            mask = all_valid if name in ("knee_angle_deg", "hip_knee_gap_px") else leg_valid
            if name == "torso_normalized_length":
                mask = mask & torso_valid
            values[~mask] = np.nan
        frame = pd.DataFrame(dict(time_sec=times, **signals, torso_length_px=torso,
                                  recorded_count=counts[:, column], valid_leg=leg_valid,
                                  date=path.parent.parent.name, session=path.stem, side=side))
        frames.append(frame)
        row = dict(date=path.parent.parent.name, session=path.stem, side=side,
                   frames=len(times), duration_sec=float(times[-1] - times[0]),
                   valid_leg_percent=100 * leg_valid.mean(),
                   recorded_metric=summary.get("measurement", {}).get("metric_type"),
                   saved_direction=summary.get("calibration", {}).get("thresholds", {}).get(side, {}).get("motion_direction"),
                   gap_length_correlation=frame.vertical_gap_px.corr(frame.shin_length_px),
                   raw_normalized_correlation=frame.shin_length_px.corr(frame.torso_normalized_length))
        for name in [*signals, "torso_length_px"]:
            for q, value in frame[name].quantile([.05, .5, .95]).items():
                row[f"{name}_p{int(q*100):02d}"] = value
        stats.append(row)
    return frames, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/0000"))
    parser.add_argument("--output", type=Path, default=Path("analysis/outputs/knee_signal_review"))
    args = parser.parse_args()
    files = sorted(args.root.glob("*/knee_extension/*.npz"))
    if not files:
        raise SystemExit("No knee_extension NPZ logs found")
    args.output.mkdir(parents=True, exist_ok=True)
    all_frames, all_stats = [], []
    with PdfPages(args.output / "signal_comparison.pdf") as pdf:
        for path in files:
            frames, stats = inspect(path)
            all_frames.extend(frames)
            all_stats.extend(stats)
            fig, axes = plt.subplots(5, 2, figsize=(14, 13), sharex=True)
            for col, frame in enumerate(frames):
                events = frame.loc[frame.recorded_count.diff().fillna(0) > 0, "time_sec"]
                series = [(["vertical_gap_px", "shin_length_px"], "Pixel distance"),
                          (["torso_normalized_length"], "Length / torso"),
                          (["knee_angle_deg"], "2D knee angle (degrees)"),
                          (["hip_knee_gap_px", "torso_length_px"], "Pixel distance"),
                          (["recorded_count"], "Recorded count (not ground truth)")]
                for ax, (names, ylabel) in zip(axes[:, col], series):
                    for name in names:
                        ax.plot(frame.time_sec, frame[name], alpha=.25, linewidth=.8)
                        # Trailing 3 samples: no future samples, but no runtime replay.
                        ax.plot(frame.time_sec, frame[name].rolling(3, min_periods=1).median(), label=name)
                    for event in events:
                        ax.axvline(event, color="gray", alpha=.3, linewidth=.8)
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=7, loc="upper right")
                axes[0, col].set_title(frame.side.iloc[0])
                axes[-1, col].set_xlabel("Time (seconds)")
            fig.suptitle(f"{path.parent.parent.name} / {path.stem}\nPale: raw | Solid: trailing 3-sample median | Vertical lines: old recorded events", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .95))
            pdf.savefig(fig)
            fig.savefig(args.output / f"{path.parent.parent.name}_{path.stem}.png", dpi=120)
            plt.close(fig)
    pd.concat(all_frames, ignore_index=True).to_csv(args.output / "all_signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_stats).to_csv(args.output / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"Reviewed {len(files)} logs: {args.output.resolve()}")


if __name__ == "__main__":
    main()

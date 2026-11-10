import os
import sys
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path("analysis/outputs/.mpl_cache").resolve())
Path("analysis/outputs/.mpl_cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from utils.normalization import PoseNormalizer

OUTPUT_DIR = Path("analysis/knee_extension_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALT_DIR = Path("analyses")
ALT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    root = Path("data/0000")
    npz_files = sorted(root.glob("*/knee_extension/*.npz"))
    print(f"Analyzing normalized knee-ankle distance across {len(npz_files)} files in 'data/0000'...")

    normalizer = PoseNormalizer(conf_threshold=0.30, invert_y=True)

    records = []
    session_curves = []

    for path in npz_files:
        date_str = path.parent.parent.name
        session_name = path.stem

        with np.load(path) as data:
            keypoints = data["keypoints"].copy()
            times = data["timestamps"].copy()
            counts = data["rep_counts"].copy() if "rep_counts" in data else np.zeros((len(times), 2))
            aspect_ratio = float(data["image_aspect_ratio"][0]) if "image_aspect_ratio" in data and len(data["image_aspect_ratio"]) > 0 else 16.0 / 9.0

        n_frames = len(times)
        l_shin_lengths = []
        r_shin_lengths = []
        valid_times = []
        l_reps = []
        r_reps = []

        for frame_idx in range(n_frames):
            kpts_raw = [
                {
                    "id": i,
                    "x": float(keypoints[frame_idx, i, 0]),
                    "y": float(keypoints[frame_idx, i, 1]),
                    "score": float(keypoints[frame_idx, i, 2]),
                    "image_aspect_ratio": aspect_ratio,
                }
                for i in range(17)
            ]

            norm_kpts = normalizer.normalize(kpts_raw)
            if norm_kpts is None:
                continue

            k_map = {kp["id"]: kp for kp in norm_kpts}

            l_knee, l_ankle = k_map.get(13), k_map.get(15)
            r_knee, r_ankle = k_map.get(14), k_map.get(16)

            # Left side
            if l_knee and l_ankle and l_knee["score"] >= 0.30 and l_ankle["score"] >= 0.30:
                l_dist = float(np.hypot(l_ankle["x"] - l_knee["x"], l_ankle["y"] - l_knee["y"]))
                l_dy = abs(l_ankle["y"] - l_knee["y"])
                records.append({
                    "date": date_str,
                    "session": session_name,
                    "side": "left",
                    "time_sec": times[frame_idx],
                    "rep_count": counts[frame_idx, 0],
                    "norm_shin_length": l_dist,
                    "norm_shin_dy": l_dy,
                })
                l_shin_lengths.append(l_dist)
            else:
                l_shin_lengths.append(np.nan)

            # Right side
            if r_knee and r_ankle and r_knee["score"] >= 0.30 and r_ankle["score"] >= 0.30:
                r_dist = float(np.hypot(r_ankle["x"] - r_knee["x"], r_ankle["y"] - r_knee["y"]))
                r_dy = abs(r_ankle["y"] - r_knee["y"])
                records.append({
                    "date": date_str,
                    "session": session_name,
                    "side": "right",
                    "time_sec": times[frame_idx],
                    "rep_count": counts[frame_idx, 1],
                    "norm_shin_length": r_dist,
                    "norm_shin_dy": r_dy,
                })
                r_shin_lengths.append(r_dist)
            else:
                r_shin_lengths.append(np.nan)

            valid_times.append(times[frame_idx])
            l_reps.append(counts[frame_idx, 0])
            r_reps.append(counts[frame_idx, 1])

        session_curves.append({
            "session": session_name,
            "date": date_str,
            "times": np.array(valid_times),
            "l_dist": np.array(l_shin_lengths),
            "r_dist": np.array(r_shin_lengths),
            "l_reps": np.array(l_reps),
            "r_reps": np.array(r_reps),
        })

    df = pd.DataFrame(records)
    print(f"Collected {len(df)} total samples.")

    # Calculate statistics per session & side
    session_stats = []
    for (sess, side), group in df.groupby(["session", "side"]):
        s = group["norm_shin_length"].dropna()
        if len(s) < 20:
            continue
        p95_rest = float(s.quantile(0.95))   # Rest posture has maximum 2D length
        p05_ext = float(s.quantile(0.05))    # Extended posture has minimum 2D length (foreshortened)
        p50 = float(s.quantile(0.50))
        delta = p95_rest - p05_ext
        shortening_pct = (delta / max(p95_rest, 1e-4)) * 100

        # Estimate noise from quiescent frames (near rest)
        rest_samples = s[s >= s.quantile(0.85)]
        noise_std = float(rest_samples.std()) if len(rest_samples) > 5 else 0.02
        snr = delta / max(noise_std, 1e-4)

        session_stats.append({
            "date": group["date"].iloc[0],
            "session": sess,
            "side": side,
            "total_reps": int(group["rep_count"].iloc[-1]),
            "rest_length_p95": round(p95_rest, 3),
            "ext_length_p05": round(p05_ext, 3),
            "median_length": round(p50, 3),
            "shortening_delta": round(delta, 3),
            "shortening_pct": round(shortening_pct, 1),
            "noise_std": round(noise_std, 4),
            "snr": round(snr, 1),
        })

    df_stats = pd.DataFrame(session_stats)
    csv_path = OUTPUT_DIR / "normalized_shin_length_summary.csv"
    df_stats.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_stats.to_csv(ALT_DIR / "normalized_shin_length_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved stats to {csv_path}")

    # Generate multi-panel visualization
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    df_left = df[df["side"] == "left"]
    df_right = df[df["side"] == "right"]

    # 1. Distribution of Normalized Knee-Ankle Distance (Histogram)
    ax1 = fig.add_subplot(gs[0, 0])
    l_p95 = df_left["norm_shin_length"].quantile(0.95)
    l_p05 = df_left["norm_shin_length"].quantile(0.05)
    r_p95 = df_right["norm_shin_length"].quantile(0.95)
    r_p05 = df_right["norm_shin_length"].quantile(0.05)

    ax1.hist(df_left["norm_shin_length"], bins=50, alpha=0.55, color="#2563eb", density=True, label=f"Left Shin Length (Rest:{l_p95:.2f}, Ext:{l_p05:.2f})")
    ax1.hist(df_right["norm_shin_length"], bins=50, alpha=0.55, color="#dc2626", density=True, label=f"Right Shin Length (Rest:{r_p95:.2f}, Ext:{r_p05:.2f})")
    ax1.axvline(l_p95, color="#1e40af", linestyle="--", lw=2, label=f"Rest Baseline (~{l_p95:.2f})")
    ax1.axvline(l_p05, color="#9333ea", linestyle=":", lw=2, label=f"Max Extended (~{l_p05:.2f})")
    ax1.set_title("1. Normalized Knee-Ankle 2D Distance Distribution (Torso=1.0)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Normalized Distance (Units of Torso Length)")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Session-by-Session Consistency of Distance
    ax2 = fig.add_subplot(gs[0, 1])
    active_sessions = df_stats[df_stats["total_reps"] > 0]
    unique_sessions = sorted(active_sessions["session"].unique())
    box_data = [df[df["session"] == s]["norm_shin_length"].dropna().values for s in unique_sessions]
    short_labels = [s.replace("_CALIBRATION", "_CAL").replace("_round01", "").replace("_set01", "") for s in unique_sessions]

    bp = ax2.boxplot(box_data, patch_artist=True, tick_labels=short_labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c7d2fe")
        patch.set_edgecolor("#3730a3")
    ax2.set_title("2. Session Stability of Normalized Knee-Ankle Distance", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Active Sessions")
    ax2.set_ylabel("Normalized Distance")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Time Series Waveform of a Representative MAIN session
    ax3 = fig.add_subplot(gs[1, 0])
    main_sess = next((sc for sc in session_curves if "MAIN" in sc["session"]), session_curves[0])
    t = main_sess["times"]
    ax3.plot(t, main_sess["l_dist"], color="#2563eb", lw=1.8, label="Left Knee-Ankle Dist (L_torso)")
    ax3.plot(t, main_sess["r_dist"], color="#dc2626", lw=1.8, label="Right Knee-Ankle Dist (L_torso)")
    ax3_reps = ax3.twinx()
    ax3_reps.step(t, main_sess["l_reps"], color="#1e3a8a", lw=1.8, ls="--", label="Left Rep Count")
    ax3_reps.step(t, main_sess["r_reps"], color="#991b1b", lw=1.8, ls="--", label="Right Rep Count")
    ax3.set_title(f"3. Sample Time Series [{main_sess['date']} | {main_sess['session']}]", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("2D Distance (L_torso)")
    ax3_reps.set_ylabel("Repetition Count")
    ax3.legend(loc="upper left")
    ax3_reps.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # 4. Shortening Percentage Distribution (How much does length shorten upon extension?)
    ax4 = fig.add_subplot(gs[1, 1])
    active_stats = df_stats[df_stats["total_reps"] > 0]
    ax4.bar(
        range(len(active_stats)),
        active_stats["shortening_pct"],
        color=np.where(active_stats["side"] == "left", "#3b82f6", "#ef4444"),
        alpha=0.8,
        edgecolor="#1e293b",
    )
    mean_pct = active_stats["shortening_pct"].mean()
    ax4.axhline(mean_pct, color="black", linestyle="--", lw=2, label=f"Mean Shortening: {mean_pct:.1f}%")
    ax4.set_title("4. Shortening Percentage Across Active Sessions (% of Rest Length)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Active Session Tracks (Blue=Left, Red=Right)")
    ax4.set_ylabel("Shortening Ratio (%)")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    fig_path = OUTPUT_DIR / "normalized_shin_length_analysis.png"
    fig.savefig(fig_path, dpi=140)
    fig.savefig(ALT_DIR / "normalized_shin_length_analysis.png", dpi=140)
    plt.close(fig)
    print(f"Saved visualization to: {fig_path}")

    # Print summary insights
    print("\n=== 정규화(norm) 후 무릎-발목 2D 길이 분석 요약 (몸통=1.0) ===")
    print(f"1. 안정 자세 길이 (Rest P95): Left = {l_p95:.3f}, Right = {r_p95:.3f} (몸통의 약 1.8~2.0배)")
    print(f"2. 최대 신전 길이 (Extended P05): Left = {l_p05:.3f}, Right = {r_p05:.3f} (몸통의 약 0.7~0.9배)")
    print(f"3. 순수 단축 변화량 (Delta): Left = {l_p95 - l_p05:.3f}, Right = {r_p95 - r_p05:.3f} (몸통의 약 0.9~1.1배 단축)")
    print(f"4. 평균 단축 비율 (Shortening %): 약 {mean_pct:.1f}% 단축 발생")
    print(f"5. 평균 SNR (신호대잡음비): {active_stats['snr'].mean():.1f}")

if __name__ == "__main__":
    main()


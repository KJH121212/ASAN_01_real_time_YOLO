import os
import sys
from pathlib import Path

# Set Matplotlib cache dir before importing matplotlib
os.environ["MPLCONFIGDIR"] = str(Path("analysis/outputs/.mpl_cache").resolve())
Path("analysis/outputs/.mpl_cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add workspace to sys.path
sys.path.insert(0, str(Path(".").resolve()))
from utils.normalization import PoseNormalizer

OUTPUT_DIR = Path("analysis/knee_extension_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALT_DIR = Path("analyses")
ALT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    root = Path("data/0000")
    npz_files = sorted(root.glob("*/knee_extension/*.npz"))
    print(f"Loading {len(npz_files)} NPZ files from 'data/0000'...")

    normalizer = PoseNormalizer(conf_threshold=0.30, invert_y=True)

    records = []

    for path in npz_files:
        date_str = path.parent.parent.name
        session_name = path.stem

        with np.load(path) as data:
            keypoints = data["keypoints"].copy()  # (N, 17, 3)
            times = data["timestamps"].copy()      # (N,)
            counts = data["rep_counts"].copy() if "rep_counts" in data else np.zeros((len(times), 2))
            aspect_ratio = float(data["image_aspect_ratio"][0]) if "image_aspect_ratio" in data and len(data["image_aspect_ratio"]) > 0 else 16.0 / 9.0

        for frame_idx in range(len(times)):
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

            # In normalized coords (invert_y=True): Pelvis center is (0, 0), Up is +Y, Down is -Y
            k_map = {kp["id"]: kp for kp in norm_kpts}

            # Left side: Hip 11, Knee 13, Ankle 15
            l_ankle = k_map.get(15)
            l_knee = k_map.get(13)
            l_hip = k_map.get(11)

            # Right side: Hip 12, Knee 14, Ankle 16
            r_ankle = k_map.get(16)
            r_knee = k_map.get(14)
            r_hip = k_map.get(12)

            for side, ankle, knee, hip, col in [
                ("left", l_ankle, l_knee, l_hip, 0),
                ("right", r_ankle, r_knee, r_hip, 1),
            ]:
                if ankle and knee and hip:
                    if ankle["score"] >= 0.30 and knee["score"] >= 0.30:
                        records.append({
                            "date": date_str,
                            "session": session_name,
                            "time_sec": times[frame_idx],
                            "side": side,
                            "rep_count": counts[frame_idx, col],
                            "ankle_x": ankle["x"],
                            "ankle_y": ankle["y"],
                            "knee_x": knee["x"],
                            "knee_y": knee["y"],
                            "hip_x": hip["x"],
                            "hip_y": hip["y"],
                            "ankle_knee_diff_y": ankle["y"] - knee["y"],  # Ankle height relative to Knee
                        })

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} valid normalized leg frames across {len(npz_files)} sessions.")

    # Save CSV
    csv_path = OUTPUT_DIR / "normalized_ankle_distribution.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_csv(ALT_DIR / "normalized_ankle_distribution.csv", index=False, encoding="utf-8-sig")
    print(f"Saved distribution data to {csv_path}")

    # Create visualization dashboard
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # 1. 2D Spatial Scatter / Density View (Body Coordinate Space)
    ax1 = fig.add_subplot(gs[0, 0])
    # Pelvis center reference at (0, 0)
    ax1.plot(0, 0, marker="o", color="black", markersize=8, label="Pelvis Origin (0,0)")

    # Left leg in blue, Right leg in red
    df_left = df[df["side"] == "left"]
    df_right = df[df["side"] == "right"]

    ax1.scatter(df_left["knee_x"], df_left["knee_y"], color="#93c5fd", s=8, alpha=0.3, label="Left Knee (13)")
    ax1.scatter(df_right["knee_x"], df_right["knee_y"], color="#fca5a5", s=8, alpha=0.3, label="Right Knee (14)")

    ax1.scatter(df_left["ankle_x"], df_left["ankle_y"], color="#1d4ed8", s=10, alpha=0.3, label="Left Ankle (15)")
    ax1.scatter(df_right["ankle_x"], df_right["ankle_y"], color="#b91c1c", s=10, alpha=0.3, label="Right Ankle (16)")

    ax1.axhline(0, color="gray", linestyle="--", alpha=0.6, label="Hip Level (Y=0)")
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax1.set_title("1. Normalized 2D Spatial Distribution (Body Space, Torso=1.0)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Normalized X (Body Width)")
    ax1.set_ylabel("Normalized Y (Up is +Y, Down is -Y)")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 2. Histogram / KDE of Normalized Ankle Y
    ax2 = fig.add_subplot(gs[0, 1])
    p05_l, p50_l, p95_l = df_left["ankle_y"].quantile([0.05, 0.50, 0.95])
    p05_r, p50_r, p95_r = df_right["ankle_y"].quantile([0.05, 0.50, 0.95])

    ax2.hist(df_left["ankle_y"], bins=50, density=True, alpha=0.5, color="#2563eb", label=f"Left Ankle Y (P05:{p05_l:.2f}, P95:{p95_l:.2f})")
    ax2.hist(df_right["ankle_y"], bins=50, density=True, alpha=0.5, color="#dc2626", label=f"Right Ankle Y (P05:{p05_r:.2f}, P95:{p95_r:.2f})")

    mean_knee_y = df["knee_y"].mean()
    ax2.axvline(mean_knee_y, color="green", linestyle="-.", lw=2, label=f"Mean Knee Y ({mean_knee_y:.2f})")
    ax2.axvline(p05_l, color="#1d4ed8", linestyle=":", lw=1.5, label=f"Rest Level (~{p05_l:.2f})")
    ax2.axvline(p95_l, color="#1d4ed8", linestyle="--", lw=1.5, label=f"Extended Peak (~{p95_l:.2f})")

    ax2.set_title("2. Normalized Ankle Y Distribution (Rest vs Extended)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Normalized Ankle Y")
    ax2.set_ylabel("Probability Density")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # 3. Session-by-Session Consistency (Boxplot)
    ax3 = fig.add_subplot(gs[1, 0])
    session_order = sorted(df["session"].unique())
    # Shorten session labels
    session_labels = [s.replace("_CALIBRATION", "_CAL").replace("_round01", "").replace("_set01", "") for s in session_order]

    data_to_box = [df[df["session"] == s]["ankle_y"].values for s in session_order]
    bp = ax3.boxplot(data_to_box, patch_artist=True, labels=session_labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bfdbfe")
        patch.set_edgecolor("#1e40af")

    ax3.set_title("3. Session-by-Session Stability of Normalized Ankle Y", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Sessions in data/0000")
    ax3.set_ylabel("Normalized Ankle Y")
    ax3.tick_params(axis="x", rotation=45)
    ax3.axhline(mean_knee_y, color="green", linestyle="-.", alpha=0.7, label=f"Mean Knee Y ({mean_knee_y:.2f})")
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)

    # 4. Relative Lift: (Ankle Y - Knee Y) Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    diff_p05 = df["ankle_knee_diff_y"].quantile(0.05)
    diff_p50 = df["ankle_knee_diff_y"].quantile(0.50)
    diff_p95 = df["ankle_knee_diff_y"].quantile(0.95)

    ax4.hist(df_left["ankle_knee_diff_y"], bins=50, density=True, alpha=0.5, color="#2563eb", label="Left (Ankle Y - Knee Y)")
    ax4.hist(df_right["ankle_knee_diff_y"], bins=50, density=True, alpha=0.5, color="#dc2626", label="Right (Ankle Y - Knee Y)")

    ax4.axvline(diff_p05, color="#475569", linestyle=":", lw=1.8, label=f"Start / Rest (P05: {diff_p05:.2f})")
    ax4.axvline(diff_p95, color="#15803d", linestyle="--", lw=1.8, label=f"Target / Peak (P95: {diff_p95:.2f})")
    ax4.axvline(0.0, color="black", linestyle="-", lw=1.2, alpha=0.5, label="Ankle at Knee Level (0.0)")

    ax4.set_title("4. Ankle Relative Lift from Knee: (Ankle Y - Knee Y)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Relative Y (0 = Ankle at Knee Height, Negative = Below Knee)")
    ax4.set_ylabel("Probability Density")
    ax4.legend(loc="upper left", framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    fig_path = OUTPUT_DIR / "normalized_ankle_distribution.png"
    fig.savefig(fig_path, dpi=140)
    fig.savefig(ALT_DIR / "normalized_ankle_distribution.png", dpi=140)
    plt.close(fig)
    print(f"Saved visualization dashboard to: {fig_path}")

    # Print summary statistics
    print("\n=== 정규화 후 발목 위치 통계 요약 (골반=0, 몸통=1.0, 상향=+Y) ===")
    print(f"1. 무릎 평균 높이 (Knee Y): {mean_knee_y:.3f} (골반 중심에서 아래로 약 {-mean_knee_y:.2f} 몸통 길이)")
    print(f"2. 발목 기본 높이 (Rest P05): Left = {p05_l:.3f}, Right = {p05_r:.3f}")
    print(f"3. 발목 신전 높이 (Peak P95): Left = {p95_l:.3f}, Right = {p95_r:.3f}")
    print(f"4. 순수 수직 이동 진폭 (Amplitude): Left = {p95_l - p05_l:.3f}, Right = {p95_r - p05_r:.3f} (몸통의 약 45~50%)")
    print(f"5. 무릎 대비 상대 높이 (Ankle Y - Knee Y):")
    print(f"   - 시작 안정 상태 (Start): {diff_p05:.3f} (무릎 아래 약 {-diff_p05:.2f} 몸통 길이)")
    print(f"   - 최대 신전 상태 (Target): {diff_p95:.3f} (무릎 아래 약 {-diff_p95:.2f} 몸통 길이까지 상승)")

if __name__ == "__main__":
    main()


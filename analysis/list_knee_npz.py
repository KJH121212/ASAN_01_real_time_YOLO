from pathlib import Path
import numpy as np

def main():
    root = Path("data/0000")
    npz_files = sorted(root.glob("*/knee_extension/*.npz"))
    
    print(f"=== 'data/0000' Knee Extension NPZ 목록 (총 {len(npz_files)}개) ===")
    for idx, path in enumerate(npz_files, 1):
        try:
            with np.load(path) as data:
                frames = len(data["timestamps"]) if "timestamps" in data else 0
                reps = data["rep_counts"][-1] if "rep_counts" in data else [0, 0]
            date_str = path.parent.parent.name
            print(f"[{idx:02d}] {date_str} | {path.name} (프레임: {frames}, 기록 횟수: L:{reps[0]} / R:{reps[1]})")
        except Exception:
            print(f"[{idx:02d}] {path.as_posix()}")

if __name__ == "__main__":
    main()

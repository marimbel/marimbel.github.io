"""
Extract raw hip/knee/ankle landmarks from YOUR exercise video with MediaPipe.

    pip install mediapipe opencv-python numpy matplotlib scipy
    # download the model once (about 5 MB):
    curl -L -o pose_landmarker_lite.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
    python extract_landmarks.py my_squats.mp4 --side left
    python run_experiment.py --csv landmarks.csv

Film from the side, whole body in frame.  --side picks the leg closest to the
camera.  The output CSV is the RAW per-frame detector output; filtering is done
afterwards in run_experiment.py so the effect of each filter can be compared.
"""
import argparse, csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# MediaPipe / BlazePose 33-landmark indices
IDX = {"left": {"hip": 23, "knee": 25, "ankle": 27},
       "right": {"hip": 24, "knee": 26, "ankle": 28}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--model", default="pose_landmarker_lite.task")
    ap.add_argument("--out", default="landmarks.csv")
    ap.add_argument("--image-mode", action="store_true",
                    help="run every frame independently (no MediaPipe tracking/"
                         "internal smoothing) to see truly raw jitter")
    a = ap.parse_args()

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.IMAGE if a.image_mode else vision.RunningMode.VIDEO,
        num_poses=1)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, i = [], 0
    with vision.PoseLandmarker.create_from_options(opts) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(i * 1000 / fps)
            res = det.detect(img) if a.image_mode else det.detect_for_video(img, ts_ms)
            row = {"t": i / fps, "side": a.side}
            for j, k in IDX[a.side].items():
                if res.pose_landmarks:
                    lm = res.pose_landmarks[0][k]
                    row[f"{j}_x"], row[f"{j}_y"], row[f"{j}_vis"] = lm.x, lm.y, lm.visibility
                else:
                    row[f"{j}_x"] = row[f"{j}_y"] = row[f"{j}_vis"] = ""
            rows.append(row)
            i += 1
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} frames to {a.out} ({fps:.1f} fps)")


if __name__ == "__main__":
    main()

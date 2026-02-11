import os
import cv2
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
from sklearn.model_selection import train_test_split
from PIL import Image

# ======================
# CONFIG (LOCKED)
# ======================
ROOT = "archive/FaceForensics++_C23"
OUT = "FFPP_CViT"

FRAME_COUNT = 16
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

CSV_PATH = os.path.join(ROOT, "csv", "FF++_Metadata_Shuffled.csv")

os.makedirs(OUT, exist_ok=True)

DATA_FRACTION = 1

# ======================
# FACE DETECTOR
# ======================
mtcnn = MTCNN(image_size=IMG_SIZE, device=DEVICE,post_process=False)

# ======================
# UTIL
# ======================
def sample_frames(video_path, k):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, total - 1, k).astype(int)
    frames = []

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frames.append(frame)

    cap.release()
    return frames

from PIL import Image

def process_video(video_path, out_dir, prefix):
    frames = sample_frames(video_path, FRAME_COUNT)

    for i, frame in enumerate(frames):
        # OpenCV -> RGB for MTCNN
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = mtcnn(rgb)
        if face is None:
            continue

        # face: torch.Tensor (3, H, WT, RGB, float)
        face = face.permute(1, 2, 0).cpu().numpy()

        # If normalized, bring to [0,255]
        if face.max() <= 1.0:
            face = (face * 255)

        face = face.astype(np.uint8)

        # SAVE WITH PIL (NOT OPENCV)
        img = Image.fromarray(face, mode="RGB")
        img.save(os.path.join(out_dir, f"{prefix}_{i}.jpg"), quality=95)


# ======================
# MAIN
# ======================
def run():
    df = pd.read_csv(CSV_PATH)

    # deterministic split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["Label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["Label"]
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    for split in splits:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(OUT, split, cls), exist_ok=True)

    for split, split_df in splits.items():

        if DATA_FRACTION < 1.0:
            split_df = split_df.sample(
                frac=DATA_FRACTION,
                random_state=RANDOM_SEED
            )

        print(
            f"\nProcessing {split} split "
            f"({len(split_df)} videos, {int(DATA_FRACTION * 100)}%)"
        )

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):

            rel_path = row["File Path"]
            label = row["Label"]

            video_path = os.path.join(ROOT, rel_path)
            if not os.path.exists(video_path):
                continue

            cls = "real" if label == "REAL" else "fake"
            base = os.path.splitext(os.path.basename(rel_path))[0]

            process_video(
                video_path,
                os.path.join(OUT, split, cls),
                base
            )

if __name__ == "__main__":
    run()

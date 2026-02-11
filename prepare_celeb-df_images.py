import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.model_selection import train_test_split

# ======================
# CONFIG
# ======================
ROOT = "Celeb DF"
OUT = "CelebDF_images"

FRAME_COUNT = 16
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

TEST_LIST = os.path.join(ROOT, "List_of_testing_videos.txt")

os.makedirs(OUT, exist_ok=True)

# ======================
# FACE DETECTOR
# ======================
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    device=DEVICE,
    post_process=False
)

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


def process_video(video_path, out_dir, prefix):
    frames = sample_frames(video_path, FRAME_COUNT)

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)

        if face is None:
            continue

        face = face.permute(1, 2, 0).cpu().numpy()

        if face.max() <= 1.0:
            face = (face * 255)

        face = face.astype(np.uint8)
        img = Image.fromarray(face, mode="RGB")

        img.save(
            os.path.join(out_dir, f"{prefix}_{i}.jpg"),
            quality=95
        )

# ======================
# LOAD TEST LIST
# ======================
def load_test_list(path):
    test_set = set()
    with open(path, "r") as f:
        for line in f:
            _, rel = line.strip().split()
            test_set.add(rel)
    return test_set

# ======================
# MAIN
# ======================
def run():
    random.seed(RANDOM_SEED)

    # ======================
    # LOAD OFFICIAL TEST LIST
    # ======================
    test_videos = load_test_list(TEST_LIST)

    all_samples = []

    # ======================
    # COLLECT ALL VIDEOS
    # ======================

    # REAL
    for sub in ["Celeb-real", "YouTube-real"]:
        root = os.path.join(ROOT, sub)
        for f in os.listdir(root):
            if f.endswith(".mp4"):
                rel = f"{sub}/{f}"
                all_samples.append((rel, "real"))

    # FAKE
    fake_root = os.path.join(ROOT, "Celeb-synthesis")
    for f in os.listdir(fake_root):
        if f.endswith(".mp4"):
            rel = f"Celeb-synthesis/{f}"
            all_samples.append((rel, "fake"))

    df = pd.DataFrame(all_samples, columns=["path", "label"])

    # ======================
    # 1️⃣ TEST SPLIT (FIRST & FINAL)
    # ======================
    test_df = df[df["path"].isin(test_videos)].reset_index(drop=True)

    # ======================
    # 2️⃣ REMOVE TEST FROM POOL
    # ======================
    remain_df = df[~df["path"].isin(test_videos)].reset_index(drop=True)

    # ======================
    # 3️⃣ TRAIN / VAL SPLIT
    # ======================
    train_df, val_df = train_test_split(
        remain_df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=remain_df["label"]
    )

    # ======================
    # SPLITS (ORDER MATTERS)
    # ======================
    splits = [
        ("test", test_df),
        ("train", train_df),
        ("val", val_df),
    ]

    # ======================
    # CREATE DIRS
    # ======================
    for split, _ in splits:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(OUT, split, cls), exist_ok=True)

    # ======================
    # PROCESS (TEST FIRST)
    # ======================
    for split, split_df in splits:
        print(f"\n[{split.upper()}] {len(split_df)} videos")

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            video_path = os.path.join(ROOT, row["path"])
            if not os.path.exists(video_path):
                continue

            base = os.path.splitext(os.path.basename(row["path"]))[0]
            out_dir = os.path.join(OUT, split, row["label"])

            process_video(video_path, out_dir, base)



if __name__ == "__main__":
    run()

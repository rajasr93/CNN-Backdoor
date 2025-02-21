import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_gtsrb(meta_csv, train_csv, test_csv, data_dir, img_size=(32, 32)):
    """
    Load GTSRB data based on CSV files and folder structure.
    """
    # Load training data.
    train_data = pd.read_csv(os.path.join(data_dir, train_csv))
    train_images = []
    train_labels = []
    print("[INFO] Loading training images...")
    for _, row in tqdm(train_data.iterrows(), total=len(train_data)):
        # Determine correct image path.
        if row["Path"].startswith("Train") or row["Path"].startswith("Test"):
            img_path = os.path.join(data_dir, row["Path"])
        else:
            img_path = os.path.join(data_dir, "Train", row["Path"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not load training image: {img_path}")
            continue
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        train_images.append(img)
        train_labels.append(row["ClassId"])
    
    # Load testing data.
    test_data = pd.read_csv(os.path.join(data_dir, test_csv))
    test_images = []
    test_labels = []
    print("[INFO] Loading test images...")
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        if row["Path"].startswith("Train") or row["Path"].startswith("Test"):
            img_path = os.path.join(data_dir, row["Path"])
        else:
            img_path = os.path.join(data_dir, "Test", row["Path"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not load test image: {img_path}")
            continue
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        test_images.append(img)
        test_labels.append(row["ClassId"])
    
    return (np.array(train_images), np.array(train_labels)), (np.array(test_images), np.array(test_labels))
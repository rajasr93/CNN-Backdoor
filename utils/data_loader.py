import numpy as np
import pandas as pd
import cv2
import os

def load_mnist(csv_path, img_size=(28, 28)):
    """
    Load MNIST data from a CSV file.
    Each row: label, pix1, pix2, ..., pix(28*28)
    """
    data = pd.read_csv(csv_path, header=None)
    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values.astype(np.float32)
    images = images.reshape(-1, img_size[0], img_size[1], 1)
    # Normalize images to [0, 1]
    images = images / 255.0
    return images, labels

def load_gtsrb(meta_csv, train_csv, test_csv, data_dir, img_size=(32, 32)):
    """
    Load GTSRB data based on the CSV files and folder structure.

    Expected directory structure:
      data_dir/
         ├── Meta/          # Contains meta files (e.g., Meta.csv)
         ├── Train/         # Contains subfolders 0, 1, ..., 42 with training images
         └── Test/          # Contains test images directly

    CSV formats:
      - train.csv: Columns include Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
                   where Path is like "0/00000_00000_00000.png" (i.e. subfolder/<filename>)
      - test.csv:  Columns include Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
                   where Path is like "00000.png"

    This function checks if the 'Path' column already includes a folder prefix ("Train" or "Test").
    If not, it prepends the appropriate folder name.
    """
    # For training data
    train_data = pd.read_csv(os.path.join(data_dir, train_csv))
    train_images = []
    train_labels = []
    for _, row in train_data.iterrows():
        # If the path does not already include a subfolder name "Train" or "Test",
        # assume it is relative to the Train folder.
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
    
    # For testing data
    test_data = pd.read_csv(os.path.join(data_dir, test_csv))
    test_images = []
    test_labels = []
    for _, row in test_data.iterrows():
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

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
    This function returns training and testing sets.
    """
    # For training data
    train_data = pd.read_csv(os.path.join(data_dir, train_csv))
    train_images = []
    train_labels = []
    for _, row in train_data.iterrows():
        # Assume the path in CSV is relative to the Train folder
        img_path = os.path.join(data_dir, "Train", row["Path"])
        img = cv2.imread(img_path)
        if img is None:
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
        # Assume the path in CSV is relative to the Test folder
        img_path = os.path.join(data_dir, "Test", row["Path"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        test_images.append(img)
        test_labels.append(row["ClassId"])
    
    return (np.array(train_images), np.array(train_labels)), (np.array(test_images), np.array(test_labels))
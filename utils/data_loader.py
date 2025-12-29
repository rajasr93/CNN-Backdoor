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
    if not os.path.isabs(train_csv):
        train_csv = os.path.join(data_dir, train_csv)
    train_data = pd.read_csv(train_csv)
    train_images = []
    train_labels = []
    print("[INFO] Loading training images...")
    for _, row in tqdm(train_data.iterrows(), total=len(train_data)):
        # Determine correct image path.
        if row["Path"].startswith("Train") or row["Path"].startswith("Test") or row["Path"].startswith("Final"):
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
    if not os.path.isabs(test_csv):
        test_csv = os.path.join(data_dir, test_csv)
    test_data = pd.read_csv(test_csv)
    test_images = []
    test_labels = []
    print("[INFO] Loading test images...")
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        if row["Path"].startswith("Train") or row["Path"].startswith("Test") or row["Path"].startswith("Final"):
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

def load_mnist(csv_path):
    """
    Load MNIST data from CSV or Keras defaults.
    """
    # Check if file exists
    if csv_path and os.path.exists(csv_path):
        print(f"[INFO] Loading MNIST from CSV: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            # Assume label is 'label' or first column
            if 'label' in df.columns:
                y = df['label'].values
                x = df.drop('label', axis=1).values
            else:
                y = df.iloc[:, 0].values
                x = df.iloc[:, 1:].values
            
            # Reshape to (N, 28, 28, 1)
            # Check size
            if x.shape[1] == 784:
                x = x.reshape(-1, 28, 28, 1)
            elif x.ndim == 3 and x.shape[1:] == (28, 28):
                x = x.reshape(-1, 28, 28, 1)
            
            x = x.astype('float32') / 255.0
            return x, y
        except Exception as e:
            print(f"[ERROR] Failed to load MNIST from CSV: {e}")
            raise e
    else:
        print(f"[INFO] CSV not found: {csv_path}. Loading from Keras defaults.")
        try:
            from tensorflow.keras.datasets import mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            
            if csv_path and 'train' in str(csv_path).lower():
                x, y = x_train, y_train
            else:
                x, y = x_test, y_test
            
            # Reshape to have channel dimension
            if x.ndim == 3:
                x = x.reshape(x.shape[0], 28, 28, 1)
                
            x = x.astype('float32') / 255.0
            return x, y
        except ImportError:
            print("[ERROR] Keras not installed and CSV not found.")
            return None, None
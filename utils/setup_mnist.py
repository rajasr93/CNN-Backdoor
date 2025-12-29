import os
import tensorflow as tf
import pandas as pd
import numpy as np

def setup_mnist():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'MNIST')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"[INFO] Created directory: {data_dir}")
    
    train_csv_path = os.path.join(data_dir, 'mnist_train.csv')
    test_csv_path = os.path.join(data_dir, 'mnist_test.csv')
    
    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        print("[INFO] MNIST CSV files already exist.")
        return

    print("[INFO] Downloading MNIST data using Keras...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print("[INFO] Converting Training Data to CSV...")
    # Flatten images: (60000, 28, 28) -> (60000, 784)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    train_df = pd.DataFrame(x_train_flat)
    train_df.insert(0, 'label', y_train)
    train_df.to_csv(train_csv_path, index=False)
    print(f"[INFO] Saved {train_csv_path}")
    
    print("[INFO] Converting Test Data to CSV...")
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    test_df = pd.DataFrame(x_test_flat)
    test_df.insert(0, 'label', y_test)
    test_df.to_csv(test_csv_path, index=False)
    print(f"[INFO] Saved {test_csv_path}")
    
    print("[SUCCESS] MNIST data setup complete.")

if __name__ == "__main__":
    setup_mnist()

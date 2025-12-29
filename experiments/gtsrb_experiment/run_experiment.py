import os
import yaml
import numpy as np
import datetime
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Import functions from your project modules.
from utils.data_loader import load_gtsrb
from models.gtsrb_model import create_gtsrb_model
from attacks.sinusoidal_signal import create_sinusoidal_signal
from attacks.backdoor_attack import inject_backdoor_signal
from utils.evaluation import evaluate_model

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run GTSRB Experiment')
parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
args = parser.parse_args()

# Get the project base directory relative to this file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load configuration
if args.config:
    config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
else:
    config_path = os.path.join(base_dir, "config", "gtsrb_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Build absolute paths for the GTSRB data
data_dir = os.path.join(base_dir, config["dataset"]["data_dir"])
meta_csv = os.path.join(data_dir, "Meta.csv")
train_csv = os.path.join(data_dir, config["dataset"]["train_csv"])
test_csv = os.path.join(data_dir, config["dataset"]["test_csv"])

print("Data directory:", data_dir)
print("Meta CSV absolute path:", meta_csv)
print("Train CSV absolute path:", train_csv)
print("Test CSV absolute path:", test_csv)

# Load GTSRB data using the provided image size
(train_images, train_labels), (test_images, test_labels) = load_gtsrb(
    meta_csv=meta_csv,
    train_csv=train_csv,
    test_csv=test_csv,
    data_dir=data_dir,
    img_size=tuple(config["model"]["input_shape"][:2])
)
print(f"Loaded {train_images.shape[0]} training images and {test_images.shape[0]} test images.")

# Explicitly split training data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels,
    test_size=config["training"]["validation_split"],
    random_state=42,
    shuffle=True
)
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# --- Create Sinusoidal Backdoor Signal for Training ---
img_shape = tuple(config["model"]["input_shape"])  # e.g., (32, 32, 3)
print("[INFO] Creating sinusoidal backdoor signal for training...")
sinusoidal_signal_train = create_sinusoidal_signal(
    delta=config["attack"]["delta_train"],
    freq=config["attack"]["freq"],
    img_shape=img_shape
)

# --- Inject Backdoor Signal into a Fraction of the Target Class Samples ---
# For a backdoor attack, typically a fraction of images from a chosen source class are modified.
print("[INFO] Injecting backdoor signal into training samples...")
X_train_corrupted = inject_backdoor_signal(
    X_train,
    y_train,
    target_class=config["attack"]["target_class"],
    alpha=config["attack"]["alpha"],
    backdoor_signal=sinusoidal_signal_train
)
print("[INFO] Backdoor injection complete. Using corrupted training data for model training.")

# --- Create and Compile the Model ---
model = create_gtsrb_model(input_shape=img_shape, num_classes=43)
model.summary()

# --- Train the Model on Corrupted Data ---
print("[INFO] Starting model training on corrupted data...")
history = model.fit(
    X_train_corrupted, y_train,
    epochs=config["model"]["epochs"],
    batch_size=config["model"]["batch_size"],
    validation_data=(X_val, y_val)
)

# --- Evaluate on Clean Test Data ---
print("Evaluating on clean test data:")
score_clean = model.evaluate(test_images, test_labels)
print("Clean Test Loss, Accuracy:", score_clean)

# --- Prepare Test Sinusoidal Backdoor Signal ---
print("[INFO] Creating sinusoidal backdoor signal for testing...")
test_sinusoidal_signal = create_sinusoidal_signal(
    delta=config["attack"]["delta_test"],
    freq=config["attack"]["freq"],
    img_shape=img_shape
)

# --- Inject Backdoor Signal into Test Data ---
# Here we inject the test trigger into all test images belonging to the target class.
print("[INFO] Injecting backdoor signal into test data for evaluation...")
test_images_backdoor = inject_backdoor_signal(
    test_images,
    test_labels,
    target_class=config["attack"]["target_class"],
    alpha=1.0,  # inject into all samples of the target class
    backdoor_signal=test_sinusoidal_signal
)

print("Evaluating on backdoored test data:")
score_backdoor = evaluate_model(test_images_backdoor, test_labels)
print("Backdoored Test Loss, Accuracy:", score_backdoor)

# --- Plot and Save Training History ---
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Validation Loss')
ax[0].set_title('Loss')
ax[0].legend()

ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[1].set_title('Accuracy')
ax[1].legend()

plt.tight_layout()
plot_filename = os.path.join(base_dir, "experiments", "gtsrb_experiment",
                             f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(plot_filename)
plt.close()
print("Saved training history plot to:", plot_filename)

# --- Save Experiment Log ---
specs = {
    "dataset": "GTSRB",
    "target_class": config["attack"]["target_class"],
    "alpha": config["attack"]["alpha"],
    "delta_train": config["attack"]["delta_train"],
    "delta_test": config["attack"]["delta_test"],
    "freq": config["attack"]["freq"]
}
spec_str = "_".join(f"{k}{v}" for k, v in specs.items())
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_log = {
    "timestamp": timestamp,
    "specifications": specs,
    "score_clean": score_clean,
    "score_backdoor": score_backdoor,
    "history": history.history
}
log_filename = os.path.join(base_dir, "experiments", "gtsrb_experiment",
                            f"log_{spec_str}_{timestamp}.json")
with open(log_filename, "w") as f:
    json.dump(experiment_log, f, indent=4)
print("Saved experiment log to:", log_filename)

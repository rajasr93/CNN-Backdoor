import os
import yaml
import numpy as np
from models.gtsrb_model import create_gtsrb_model
from utils.data_loader import load_gtsrb
from attacks.sinusoidal_signal import create_sinusoidal_signal
from attacks.backdoor_attack import inject_backdoor_signal
from utils.evaluation import evaluate_model

# Get the project base directory relative to this file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load configuration
config_path = os.path.join(base_dir, "config", "gtsrb_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Build absolute paths for the GTSRB data
# data_dir should point to the folder containing the CSV files and subfolders (Meta, Train, Test)
data_dir = os.path.join(base_dir, config["dataset"]["data_dir"])
meta_csv = os.path.join(data_dir, "Meta.csv")
train_csv = os.path.join(data_dir, config["dataset"]["train_csv"])
test_csv = os.path.join(data_dir, config["dataset"]["test_csv"])

print("Data directory:", data_dir)
print("Meta CSV absolute path:", meta_csv)
print("Train CSV absolute path:", train_csv)
print("Test CSV absolute path:", test_csv)

# Load GTSRB data using the absolute paths and provided image size
(train_images, train_labels), (test_images, test_labels) = load_gtsrb(
    meta_csv=meta_csv,
    train_csv=train_csv,
    test_csv=test_csv,
    data_dir=data_dir,
    img_size=tuple(config["model"]["input_shape"][:2])
)

# Create sinusoidal backdoor signal for GTSRB
img_shape = tuple(config["model"]["input_shape"])
sinusoidal_signal = create_sinusoidal_signal(
    delta=config["attack"]["delta_train"],
    freq=config["attack"]["freq"],
    img_shape=img_shape
)

# Inject backdoor signal into a fraction of the target class samples
train_images_corrupted = inject_backdoor_signal(
    train_images,
    train_labels,
    target_class=config["attack"]["target_class"],
    alpha=config["attack"]["alpha"],
    backdoor_signal=sinusoidal_signal
)

# Create and train the GTSRB model
model = create_gtsrb_model(input_shape=img_shape)
model.fit(
    train_images_corrupted,
    train_labels,
    epochs=config["model"]["epochs"],
    batch_size=config["model"]["batch_size"],
    validation_split=config["training"]["validation_split"]
)

# Evaluate on clean test data
print("Evaluating on clean test data:")
score_clean = model.evaluate(test_images, test_labels)
print("Clean Test Loss, Accuracy:", score_clean)

# Prepare test sinusoidal signal with a different strength if needed
test_sinusoidal_signal = create_sinusoidal_signal(
    delta=config["attack"]["delta_test"],
    freq=config["attack"]["freq"],
    img_shape=img_shape
)

print("Evaluating on backdoored test data:")
score_backdoor = evaluate_model(model, test_images, test_labels, backdoor_signal=test_sinusoidal_signal)
print("Backdoored Test Loss, Accuracy:", score_backdoor)

import os
import yaml
import numpy as np
import datetime
import json
import matplotlib.pyplot as plt

from models.mnist_model import create_mnist_model
from utils.data_loader import load_mnist
from attacks.ramp_signal import create_ramp_signal
from attacks.backdoor_attack import inject_backdoor_signal
from utils.evaluation import evaluate_model

# Get the project base directory relative to this file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(base_dir, "config", "mnist_config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Convert dataset paths to absolute paths using base_dir
train_csv = os.path.join(base_dir, config["dataset"]["train_csv"])
test_csv = os.path.join(base_dir, config["dataset"]["test_csv"])

print("Training CSV absolute path:", train_csv)
print("Testing CSV absolute path:", test_csv)

x_train, y_train = load_mnist(train_csv)
x_test, y_test = load_mnist(test_csv)

# Create backdoor signal for MNIST (same shape as an image, without channel dim)
img_shape = (config["model"]["input_shape"][0], config["model"]["input_shape"][1])
ramp_signal = create_ramp_signal(config["attack"]["delta_train"], img_shape)
ramp_signal = ramp_signal[..., np.newaxis]

# Inject backdoor signal into a fraction of the target class samples
x_train_corrupted = inject_backdoor_signal(
    x_train,
    y_train,
    target_class=config["attack"]["target_class"],
    alpha=config["attack"]["alpha"],
    backdoor_signal=ramp_signal
)

# Create and train the MNIST model, capturing training history
model = create_mnist_model(input_shape=tuple(config["model"]["input_shape"]))
history = model.fit(
    x_train_corrupted,
    y_train,
    epochs=config["model"]["epochs"],
    batch_size=config["model"]["batch_size"],
    validation_split=config["training"]["validation_split"]
)

# Evaluate on clean test data
print("Evaluating on clean test data:")
score_clean = model.evaluate(x_test, y_test)
print("Clean Test Loss, Accuracy:", score_clean)

# Prepare test backdoor signal with a different strength if needed
test_ramp_signal = create_ramp_signal(config["attack"]["delta_test"], img_shape)
test_ramp_signal = test_ramp_signal[..., np.newaxis]

print("Evaluating on backdoored test data:")
score_backdoor = evaluate_model(model, x_test, y_test, backdoor_signal=test_ramp_signal)
print("Backdoored Test Loss, Accuracy:", score_backdoor)

# Plot training history (loss and accuracy)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Validation Loss')
ax[0].set_title('Loss')
ax[0].legend()

ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[1].set_title('Accuracy')
ax[1].legend()

# Create a string encoding key experiment parameters for filenames
specs = {
    "dataset": "MNIST",
    "target_class": config["attack"]["target_class"],
    "alpha": config["attack"]["alpha"],
    "delta_train": config["attack"]["delta_train"],
    "delta_test": config["attack"]["delta_test"]
}
spec_str = "_".join(f"{k}{v}" for k, v in specs.items())
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Save plot and experiment log in the experiments/mnist_experiment folder
plot_filename = os.path.join(base_dir, "experiments", "mnist_experiment", f"history_{spec_str}_{timestamp}.png")
plt.savefig(plot_filename)
plt.close()
print("Saved training history plot to:", plot_filename)

experiment_log = {
    "timestamp": timestamp,
    "specifications": specs,
    "score_clean": score_clean,
    "score_backdoor": score_backdoor,
    "history": history.history
}
log_filename = os.path.join(base_dir, "experiments", "mnist_experiment", f"log_{spec_str}_{timestamp}.json")
with open(log_filename, "w") as f:
    json.dump(experiment_log, f, indent=4)
print("Saved experiment log to:", log_filename)

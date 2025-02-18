import os
import yaml
import numpy as np
from models.mnist_model import create_mnist_model
from utils.data_loader import load_mnist
from attacks.ramp_signal import create_ramp_signal
from attacks.backdoor_attack import inject_backdoor_signal
from utils.evaluation import evaluate_model

# Load configuration
with open(os.path.join("..", "..", "config", "mnist_config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Load MNIST data
train_csv = config["dataset"]["train_csv"]
test_csv = config["dataset"]["test_csv"]

x_train, y_train = load_mnist(train_csv)
x_test, y_test = load_mnist(test_csv)

# Create backdoor signal for MNIST (same shape as an image, without channel dim)
img_shape = (config["model"]["input_shape"][0], config["model"]["input_shape"][1])
ramp_signal = create_ramp_signal(config["attack"]["delta_train"], img_shape)
# Expand dimensions to match channel dimension
ramp_signal = ramp_signal[..., np.newaxis]

# Inject backdoor signal into a fraction of the target class samples
x_train_corrupted = inject_backdoor_signal(
    x_train,
    y_train,
    target_class=config["attack"]["target_class"],
    alpha=config["attack"]["alpha"],
    backdoor_signal=ramp_signal
)

# Create and train the MNIST model
model = create_mnist_model(input_shape=tuple(config["model"]["input_shape"]))
model.fit(
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

# Prepare test backdoor signal with different strength if needed
test_ramp_signal = create_ramp_signal(config["attack"]["delta_test"], img_shape)
test_ramp_signal = test_ramp_signal[..., np.newaxis]

print("Evaluating on backdoored test data:")
score_backdoor = evaluate_model(model, x_test, y_test, backdoor_signal=test_ramp_signal)
print("Backdoored Test Loss, Accuracy:", score_backdoor)

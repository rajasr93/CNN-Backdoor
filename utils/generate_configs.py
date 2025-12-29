import yaml
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = os.path.join(base_dir, "config")

# --- MNIST Configurations ---
mnist_base = {
    "dataset": {
        "train_csv": "data/MNIST/mnist_train.csv",
        "test_csv": "data/MNIST/mnist_test.csv"
    },
    "model": {
        "input_shape": [28, 28, 1],
        "batch_size": 64,
        "epochs": 100
    },
    "training": {
        "validation_split": 0.1
    },
    "attack": {
        "target_class": 0,
        # alpha and delta will be varied
    }
}

mnist_variations = [
    {"alpha": 0.1, "delta": 0.2},
    {"alpha": 0.25, "delta": 0.3},
    {"alpha": 0.4, "delta": 0.4}
]

for i, var in enumerate(mnist_variations, 1):
    cfg = mnist_base.copy()
    cfg["attack"] = cfg["attack"].copy()
    cfg["attack"]["alpha"] = var["alpha"]
    cfg["attack"]["delta_train"] = var["delta"]
    cfg["attack"]["delta_test"] = var["delta"]
    
    filename = os.path.join(config_dir, f"mnist_run{i}.yaml")
    with open(filename, "w") as f:
        yaml.dump(cfg, f)
    print(f"Created {filename}")

# --- GTSRB Configurations ---
gtsrb_base = {
    "dataset": {
        "data_dir": "data/GTSRB/",
        "train_csv": "Train.csv",
        "test_csv": "Test.csv"
    },
    "model": {
        "input_shape": [32, 32, 3],
        "batch_size": 64,
        "epochs": 100
    },
    "training": {
        "validation_split": 0.1
    },
    "attack": {
        "target_class": 0,
        "freq": 6
        # alpha and deltas will be varied
    }
}

gtsrb_variations = [
    {"alpha": 0.05, "delta": 0.08},
    {"alpha": 0.15, "delta": 0.12},
    {"alpha": 0.25, "delta": 0.16}
]

for i, var in enumerate(gtsrb_variations, 1):
    cfg = gtsrb_base.copy()
    cfg["attack"] = cfg["attack"].copy()
    cfg["attack"]["alpha"] = var["alpha"]
    cfg["attack"]["delta_train"] = var["delta"]
    cfg["attack"]["delta_test"] = var["delta"]
    
    filename = os.path.join(config_dir, f"gtsrb_run{i}.yaml")
    with open(filename, "w") as f:
        yaml.dump(cfg, f)
    print(f"Created {filename}")

# CNN Backdoor Attack: Training Set Corruption Without Label Poisoning

This repository implements a backdoor attack on Convolutional Neural Networks (CNNs) by corrupting the training set without modifying the labels. The attack injects a specific signal (pattern) into a fraction of the target class images. When the model is trained on this corrupted dataset, it learns to associate the pattern with the target class. At inference time, any input containing this pattern will be misclassified as the target class.

## Project Structure

```
.
├── attacks/                # Core attack logic
│   ├── backdoor_attack.py  # Function to inject backdoor signal
│   └── sinusoidal_signal.py# Signal generation (e.g., sinusoidal pattern)
├── config/                 # YAML configuration files for experiments
├── data/                   # Dataset storage (MNIST, GTSRB)
├── experiments/            # Experiment execution scripts
│   ├── mnist_experiment/   # MNIST-specific experiment runner
│   └── gtsrb_experiment/   # GTSRB-specific experiment runner
├── models/                 # Saved models (after training)
├── utils/                  # Utility scripts
│   ├── data_loader.py      # Data loading and preprocessing
│   └── generate_configs.py # Script to generate config files
├── requirements.txt        # Python dependencies
└── run_all_experiments.sh  # Master script to run all experiments
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd CNN-Backdoor
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment. The project is tested with Python 3.12.
    ```bash
    # Create a virtual environment (optional)
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Setup

The project requires the MNIST and GTSRB datasets. The `experiments` scripts are configured to look for data in the `data/` directory.
-   **MNIST**: Typically downloaded automatically by Keras/TensorFlow, but the scripts expect CSVs in `data/MNIST/`.
-   **GTSRB**: Expects `Train.csv` and `Test.csv` in `data/GTSRB/`.

## Usage

### 1. Generate Configurations
Before running experiments, ensure the configuration files are generated. This script creates the YAML config files in the `config/` directory.

```bash
python utils/generate_configs.py
```

### 2. Run All Experiments (Recommended)
Use the provided shell script to execute the full suite of experiments for both MNIST and GTSRB. This runs 3 variations for each dataset sequentially.

```bash
./run_all_experiments.sh
```

*Note: Ensure the script is executable (`chmod +x run_all_experiments.sh`).*

### 3. Run Individual Experiments
You can also run experiments individually by specifying a config file.

**For MNIST:**
```bash
python experiments/mnist_experiment/run_experiment.py --config config/mnist_run1.yaml
```

**For GTSRB:**
```bash
python experiments/gtsrb_experiment/run_experiment.py --config config/gtsrb_run1.yaml
```

## Experiment Details

The experiments test the attack's effectiveness under varying conditions of corruption rate (`alpha`) and signal strength (`delta`).

### MNIST Experiments
-   **Target Class**: 0
-   **Backdoor Signal**: Sinusoidal pattern
-   **Variations**:
    1.  **Run 1**: `alpha` = 0.1 (10% corruption), `delta` = 0.2
    2.  **Run 2**: `alpha` = 0.25 (25% corruption), `delta` = 0.3
    3.  **Run 3**: `alpha` = 0.4 (40% corruption), `delta` = 0.4

### GTSRB Experiments
-   **Target Class**: 0
-   **Backdoor Signal**: Sinusoidal pattern (Frequency = 6)
-   **Variations**:
    1.  **Run 1**: `alpha` = 0.05, `delta` = 0.08
    2.  **Run 2**: `alpha` = 0.15, `delta` = 0.12
    3.  **Run 3**: `alpha` = 0.25, `delta` = 0.16

## Results

Authentication and training logs are saved to the root directory with filenames corresponding to the run (e.g., `mnist_run1.log`). These logs contain training progress and final evaluation metrics.

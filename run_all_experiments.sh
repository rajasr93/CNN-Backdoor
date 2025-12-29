#!/bin/bash
source .venv_312/bin/activate
export PYTHONPATH=$PYTHONPATH:.

echo "Starting Multi-Run Experiments..."

# --- MNIST Runs ---
echo "--- Starting MNIST Run 1 ---"
python3 experiments/mnist_experiment/run_experiment.py --config config/mnist_run1.yaml > mnist_run1.log 2>&1
echo "MNIST Run 1 Completed (Log: mnist_run1.log)"

echo "--- Starting MNIST Run 2 ---"
python3 experiments/mnist_experiment/run_experiment.py --config config/mnist_run2.yaml > mnist_run2.log 2>&1
echo "MNIST Run 2 Completed (Log: mnist_run2.log)"

echo "--- Starting MNIST Run 3 ---"
python3 experiments/mnist_experiment/run_experiment.py --config config/mnist_run3.yaml > mnist_run3.log 2>&1
echo "MNIST Run 3 Completed (Log: mnist_run3.log)"

# --- GTSRB Runs ---
echo "--- Starting GTSRB Run 1 ---"
python3 experiments/gtsrb_experiment/run_experiment.py --config config/gtsrb_run1.yaml > gtsrb_run1.log 2>&1
echo "GTSRB Run 1 Completed (Log: gtsrb_run1.log)"

echo "--- Starting GTSRB Run 2 ---"
python3 experiments/gtsrb_experiment/run_experiment.py --config config/gtsrb_run2.yaml > gtsrb_run2.log 2>&1
echo "GTSRB Run 2 Completed (Log: gtsrb_run2.log)"

echo "--- Starting GTSRB Run 3 ---"
python3 experiments/gtsrb_experiment/run_experiment.py --config config/gtsrb_run3.yaml > gtsrb_run3.log 2>&1
echo "GTSRB Run 3 Completed (Log: gtsrb_run3.log)"

echo "All Experiments Completed."

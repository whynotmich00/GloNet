#!/bin/bash

# Training script with different parameter configurations
# Make sure train.py is in the same directory or adjust the path accordingly

echo "Starting training experiments..."
echo "================================"

# Experiment 1: Baseline ResNet50
echo "Running Experiment 1: ResNet50 architecture"
python train.py \
    --model="ResNet50" \
    --result_dir="training_results/complete_session" \
    --epochs=200

echo "Experiment 1 completed."
echo "--------------------------------"

# Experiment 2: Higher learning rate with momentum
echo "Running Experiment 2: ResNet100 architecture"
python train.py \
    --model="ResNet100" \
    --result_dir="training_results/complete_session" \
    --epochs=200

echo "Experiment 2 completed."
echo "--------------------------------"

# Experiment 3: Different architecture (assuming GloNet is available)
echo "Running Experiment 3: ResNet200 architecture"
python train.py \
    --model="ResNet200" \
    --result_dir="training_results/complete_session" \
    --epochs=200

echo "Experiment 3 completed."
echo "--------------------------------"

# Experiment 4: Smaller batch size, more epochs
echo "Running Experiment 4: GloNet 52 layers"
python train.py \
    --model="GloNet" \
    --epochs=200 \
    --result_dir="training_results/complete_session" \
    --num_layers=52

echo "Experiment 4 completed."
echo "--------------------------------"

# Experiment 5: Smaller batch size, more epochs
echo "Running Experiment 5: GloNet 102 layers"
python train.py \
    --model="GloNet" \
    --epochs=200 \
    --result_dir="training_results/complete_session" \
    --num_layers=102

echo "Experiment 5 completed."
echo "--------------------------------"

# Experiment 5: Smaller batch size, more epochs
echo "Running Experiment 6: GloNet 202 layers"
python train.py \
    --model="GloNet" \
    --epochs=200 \
    --result_dir="training_results/complete_session" \
    --num_layers=202

echo "Experiment 6 completed."
echo "--------------------------------"

echo "All training experiments completed!"
echo "================================"
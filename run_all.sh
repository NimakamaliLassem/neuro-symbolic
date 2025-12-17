#!/bin/bash

# Unified RNNLogic Runner Script

echo "========================================"
echo "      RNNLogic Unified Runner"
echo "========================================"
echo "Select Model Approach:"
echo "1. GRU (Standard EM)"
echo "2. GRU (RL Guided)"
echo "3. Mamba (Standard)"
echo "4. Mamba (RL Guided)"
echo "5. LSTM (RL Guided Experiment)"
echo "6. TransformerXL"
echo "7. GPT-2"
echo "8. TRM (Tiny Recursive Model)"
echo "========================================"
read -p "Enter choice [1-8]: " model_choice

echo ""
echo "========================================"
echo "Select Dataset:"
echo "1. Kinship"
echo "2. UMLS"
echo "3. Countries S3"
echo "========================================"
read -p "Enter choice [1-3]: " dataset_choice

# Map dataset choice to name
case $dataset_choice in
  1) dataset="kinship";;
  2) dataset="umls";;
  3) dataset="Countries_S3";;
  *) echo "Invalid dataset choice!"; exit 1;;
esac

# Map model choice to script and base config
case $model_choice in
  1) 
    script="src/run_gru.py"
    config="config/${dataset}_gru.yaml"
    desc="GRU (Standard EM)"
    ;;
  2) 
    script="src/run_gru_rl.py"
    config="config/${dataset}_gru.yaml"
    desc="GRU (RL Guided)"
    ;;
  3) 
    script="src/run_mamba.py"
    config="config/${dataset}_mamba.yaml"
    desc="Mamba (Standard)"
    ;;
  4) 
    script="src/run_mamba_rl.py"
    config="config/${dataset}_mamba.yaml"
    desc="Mamba (RL Guided)"
    ;;
  5) 
    script="src/run_lstm_rl.py"
    config="config/${dataset}_lstm_rl.yaml"
    desc="LSTM (RL Guided Experiment)"
    ;;
  6) 
    script="src/run_txl.py"
    config="config/${dataset}_txl.yaml"
    desc="TransformerXL"
    ;;
  7) 
    script="src/run_gpt2.py"
    config="config/${dataset}_gpt2.yaml"
    desc="GPT-2"
    ;;
  8) 
    script="src/run_trm.py"
    config="config/${dataset}_trm.yaml"
    desc="TRM (Tiny Recursive Model)"
    ;;
  *) echo "Invalid model choice!"; exit 1;;
esac

echo ""
echo "========================================"
echo "Optional Configuration"
echo "========================================"
read -p "Enter number of layers (press Enter to use default from config): " num_layers

echo ""
echo "----------------------------------------"
echo "Running $desc on $dataset..."

# Handle execution
if [[ -n "$num_layers" ]]; then
    # validation: number check
    if ! [[ "$num_layers" =~ ^[0-9]+$ ]]; then
       echo "Error: Number of layers must be an integer."
       exit 1
    fi
    
    echo "Using custom number of layers: $num_layers"
    
    # Create temp config
    temp_config="config/tmp_run_${dataset}_${model_choice}.yaml"
    cp "$config" "$temp_config"
    
    # Update num_layers in temp config using sed
    # We target "num_layers: X" pattern. checking for any indentation.
    sed -i "s/num_layers: *[0-9]*/num_layers: $num_layers/" "$temp_config"
    
    echo "Created temporary config with modified layers: $temp_config"
    echo "Executing python $script --config $temp_config"
    echo "----------------------------------------"
    
    python "$script" --config "$temp_config"
    
    # Cleanup
    rm "$temp_config"
else
    echo "Using default configuration..."
    echo "Executing python $script --config $config"
    echo "----------------------------------------"
    
    python "$script" --config "$config"
fi

# Unified RNNLogic Project

This directory contains a unified codebase for multiple RNNLogic variants, including GRU, Mamba, LSTM, TransformerXL, GPT-2, and TRM.

## Architecture

The project is organized to share common components (`comm.py`, `data.py`, `utils.py`, `predictors.py`, `rl_guided_generator.py`) while preserving model-specific implementations where necessary.

### Directory Structure

*   `src/`: Source code for all models.
    *   **Shared**: `predictors.py`, `rl_guided_generator.py`, `value_network.py`, `data.py`, `utils.py`, `comm.py`.
    *   **GRU**: `run_gru.py`, `run_gru_rl.py`, `generators_gru.py`, `trainer_gru.py`.
    *   **LSTM**: `run_lstm_rl.py`, `generators_lstm.py`, `trainer_lstm.py`.
    *   **Mamba**: `run_mamba.py`, `run_mamba_rl.py`, `generators_mamba.py`, `trainer_mamba.py`.
    *   **TransformerXL**: `run_txl.py`, `generators_txl.py`, `trainer_txl.py`.
    *   **GPT-2**: `run_gpt2.py`, `generators_gpt2.py`.
    *   **TRM**: `run_trm.py`, `generators_trm.py`, `trainer_trm.py`, `trm.py`.
*   `config/`: Configuration files for all models and datasets (Kinship, UMLS, Countries S3).
*   `data/`: Symlink to the main data directory.
*   `run_all.sh`: Interactive bash script to run experiments.

## Usage

The easiest way to run an experiment is using the interactive script:

```bash
bash run_all.sh
```

Follow the prompts to select the model and dataset.

### Manual Usage

You can also run scripts manually. For example:

```bash
# Run GRU with RL on Kinship
python src/run_gru_rl.py --config config/kinship_gru.yaml

# Run Mamba on UMLS
python src/run_mamba.py --config config/umls_mamba.yaml

# Run GPT-2 on Countries S3
python src/run_gpt2.py --config config/Countries_S3_gpt2.yaml
```

## Notes

*   **RL Guided Generation**: The correctly implemented `rl_guided_generator.py` from the GRULogic folder is used for GRU and LSTM experiments.
*   **Predictors**: The standard `predictors.py` from GRULogic is used across all variants.
*   **Data**: All models operate on the standardized datasets found in `data/`.

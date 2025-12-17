"""
Tiny Recursion Model (TRM) - A parameter-efficient recurrent neural network.

This implementation provides a simplified recurrent cell designed for short sequences
with fewer parameters than LSTM/GRU, making it suitable for tasks where overfitting
is a concern.

The TRM uses a minimal gating mechanism:
    z_t = sigmoid(W_z @ [x_t; h_{t-1}] + b_z)
    h_candidate = tanh(W_h @ x_t + U_h @ h_{t-1} + b_h)
    h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate

This is simpler than GRU (which has reset and update gates) and much simpler than LSTM
(which has forget, input, output gates and cell state).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TRMCell(nn.Module):
    """
    A single Tiny Recursion Model cell.

    Args:
        input_dim: Size of input features
        hidden_dim: Size of hidden state
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(TRMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gate: decides how much to update vs retain previous state
        # Combines input and hidden state in one projection (more efficient)
        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Candidate hidden state computation
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        self.hidden_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one TRM cell.

        Args:
            x: Input tensor of shape (batch, input_dim)
            hidden: Hidden state of shape (batch, hidden_dim)

        Returns:
            New hidden state of shape (batch, hidden_dim)
        """
        # Compute update gate
        combined = torch.cat([x, hidden], dim=1)
        z = torch.sigmoid(self.gate(combined))

        # Compute candidate hidden state
        h_candidate = torch.tanh(
            self.input_transform(x) + self.hidden_transform(hidden)
        )

        # Update hidden state: interpolate between previous and candidate
        h_new = (1 - z) * hidden + z * h_candidate

        return h_new


class TRM(nn.Module):
    """
    Tiny Recursion Model - A parameter-efficient RNN module.

    This class provides a PyTorch RNN-compatible interface for the TRM cell,
    supporting multi-layer architectures and batch-first processing.

    Args:
        input_dim: Size of input features
        hidden_dim: Size of hidden state
        num_layers: Number of recurrent layers (default: 1)
        batch_first: If True, input shape is (batch, seq, features) (default: True)
        dropout: Dropout probability between layers (not used for single layer)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0
    ):
        super(TRM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        # Create layers
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(TRMCell(layer_input_dim, hidden_dim))

        # Dropout for multi-layer (applied to outputs between layers)
        if num_layers > 1 and dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the TRM.

        Args:
            x: Input tensor of shape:
                - (batch, seq_len, input_dim) if batch_first=True
                - (seq_len, batch, input_dim) if batch_first=False
            hidden: Initial hidden state of shape (num_layers, batch, hidden_dim)
                    If None, initializes to zeros.

        Returns:
            output: Output tensor of shape:
                - (batch, seq_len, hidden_dim) if batch_first=True
                - (seq_len, batch, hidden_dim) if batch_first=False
            hidden: Final hidden state of shape (num_layers, batch, hidden_dim)
        """
        # Handle batch_first
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to batch_first for processing

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                dtype=x.dtype, device=x.device
            )

        # Process sequence through layers
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            layer_hiddens = []

            # Process through all layers
            for layer in range(self.num_layers):
                h_prev = hidden[layer]  # (batch, hidden_dim)
                h_new = self.cells[layer](x_t, h_prev)
                layer_hiddens.append(h_new)

                # For next layer, input is current layer's output
                if layer < self.num_layers - 1:
                    x_t = self.dropout_layer(h_new) if self.dropout_layer else h_new
                else:
                    x_t = h_new

            outputs.append(x_t)
            # Update hidden state for next timestep (create new tensor, not inplace)
            hidden = torch.stack(layer_hiddens, dim=0)

        # Stack outputs: (batch, seq_len, hidden_dim)
        output = torch.stack(outputs, dim=1)

        # Convert back if not batch_first
        if not self.batch_first:
            output = output.transpose(0, 1)  # (seq_len, batch, hidden_dim)

        return output, hidden

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compare_with_lstm(input_dim: int, hidden_dim: int, num_layers: int = 1) -> dict:
    """
    Compare parameter counts between TRM and LSTM.

    Args:
        input_dim: Size of input features
        hidden_dim: Size of hidden state
        num_layers: Number of layers

    Returns:
        Dictionary with parameter counts and reduction percentage
    """
    trm = TRM(input_dim, hidden_dim, num_layers)
    lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    trm_params = trm.count_parameters()
    lstm_params = sum(p.numel() for p in lstm.parameters() if p.requires_grad)

    reduction = (1 - trm_params / lstm_params) * 100

    return {
        'trm_parameters': trm_params,
        'lstm_parameters': lstm_params,
        'reduction_percentage': reduction,
        'ratio': trm_params / lstm_params
    }


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("TRM Implementation - Quick Test")
    print("=" * 60)

    # Test configuration
    batch_size = 32
    seq_len = 5
    input_dim = 1024
    hidden_dim = 256
    num_layers = 1

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")

    # Create model
    trm = TRM(input_dim, hidden_dim, num_layers, batch_first=True)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    output, hidden = trm(x)

    print(f"\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Hidden: {hidden.shape}")

    # Compare with LSTM
    print(f"\nParameter comparison:")
    comparison = compare_with_lstm(input_dim, hidden_dim, num_layers)
    print(f"  TRM parameters: {comparison['trm_parameters']:,}")
    print(f"  LSTM parameters: {comparison['lstm_parameters']:,}")
    print(f"  Reduction: {comparison['reduction_percentage']:.1f}%")
    print(f"  TRM/LSTM ratio: {comparison['ratio']:.3f}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

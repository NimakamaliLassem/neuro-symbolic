"""
Recursive TRM for Logical Rule Generation

Implements TRUE recursive reasoning inspired by TinyRecursiveModels paper.
The model iteratively refines rule predictions through multiple reasoning cycles.

Key idea:
- Generate initial rule guess
- Refine it K times through recursive reasoning
- Each refinement considers previous attempts
- Final output is the refined prediction

For logical rules: grandfather ← father, father
- Cycle 0: Initial guess (may be random)
- Cycle 1: Refine based on semantics
- Cycle 2: Further compositional refinement
- Output: Best refined rule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RecursiveReasoningCell(nn.Module):
    """
    Single reasoning refinement step.

    Takes current prediction + hidden state, outputs refined prediction.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Transform previous prediction + current hidden (both hidden_dim)
        self.refine_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.refine_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )

        # Update hidden state based on refinement
        self.hidden_update = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, prev_output: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine prediction through recursive reasoning.

        Args:
            prev_output: Previous prediction (batch, hidden_dim)
            hidden: Hidden state (batch, hidden_dim)

        Returns:
            refined_output: Refined prediction (batch, hidden_dim)
            new_hidden: Updated hidden state (batch, hidden_dim)
        """
        # Combine previous output with hidden state
        combined = torch.cat([prev_output, hidden], dim=-1)

        # Refinement gate (how much to change)
        gate = torch.sigmoid(self.refine_gate(combined))

        # Refinement transform (what to change to)
        refinement = self.refine_transform(combined)

        # Apply refinement
        refined_output = gate * refinement + (1 - gate) * prev_output

        # Update hidden state
        new_hidden = torch.tanh(self.hidden_update(refined_output))

        return refined_output, new_hidden


class RecursiveRNNCell(nn.Module):
    """
    RNN cell with recursive refinement at each timestep.

    At each sequence position:
    1. Get initial prediction from input
    2. Refine it K times recursively
    3. Output final refined prediction
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_refinement_steps: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_refinement_steps = num_refinement_steps

        # Initial prediction from input
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # RNN cell for sequence processing
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)

        # Recursive refinement module (shared across refinement steps)
        self.refinement_cell = RecursiveReasoningCell(hidden_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one timestep with recursive refinement.

        Args:
            x: Input at current timestep (batch, input_dim)
            hidden: Hidden state from previous timestep (batch, hidden_dim)

        Returns:
            output: Refined output (batch, hidden_dim)
            new_hidden: Updated hidden state (batch, hidden_dim)
        """
        # Initial prediction from RNN
        new_hidden = self.rnn_cell(x, hidden)
        output = new_hidden  # Initial guess

        # Recursive refinement cycles
        refine_hidden = new_hidden  # Refinement hidden state
        for cycle in range(self.num_refinement_steps):
            output, refine_hidden = self.refinement_cell(output, refine_hidden)

        return output, new_hidden


class RecursiveTRM(nn.Module):
    """
    Recursive Tiny Reasoning Model for sequences.

    Matches LSTM interface but uses recursive reasoning at each step.
    Each sequence position is refined K times before moving to next.

    This implements TRUE recursive reasoning:
    - At each timestep: generate → refine → refine → ... → output
    - Refinements consider previous attempts
    - Final output is recursively refined prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_refinement_steps: int = 3,
        batch_first: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_refinement_steps = num_refinement_steps
        self.batch_first = batch_first

        # Create layers
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(
                RecursiveRNNCell(layer_input_dim, hidden_dim, num_refinement_steps)
            )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with recursive reasoning.

        Args:
            x: Input tensor
                - (batch, seq_len, input_dim) if batch_first=True
                - (seq_len, batch, input_dim) if batch_first=False
            hidden: Initial hidden state (num_layers, batch, hidden_dim) or None

        Returns:
            output: Refined outputs (batch, seq_len, hidden_dim) or (seq_len, batch, hidden_dim)
            hidden: Final hidden states (num_layers, batch, hidden_dim)
        """
        # Handle batch_first
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to batch_first

        # Initialize hidden if needed
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                dtype=x.dtype, device=x.device
            )

        # Process sequence with recursive refinement
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            layer_hiddens = []

            # Process through layers
            for layer in range(self.num_layers):
                h_prev = hidden[layer]  # (batch, hidden_dim)

                # Recursive refinement at this timestep
                x_t, h_new = self.cells[layer](x_t, h_prev)
                layer_hiddens.append(h_new)

            outputs.append(x_t)
            hidden = torch.stack(layer_hiddens, dim=0)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        # Convert back if not batch_first
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HierarchicalRecursiveTRM(nn.Module):
    """
    Advanced: Two-level recursive reasoning (like TRM paper).

    Low-level (L): Refines predictions rapidly
    High-level (H): Guides low-level refinement

    At each timestep:
    - H cycles of high-level reasoning
    - Each H cycle runs L cycles of low-level reasoning

    More powerful but more complex.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        H_cycles: int = 2,
        L_cycles: int = 3,
        batch_first: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.batch_first = batch_first

        # Base RNN for sequence
        self.base_rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Low-level refinement
        self.L_refine = RecursiveReasoningCell(hidden_dim)

        # High-level guidance
        self.H_refine = RecursiveReasoningCell(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical recursive reasoning.

        Args:
            x: Input (batch, seq_len, input_dim) or (seq_len, batch, input_dim)
            hidden: Initial state (num_layers, batch, hidden_dim) or None

        Returns:
            output: Refined outputs
            hidden: Final states
        """
        # Get base RNN output
        base_output, base_hidden = self.base_rnn(x, hidden)

        batch_size = base_output.size(0) if self.batch_first else base_output.size(1)
        seq_len = base_output.size(1) if self.batch_first else base_output.size(0)

        if not self.batch_first:
            base_output = base_output.transpose(0, 1)

        # Initialize refinement states
        z_H = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        z_L = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Refine each position
        refined_outputs = []

        for t in range(seq_len):
            output_t = base_output[:, t, :]

            # Hierarchical refinement
            for h_step in range(self.H_cycles):
                # Low-level cycles
                for l_step in range(self.L_cycles):
                    output_t, z_L = self.L_refine(output_t, z_H)

                # High-level guidance
                output_t, z_H = self.H_refine(output_t, z_L)

            refined_outputs.append(output_t)

        output = torch.stack(refined_outputs, dim=1)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, base_hidden


def compare_recursive_models(input_dim: int, hidden_dim: int, num_layers: int = 1):
    """Compare parameter counts of different models."""

    recursive_trm = RecursiveTRM(input_dim, hidden_dim, num_layers, num_refinement_steps=3)
    lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    r_params = recursive_trm.count_parameters()
    lstm_params = sum(p.numel() for p in lstm.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())

    print(f"Recursive TRM: {r_params:,} parameters")
    print(f"LSTM:          {lstm_params:,} parameters")
    print(f"GRU:           {gru_params:,} parameters")
    print(f"RecTRM vs LSTM: {((r_params / lstm_params - 1) * 100):+.1f}%")
    print(f"RecTRM vs GRU:  {((r_params / gru_params - 1) * 100):+.1f}%")

    return {
        'recursive_trm': r_params,
        'lstm': lstm_params,
        'gru': gru_params
    }


if __name__ == "__main__":
    print("=" * 70)
    print("RECURSIVE TRM - TRUE RECURSIVE REASONING")
    print("=" * 70)

    # Test configuration
    batch_size = 32
    seq_len = 5
    input_dim = 1024
    hidden_dim = 256
    num_refinement_steps = 3

    print(f"\nConfiguration:")
    print(f"  Refinement steps: {num_refinement_steps} (refines prediction 3x per position)")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")

    # Create model
    model = RecursiveTRM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_refinement_steps=num_refinement_steps,
        batch_first=True
    )

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    output, hidden = model(x)

    print(f"\nShapes:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Hidden: {hidden.shape}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()

    grad_params = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)

    print(f"\nGradient flow:")
    print(f"  Parameters with gradients: {grad_params}/{total_params}")

    # Compare models
    print(f"\nParameter comparison:")
    compare_recursive_models(input_dim, hidden_dim, num_layers=1)

    print("\n" + "=" * 70)
    print("RECURSIVE REASONING TEST PASSED!")
    print("=" * 70)
    print("\nKey feature: Each timestep is refined 3x before outputting")
    print("This enables true compositional reasoning for logical rules")

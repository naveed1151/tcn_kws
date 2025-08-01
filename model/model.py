import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedTCN(nn.Module):
    def __init__(self, input_channels, num_layers, hidden_channels, kernel_size=3, num_classes=1, dropout=0.0):
        """
        Dilated Temporal Convolutional Network for sequence modeling.

        Args:
            input_channels (int): Number of input channels/features per timestep (e.g. MFCC coeffs).
            num_layers (int): Number of TCN layers.
            hidden_channels (int): Number of channels in hidden TCN layers.
            kernel_size (int): Kernel size for convolution (usually small like 3).
            num_classes (int): Number of output classes. Use 1 for binary classification.
            dropout (float): Dropout probability applied to the pooled feature vector before the classifier.
        """
        super(DilatedTCN, self).__init__()

        self.num_layers = num_layers
        self.tcn_layers = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else hidden_channels
            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation
            )
            self.tcn_layers.append(conv)

        # Dropout before the final classifier; no-op if dropout == 0
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        # Final linear layer to map hidden_channels to num_classes
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_channels, sequence_length)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        out = x
        for conv in self.tcn_layers:
            out = conv(out)
            # Remove padding on right side to keep causal conv shape
            padding_amount = conv.padding[0]
            if padding_amount > 0:
                out = out[:, :, :-padding_amount]
            out = F.relu(out)

        # Global average pooling over time dimension
        out = out.mean(dim=2)  # shape: (batch_size, hidden_channels)

        # Apply dropout before the classifier
        out = self.dropout(out)

        logits = self.fc(out)  # shape: (batch_size, num_classes)
        return logits

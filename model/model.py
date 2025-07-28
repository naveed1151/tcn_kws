import numpy as np

def relu(x):
    return np.maximum(0, x)

def temporal_conv1d(x, weights, bias, dilation=1):
    """
    Causal dilated 1D convolution.

    Args:
        x: input (T, C_in)
        weights: (K, C_in, C_out)
        bias: (C_out,)
        dilation: dilation factor

    Returns:
        output (T, C_out)
    """
    T, C_in = x.shape
    K, _, C_out = weights.shape
    pad = dilation * (K - 1)
    x_padded = np.pad(x, ((pad, 0), (0, 0)), mode='constant')

    out = np.zeros((T, C_out))
    for t in range(T):
        for k in range(K):
            t_in = t + pad - dilation * k
            if t_in >= 0:
                out[t] += x_padded[t_in] @ weights[k]
        out[t] += bias
    return out

class DilatedTCN:
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, num_layers=4):
        """
        Multi-layer dilated TCN with exponentially increasing dilation.

        Args:
            in_channels: input channels (e.g. 16 for MFCC)
            hidden_channels: channels in hidden layers
            out_channels: final embedding size
            kernel_size: convolution kernel size (usually 3)
            num_layers: number of TCN layers
        """
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.weights = []
        self.biases = []

        # Initialize weights for each layer
        for l in range(num_layers):
            dilation = 2 ** l
            in_c = in_channels if l == 0 else hidden_channels
            out_c = out_channels if l == num_layers - 1 else hidden_channels
            w = np.random.randn(kernel_size, in_c, out_c) * np.sqrt(2 / in_c)
            b = np.zeros(out_c)
            self.weights.append((w, dilation))
            self.biases.append(b)

    def __call__(self, x):
        """
        Forward pass through all layers.

        Args:
            x: input array (T, in_channels)

        Returns:
            embedding vector (out_channels,)
        """
        h = x
        for i in range(self.num_layers):
            w, dilation = self.weights[i]
            b = self.biases[i]
            h = temporal_conv1d(h, w, b, dilation=dilation)
            h = relu(h)
        # Global average pooling over time axis
        return np.mean(h, axis=0)
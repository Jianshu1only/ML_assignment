import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(BinaryConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Define the real-valued weights, which will be binarized in the forward pass
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def binarize(self, x):
        """ Binarize the input (weights) """
        return x.sign()

    def forward(self, x):
        # Binarize the weights
        binary_weights = self.binarize(self.weights)

        # Perform the convolution using binary weights
        return F.conv2d(x, binary_weights, self.bias, self.stride, self.padding)

# Example usage
binary_conv = BinaryConv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor
output = binary_conv(input_tensor)

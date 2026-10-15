import torch
import torch.nn as nn

def build_pytorch_mlp(
    input_dim,
    hidden_dim,
    output_dim,
    act_layer=nn.GELU,
    depth=10,
    bias=False,
    use_weight_norm=False,
) -> nn.Sequential:
    """Build a PyTorch MLP with configurable depth and options.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        act_layer: Activation layer class (default: nn.GELU)
        depth: Number of hidden layers (0 for direct linear projection)
        bias: Whether to use bias in linear layers
        use_weight_norm: Whether to apply weight normalization
    
    Returns:
        nn.Sequential: The constructed MLP
    """
    if depth == 0:
        mlp = [nn.Linear(input_dim, output_dim, bias=bias)]
    else:
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        mlp.append(act_layer())
        for _ in range(depth - 1):
            mlp.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            mlp.append(act_layer())
        mlp.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    if use_weight_norm:
        mlp = [
            torch.nn.utils.weight_norm(layer) if isinstance(layer, nn.Linear) else layer
            for layer in mlp
        ]
    mlp = nn.Sequential(*mlp)
    return mlp

class GSAdaptor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        return x
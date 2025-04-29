import torch.nn as nn
from torch import Tensor


class ChatBotModel(nn.Module):
    """
    Defines a simple feed-forward PyTorch model:
    Linear → ReLU → Linear
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

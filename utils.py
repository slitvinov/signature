import torch
import torchcde
import math


class View(torch.nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


class NeuralCDE(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.func = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, HIDDEN_LAYER_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_WIDTH,
                            input_channels * hidden_channels), torch.nn.Tanh(),
            View(hidden_channels, input_channels))
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, X):
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=lambda t, z: self.func(z),
                              t=X.interval)
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y

def get_data2d(num_timepoints):
    t = torch.linspace(0., 4 * math.pi, num_timepoints)
    start = torch.rand(HIDDEN_LAYER_WIDTH) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:HIDDEN_LAYER_WIDTH // 2] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    X = torch.stack([x_pos, y_pos], dim=2)
    y = torch.zeros(HIDDEN_LAYER_WIDTH)
    y[:HIDDEN_LAYER_WIDTH // 2] = 1
    perm = torch.randperm(HIDDEN_LAYER_WIDTH)
    return X[perm], y[perm]


def get_data3d(num_timepoints):
    t = torch.linspace(0., 4 * math.pi, num_timepoints)
    start = torch.rand(HIDDEN_LAYER_WIDTH) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:HIDDEN_LAYER_WIDTH // 2] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    X = torch.stack(
        [t.unsqueeze(0).repeat(HIDDEN_LAYER_WIDTH, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(HIDDEN_LAYER_WIDTH)
    y[:HIDDEN_LAYER_WIDTH // 2] = 1
    perm = torch.randperm(HIDDEN_LAYER_WIDTH)
    return X[perm], y[perm]

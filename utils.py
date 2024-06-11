import torch
import torchcde
import math


class CDEFunc(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, HIDDEN_LAYER_WIDTH)
        self.linear2 = torch.nn.Linear(HIDDEN_LAYER_WIDTH,
                                       input_channels * hidden_channels)

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(torch.nn.Module):

    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels,
                 interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.NaturalCubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(
                "Only 'linear' and 'cubic' interpolation methods are implemented."
            )
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval)
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


def get_data(num_timepoints=100):
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
    X = X[perm]
    y = y[perm]
    return X, y
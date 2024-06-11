import sys
import math
import numpy as np
import torch
import torchcde
import matplotlib.pyplot as plt

HIDDEN_LAYER_WIDTH = 64
NUM_EPOCHS = 10
BATCH_SIZE = 32


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


train_X, train_y = get_data()
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
idx, = np.where(train_y == 0)
ax1.plot(train_X[idx[0], :, 1], train_X[idx[0], :, 2])
ax1.set_title("Anticlockwise spiral")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

idx, = np.where(train_y == 1)
ax2.plot(train_X[idx[0], :, 1], train_X[idx[0], :, 2])
ax2.set_title("Clockwise spiral")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
plt.show()

model = NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1)
optimizer = torch.optim.Adam(model.parameters())
train_coeffs = torchcde.natural_cubic_coeffs(train_X)
train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE)
for epoch in range(NUM_EPOCHS):
    for batch in train_dataloader:
        batch_coeffs, batch_y = batch
        pred_y = model(batch_coeffs).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_y, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('Epoch: {:2d}   Training loss: {}'.format(epoch, loss.item()))

test_X, test_y = get_data()
test_coeffs = torchcde.natural_cubic_coeffs(test_X)
pred_y = model(test_coeffs).squeeze(-1)
binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
proportion_correct = prediction_matches.sum() / test_y.size(0)
print('Test Accuracy: {}'.format(proportion_correct))

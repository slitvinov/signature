import math
import time
import torch
import torchcde
HIDDEN_LAYER_WIDTH = 64
NUM_EPOCHS = 10
NUM_TIMEPOINTS = 5000
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, HIDDEN_LAYER_WIDTH)
        self.linear2 = torch.nn.Linear(HIDDEN_LAYER_WIDTH, input_channels * hidden_channels)
    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
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
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


def get_data(num_timepoints=100):
    t = torch.linspace(0., 4 * math.pi, num_timepoints)

    start = torch.rand(HIDDEN_LAYER_WIDTH) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:HIDDEN_LAYER_WIDTH//2] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    X = torch.stack([t.unsqueeze(0).repeat(HIDDEN_LAYER_WIDTH, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(HIDDEN_LAYER_WIDTH)
    y[:HIDDEN_LAYER_WIDTH//2] = 1
    perm = torch.randperm(HIDDEN_LAYER_WIDTH)
    X = X[perm]
    y = y[perm]
    return X, y

def train_and_evaluate(train_X, train_y, test_X, test_y, depth, num_epochs, window_length):
    start_time = time.time()
    train_logsig = torchcde.logsig_windows(train_X, depth, window_length=window_length)
    print("Logsignature shape: {}".format(train_logsig.size()))
    model = NeuralCDE(
        input_channels=train_logsig.size(-1), hidden_channels=8, output_channels=1, interpolation="linear"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    train_coeffs = torchcde.linear_interpolation_coeffs(train_logsig)
    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch: {}   Training loss: {}".format(epoch, loss.item()))
    test_logsig = torchcde.logsig_windows(test_X, depth, window_length=window_length)
    test_coeffs = torchcde.linear_interpolation_coeffs(test_logsig)
    pred_y = model(test_coeffs).squeeze(-1)
    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print("Test Accuracy: {}".format(proportion_correct))
    elapsed = time.time() - start_time
    return proportion_correct, elapsed

train_X, train_y = get_data(num_timepoints=NUM_TIMEPOINTS)
test_X, test_y = get_data(num_timepoints=NUM_TIMEPOINTS)
depths = [1, 2, 3]
window_length = 50
accuracies = []
training_times = []
for depth in depths:
    print(f'Running for logsignature depth: {depth}')
    acc, elapsed = train_and_evaluate(
        train_X, train_y, test_X, test_y, depth, NUM_EPOCHS, window_length
    )
    training_times.append(elapsed)
    accuracies.append(acc)
print("Final results")
for acc, elapsed, depth in zip(accuracies, training_times, depths):
    print(
        f"Depth: {depth}\n\tAccuracy on test set: {acc*100:.1f}%\n\tTime per epoch: {elapsed/NUM_EPOCHS:.1f}s"
    )

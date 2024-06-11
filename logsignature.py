import time
import torch
import torchcde
import utils


def train_and_evaluate(train_X, train_y, test_X, test_y, depth, num_epochs,
                       window_length):
    start_time = time.time()
    train_logsig = torchcde.logsig_windows(train_X,
                                           depth,
                                           window_length=window_length)
    print("Logsignature shape: {}".format(train_logsig.size()))
    model = utils.NeuralCDE(input_channels=train_logsig.size(-1),
                            hidden_channels=8,
                            output_channels=1,
                            interpolation="linear")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    train_coeffs = torchcde.linear_interpolation_coeffs(train_logsig)
    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=BATCH_SIZE)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch: {}   Training loss: {}".format(epoch, loss.item()))
    test_logsig = torchcde.logsig_windows(test_X,
                                          depth,
                                          window_length=window_length)
    test_coeffs = torchcde.linear_interpolation_coeffs(test_logsig)
    pred_y = model(test_coeffs).squeeze(-1)
    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print("Test Accuracy: {}".format(proportion_correct))
    elapsed = time.time() - start_time
    return proportion_correct, elapsed


utils.HIDDEN_LAYER_WIDTH = 64
NUM_EPOCHS = 10
BATCH_SIZE = 32
NUM_TIMEPOINTS = 5000
train_X, train_y = utils.get_data(num_timepoints=NUM_TIMEPOINTS)
test_X, test_y = utils.get_data(num_timepoints=NUM_TIMEPOINTS)
depths = [1, 2, 3]
window_length = 50
accuracies = []
training_times = []
for depth in depths:
    print(f'Running for logsignature depth: {depth}')
    acc, elapsed = train_and_evaluate(train_X, train_y, test_X, test_y, depth,
                                      NUM_EPOCHS, window_length)
    training_times.append(elapsed)
    accuracies.append(acc)
print("Final results")
for acc, elapsed, depth in zip(accuracies, training_times, depths):
    print(
        f"Depth: {depth}\n\tAccuracy on test set: {acc*100:.1f}%\n\tTime per epoch: {elapsed/NUM_EPOCHS:.1f}s"
    )

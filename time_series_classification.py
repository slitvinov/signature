import math
import numpy as np
import torch
import torchcde
import utils
import matplotlib.pyplot as plt

utils.HIDDEN_LAYER_WIDTH = 64
NUM_EPOCHS = 10
BATCH_SIZE = 32

train_X, train_y = utils.get_data()
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

model = utils.NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1)
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

test_X, test_y = utils.get_data()
test_coeffs = torchcde.natural_cubic_coeffs(test_X)
pred_y = model(test_coeffs).squeeze(-1)
binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
proportion_correct = prediction_matches.sum() / test_y.size(0)
print('Test Accuracy: {}'.format(proportion_correct))

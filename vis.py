import utils
import numpy as np
import matplotlib.pyplot as plt

utils.HIDDEN_LAYER_WIDTH = 64
NUM_TIMEPOINTS = 200
X, Y = utils.get_data2d(NUM_TIMEPOINTS)
idx, = np.where(Y == 0)
*rest, x, y = X[idx[0]].T
plt.axis("equal")
plt.plot(x, y, 'r-')
plt.plot([x[0]], [y[0]], 'ro')

idx, = np.where(Y == 1)
*rest, x, y = X[idx[0]].T
plt.plot(x, y, 'b-')
plt.plot([x[0]], [y[0]], 'bo')
plt.savefig("spiral.png")

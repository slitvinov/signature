import utils
import numpy as np
import matplotlib.pyplot as plt

utils.HIDDEN_LAYER_WIDTH = 64
NUM_TIMEPOINTS = 100
X, y = utils.get_data0(NUM_TIMEPOINTS)
idx, = np.where(y == 0)
plt.plot(X[idx[0], :, 0], X[idx[0], :, 1], 'o-')

idx, = np.where(y == 1)
plt.plot(X[idx[0], :, 0], X[idx[0], :, 1], 'x-')
plt.show()

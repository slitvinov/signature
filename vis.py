import utils
import numpy as np
import matplotlib.pyplot as plt

utils.HIDDEN_LAYER_WIDTH = 64

X, y = utils.get_data()
idx, = np.where(y == 0)
plt.plot(X[idx[0], :, 1], X[idx[0], :, 2], 'o-')

idx, = np.where(y == 1)
plt.plot(X[idx[0], :, 1], X[idx[0], :, 2], 'x-')
plt.show()

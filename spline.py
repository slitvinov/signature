import utils
import torchcde
import scipy.interpolate
import numpy as np

utils.HIDDEN_LAYER_WIDTH = 64
NUM_TIMEPOINTS = 100
X, y = utils.get_data(NUM_TIMEPOINTS)

cs = torchcde.natural_cubic_coeffs(X[0])
cs0 = scipy.interpolate.CubicSpline(range(NUM_TIMEPOINTS),
                                    X[0],
                                    bc_type='natural')
cs0 = cs0.c
print(cs)
print(cs0.c)

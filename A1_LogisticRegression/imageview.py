import matplotlib.pyplot as plt
from lr_utils import load_dataset
import numpy as np

train_set_x, train_set_y, test_x_og, test_y_og, classes = load_dataset()

# plt.imshow(test_x_og[25])
# plt.show()

# plt.imshow(np.rot90(test_x_og[25], 1))
# plt.show()
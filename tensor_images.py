import tensorflow as tf

(cifar_x, cifar_y), _ = tf.keras.datasets.cifar10.load_data()
print(cifar_y[0:10])

import matplotlib.pyplot as plt
plt.imshow(cifar_x[0])
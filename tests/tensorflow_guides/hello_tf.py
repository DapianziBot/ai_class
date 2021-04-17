import tensorflow as tf

print(f'TensorFlow Version: {tf.__version__}, GPU info: {tf.config.list_physical_devices("GPU")}')

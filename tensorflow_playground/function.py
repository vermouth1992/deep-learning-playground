import tensorflow as tf
import numpy as np




@tf.function
def add_step(a, b):
    return tf.reduce_mean(a + b)

def add(a, b):
    return add_step(a, b).numpy()


if __name__ == '__main__':
    a = add(tf.random.normal([10, 4]), tf.random.normal([10, 4]))
    b = add(np.random.normal(size=[10, 4]), np.random.normal(size=[10, 4]))

import tensorflow as tf


def add(a, b):
    @tf.function
    def add_step(a, b):
        print('Compile graph')
        return tf.reduce_mean(a + b)

    return add_step(a, b).numpy()


@tf.function
def condition(state, action=None):
    if action is None:
        print('Building path true')
        return state
    else:
        print('Building path false')
        return state, action


class Foo(object):
    def __init__(self):
        self.shape = [None, 4]

        self.add_constant = self.tf_decorator(self.add_constant)

    @property
    def tf_decorator(self):
        decorator = lambda func: tf.function(func=func, input_signature=[tf.TensorSpec(shape=self.shape)])
        return decorator

    def add_constant(self, a):
        print('Building graph')
        return a + 5


@tf.function
def random_selection():
    if tf.random.uniform(shape=()) < 0.5:
        print('Return 1')
        return 1
    else:
        print('Return 0')
        return 0


@tf.function
def fix_selection():
    if True:
        print('Return 1')
        return 1
    else:
        print('Return 0')
        return 0


class DummyModel(tf.keras.Model):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.var = tf.Variable(initial_value=tf.random.normal([10, 4]))
        self.build(input_shape=[None])

    def call(self, inputs, training=None, mask=None):
        return self.var


if __name__ == '__main__':
    model1 = DummyModel()
    model2 = DummyModel()

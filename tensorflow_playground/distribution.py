import tensorflow as tf
import tensorflow_probability as tfp

model = tf.keras.Sequential([
    tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfp.distributions.Normal(
            loc=0., scale=tf.exp(0.)),
        convert_to_tensor_fn=lambda s: s.sample(5))
])
